import numpy as np
from collections import deque
from scipy.spatial import cKDTree
from fea_solver_3d import FEASolver3D
from density_filter_3d import MinimumMemberSizeProjection3D, auto_filter_radius_from_mesh


class BESO3DOptimizer:
    """3D BESO optimizer with sensitivity averaging and auto-stop."""

    def __init__(self, nodes, elements, material, config):
        self.fea = FEASolver3D(nodes, elements, material)
        self.config = config
        try:
            self.fea.configure_linear_solver(
                mode=config.get('linear_solver', 'auto'),
                iterative_threshold_dofs=int(config.get('iterative_solver_dof_threshold', 50000)),
                cg_tol=float(config.get('iterative_solver_tol', 1e-8)),
                cg_maxiter=int(config.get('iterative_solver_maxiter', 2000)),
                use_pyamg=bool(config.get('use_pyamg_preconditioner', True)),
            )
        except Exception:
            pass
        self.material = material
        self.n_elements = int(elements.shape[0])

        objective = str(config.get('objective', config.get('objective_function', 'compliance'))).lower()
        self.volfrac = float(config.get('volume_fraction', config.get('volfrac', 0.5)))
        if objective == 'weight':
            wr = config.get('weight_reduction_percent', config.get('target_weight_reduction_percent', None))
            if wr is not None:
                try:
                    self.volfrac = 1.0 - float(wr) / 100.0
                except Exception:
                    pass
        self.volfrac = float(np.clip(self.volfrac, 0.05, 1.0))

        self.max_iterations = int(config.get('max_iterations_auto', config.get('iterations', 200)))
        self.min_iterations = int(max(10, config.get('min_iterations_auto', 60)))
        self.change_tol = float(config.get('density_change_tolerance', 1e-3))
        self.comp_tol = float(config.get('compliance_tolerance', 5e-4))
        self.stall_patience = int(max(3, config.get('stall_patience', 8)))

        self.er = float(config.get('evolutionary_rate', 0.02))
        self.ert = float(config.get('target_er', 0.01))
        self.addback_alpha = float(np.clip(config.get('beso_addback_alpha', 0.95), 0.0, 1.0))
        self.auto_filter_radius = bool(config.get('auto_filter_radius', True))
        self.filter_radius_factor = float(config.get('filter_radius_factor', 3.5))
        if self.auto_filter_radius or ('filter_radius' not in config):
            self.filter_radius, self.element_size = auto_filter_radius_from_mesh(
                self.fea.nodes,
                self.fea.elements,
                factor=self.filter_radius_factor,
            )
        else:
            self.filter_radius = float(max(config.get('filter_radius', 1.0), 1e-9))
            self.element_size = np.nan
        self.penalty = float(config.get('penalization', config.get('penalty', 3.0)))
        # BESO uses binary (0/1) elements — FEA should use penalty=1.0 to avoid
        # distorting compliance on a binary field where SIMP interpolation is unnecessary.
        self.fea_penalty = 1.0
        self.rho_min = float(config.get('rho_min', getattr(material, 'rho_min', 1e-6)))
        self.rho_min = float(np.clip(self.rho_min, 1e-9, 5e-2))

        self.use_min_member_projection = bool(config.get('use_min_member_projection_beso', config.get('use_min_member_projection', False)))
        self.min_member_size_factor = float(config.get('min_member_size_factor', 1.5))
        min_member_size_mm = config.get('min_member_size_mm', None)
        if min_member_size_mm is not None and np.isfinite(self.element_size) and self.element_size > 0.0:
            try:
                self.min_member_size_factor = max(self.min_member_size_factor, float(min_member_size_mm) / float(self.element_size))
            except Exception:
                pass
        self.proj_beta_start = float(config.get('min_member_projection_beta_start', 1.0))
        self.proj_beta_end = float(config.get('min_member_projection_beta_end', config.get('min_member_projection_beta', 32.0)))
        self.member_projector = None
        if self.use_min_member_projection:
            self.member_projector = MinimumMemberSizeProjection3D(
                self.fea.nodes,
                self.fea.elements,
                self.filter_radius,
                size_factor=self.min_member_size_factor,
                beta=self.proj_beta_start,
            )

        passive = config.get('passive_solid_elements', None)
        self.passive_solid = np.zeros(self.n_elements, dtype=bool)
        if passive is not None:
            p = np.asarray(passive, dtype=bool).reshape(-1)
            if p.size == self.n_elements:
                self.passive_solid = p.copy()

        passive_frac = float(np.mean(self.passive_solid)) if self.passive_solid.size else 0.0
        min_target = passive_frac + (1.0 / max(self.n_elements, 1))
        self.volfrac_eff = float(np.clip(max(self.volfrac, min_target), 0.05, 1.0))

        self.centroids = np.mean(self.fea.nodes[self.fea.elements], axis=1)
        self._build_filter_map()
        self._build_element_adjacency()

        elems = np.asarray(self.fea.elements, dtype=np.int64)
        npe = int(elems.shape[1])
        self.elem_dofs = (3 * elems[:, :, None] + np.arange(3, dtype=np.int64)).reshape(self.n_elements, 3 * npe)
        rho_probe = np.ones(self.n_elements, dtype=np.float64)
        self.ke0 = np.stack([self.fea.get_element_stiffness(i, rho_probe, penalty=1.0) for i in range(self.n_elements)], axis=0)

    def _build_filter_map(self):
        tree = cKDTree(self.centroids)
        self.neighbors = []
        self.weights = []
        for i in range(self.n_elements):
            ids = tree.query_ball_point(self.centroids[i], self.filter_radius)
            ids = np.asarray(ids, dtype=np.int64)
            d = np.linalg.norm(self.centroids[ids] - self.centroids[i], axis=1)
            w = np.maximum(0.0, self.filter_radius - d)
            s = float(np.sum(w))
            if s <= 0.0:
                ids = np.asarray([i], dtype=np.int64)
                w = np.asarray([1.0], dtype=np.float64)
            else:
                w = w / s
            self.neighbors.append(ids)
            self.weights.append(w)

    def _projection_beta(self, iteration):
        b0 = max(self.proj_beta_start, 1.0)
        b1 = max(self.proj_beta_end, b0)
        if self.max_iterations <= 1:
            return b1

        use_staged = bool(self.config.get('use_staged_beta_schedule', True))
        if not use_staged:
            t = float(iteration) / float(max(self.max_iterations - 1, 1))
            return b0 * ((b1 / b0) ** t)

        levels = [b0]
        while levels[-1] < b1:
            nxt = min(levels[-1] * 2.0, b1)
            if abs(nxt - levels[-1]) < 1e-12:
                break
            levels.append(nxt)

        stage_len = int(np.ceil(float(self.max_iterations) / float(max(len(levels), 1))))
        stage_idx = int(np.clip(iteration // max(stage_len, 1), 0, len(levels) - 1))
        return float(levels[stage_idx])

    def _build_element_adjacency(self):
        """Build element-element adjacency via shared nodes for connectivity checks."""
        elems = np.asarray(self.fea.elements, dtype=np.int64)
        n_nodes = int(self.fea.nodes.shape[0])
        node_to_elems = [[] for _ in range(n_nodes)]
        for eidx in range(self.n_elements):
            for nid in elems[eidx]:
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    node_to_elems[ni].append(eidx)
        self._elem_adj = []
        for eidx in range(self.n_elements):
            nset = set()
            for nid in elems[eidx]:
                nset.update(node_to_elems[int(nid)])
            nset.discard(eidx)
            self._elem_adj.append(np.asarray(sorted(nset), dtype=np.int64))

    def _enforce_connectivity(self, rho, fixed_dofs, forces):
        """Ensure load paths remain connected between supports and loads.

        After binary thresholding, checks if all loaded elements can reach
        support elements through solid elements. If disconnected, restores
        bridging elements and their immediate neighbors to form a viable
        load path (preventing 1-element-wide bottlenecks).
        """
        elems = self.fea.elements

        # Identify support and load nodes
        fd = np.asarray(fixed_dofs, dtype=np.int64)
        support_nodes = np.unique(fd // 3)
        f3 = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        load_nodes = np.where(np.max(np.abs(f3), axis=1) > 1e-30)[0]

        if support_nodes.size == 0 or load_nodes.size == 0:
            return rho

        # Map nodes to elements
        support_mask = np.any(np.isin(elems, support_nodes), axis=1)
        load_mask = np.any(np.isin(elems, load_nodes), axis=1)

        # Keep support and load elements solid
        rho[support_mask] = 1.0
        rho[load_mask] = 1.0

        support_set = set(np.where(support_mask)[0].tolist())
        load_set = set(np.where(load_mask)[0].tolist())

        total_restored = 0
        max_bridge_passes = 5  # Handle multiple disconnected components

        for _pass in range(max_bridge_passes):
            is_solid = rho > 0.5

            # BFS from support elements through solid elements
            reachable = np.zeros(self.n_elements, dtype=bool)
            queue = deque()
            for e in support_set:
                if is_solid[e]:
                    reachable[e] = True
                    queue.append(e)
            while queue:
                e = queue.popleft()
                for nb in self._elem_adj[e]:
                    if not reachable[nb] and is_solid[nb]:
                        reachable[nb] = True
                        queue.append(nb)

            # Check if all load elements are reachable
            unreachable = load_set - set(np.where(reachable)[0].tolist())
            if not unreachable:
                break

            # Bridge: BFS from reachable set through ALL elements
            parent = np.full(self.n_elements, -1, dtype=np.int64)
            for e in range(self.n_elements):
                if reachable[e]:
                    parent[e] = e
            bq = deque(e for e in range(self.n_elements) if reachable[e])
            found = -1
            while bq and found < 0:
                e = bq.popleft()
                for nb in self._elem_adj[e]:
                    if parent[nb] < 0:
                        parent[nb] = e
                        if nb in unreachable:
                            found = nb
                            break
                        bq.append(nb)

            if found < 0:
                break

            # Trace back path and restore bridging elements + neighbors
            bridge_path = []
            e = found
            while e >= 0 and not reachable[e]:
                bridge_path.append(e)
                reachable[e] = True
                e = int(parent[e])

            # Restore bridge path elements
            for be in bridge_path:
                if rho[be] < 0.5:
                    rho[be] = 1.0
                    total_restored += 1

            # Thicken bridge: also restore immediate neighbors of bridge
            for be in bridge_path:
                for nb in self._elem_adj[be]:
                    if rho[nb] < 0.5:
                        rho[nb] = 1.0
                        total_restored += 1

        if total_restored > 0:
            print(f'    [CONNECTIVITY] Restored {total_restored} bridging elements')

        return rho

    def _enforce_passive_solid(self, rho):
        if np.any(self.passive_solid):
            rho[self.passive_solid] = 1.0
        return rho

    def _apply_sensitivity_filter(self, sens, rho):
        out = np.zeros_like(sens)
        for i, (ids, w) in enumerate(zip(self.neighbors, self.weights)):
            # Standard BESO: weighted average without density-normalization
            # (avoids amplifying void-element sensitivities by 1/rho_min)
            out[i] = float(np.dot(w, sens[ids]))
        return out

    def _compute_sensitivity(self, u, rho):
        ue = u[self.elem_dofs]
        ce0 = np.einsum('ni,nij,nj->n', ue, self.ke0, ue)
        scale = max(1.0 - float(getattr(self.material, 'rho_min', 0.0)), 1e-12)
        # Classic BESO sensitivity uses elemental strain energy ranking without SIMP penalty interpolation.
        dc = -scale * ce0
        dc = self._apply_sensitivity_filter(dc, rho)
        if np.any(self.passive_solid):
            dc[self.passive_solid] = np.max(dc) + 1.0
        return dc

    def _threshold_by_volume(self, sens, volfrac, rho_prev=None):
        n_keep = int(np.clip(round(volfrac * self.n_elements), 1, self.n_elements))
        order = np.argsort(sens)[::-1]

        rho = np.full(self.n_elements, self.rho_min, dtype=np.float64)
        keep_ids = order[:n_keep]
        rho[keep_ids] = 1.0

        # BESO add-back: allow previously removed elements to return if sensitivity recovers.
        if rho_prev is not None and n_keep < self.n_elements:
            cutoff = float(sens[order[n_keep - 1]])
            add_thr = self.addback_alpha * cutoff
            was_void = np.asarray(rho_prev, dtype=np.float64) <= (self.rho_min + 1e-12)
            add_mask = was_void & (sens >= add_thr)
            if np.any(add_mask):
                rho[add_mask] = 1.0
                solid = np.flatnonzero(rho > 0.5)
                if solid.size > n_keep:
                    keep = solid[np.argsort(sens[solid])[::-1][:n_keep]]
                    rho[:] = self.rho_min
                    rho[keep] = 1.0

        self._enforce_passive_solid(rho)
        return rho

    def optimize(self, forces, fixed_dofs, n_iterations=None, visualizer=None):
        n_iter = self.max_iterations if n_iterations is None else int(min(max(1, n_iterations), self.max_iterations))
        rho = np.full(self.n_elements, 1.0, dtype=np.float64)
        self._enforce_passive_solid(rho)
        sens_hist = None
        compliance_history = []
        vol = 1.0
        prev_comp = None
        stable_count = 0

        print('\n' + '=' * 60)
        print('3D BESO Optimization (Auto-Stop)')
        print('=' * 60)

        if self.volfrac_eff > self.volfrac + 1e-12:
            print(f'  [CONSTRAINT] Volume target increased from {self.volfrac:.4f} to {self.volfrac_eff:.4f} to keep support/load regions solid.')

        for it in range(n_iter):
            vol = max(self.volfrac_eff, vol * (1.0 - self.er))

            if self.member_projector is not None:
                self.member_projector.set_beta(self._projection_beta(it))
                rho_phys = self.member_projector.apply(rho, target_volume=vol, rho_min=self.rho_min)
            else:
                rho_phys = rho.copy()
            self._enforce_passive_solid(rho_phys)

            u = self.fea.solve(rho_phys, forces, fixed_dofs, self.fea_penalty)
            compliance = float(self.fea.calculate_compliance(u, rho_phys, self.fea_penalty))
            compliance_history.append(compliance)

            sens = self._compute_sensitivity(u, rho_phys)
            if sens_hist is None:
                sens_avg = sens
            else:
                sens_avg = 0.5 * (sens + sens_hist)
            sens_hist = sens_avg

            rho_new = self._threshold_by_volume(sens_avg, vol, rho_prev=rho)
            rho_new = self._enforce_connectivity(rho_new, fixed_dofs, forces)
            if self.member_projector is not None:
                self.member_projector.set_beta(self._projection_beta(it))
                rho_new = self.member_projector.apply(rho_new, target_volume=vol, rho_min=self.rho_min)
            self._enforce_passive_solid(rho_new)

            comp_rel = abs(compliance - prev_comp) / max(abs(compliance), 1e-12) if prev_comp is not None else np.inf
            prev_comp = compliance
            rho_change = float(np.max(np.abs(rho_new - rho)))
            rho = rho_new

            if (it + 1) >= self.min_iterations and rho_change < self.change_tol and comp_rel < self.comp_tol:
                stable_count += 1
            else:
                stable_count = 0

            vnow = float(np.mean(rho))
            if visualizer is not None:
                try:
                    rho_vis = np.where(rho >= 0.5, 1.0, self.rho_min)
                    self._enforce_passive_solid(rho_vis)
                    visualizer.update(it, rho_vis.copy(), compliance, vnow)
                except Exception as e:
                    print(f'  [VIEWER] {e}')

            print(f'  Iteration {it + 1}: C={compliance:.6e}, V={vnow:.4f}, dR={rho_change:.3e}, ER={self.er:.4f}')

            if stable_count >= self.stall_patience:
                print(f'  [CONVERGED] Objective stabilized for {self.stall_patience} iterations; stopping at {it + 1}.')
                break

            self.er = max(self.er * 0.97, self.ert)

        return rho, compliance_history


