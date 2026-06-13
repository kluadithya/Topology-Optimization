"""
3D Level Set Method (LSM) optimizer with Hamilton-Jacobi PDE evolution.

Implements a proper geometric level-set method:
  - Level-set function phi defined on element centroids
  - Hamilton-Jacobi PDE: dphi/dt + V_n * |grad_phi| = 0
  - Shape derivative velocity from strain energy at the interface
  - Least-squares gradient reconstruction on unstructured tet mesh
  - Fast Marching Method (graph-based) for signed-distance reinitialization
  - Regularized Heaviside for smooth density mapping
"""

import time
import heapq
import numpy as np
from collections import deque
from scipy.spatial import cKDTree
from fea_solver_3d import FEASolver3D
from density_filter_3d import DensityFilter3D, MinimumMemberSizeProjection3D, auto_filter_radius_from_mesh


class LSM3DOptimizer:
    """3D Level-Set Method optimizer with Hamilton-Jacobi PDE evolution and auto convergence."""

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
        # LSM's Heaviside-projected densities create near-binary stiffness jumps
        # that choke PyAMG's multigrid hierarchy construction. CHOLMOD with cached
        # symbolic factorization gives exact solutions in 2-3s per iteration.
        self.fea.linear_solver_mode = 'direct'
        self.material = material
        self.n_elements = int(elements.shape[0])

        objective = str(config.get('objective', config.get('objective_function', 'compliance'))).lower()
        volfrac = float(config.get('volume_fraction', config.get('volfrac', 0.5)))
        if objective == 'weight':
            wr = config.get('weight_reduction_percent', config.get('target_weight_reduction_percent', None))
            if wr is not None:
                try:
                    volfrac = 1.0 - float(wr) / 100.0
                except Exception:
                    pass
        self.volfrac = float(np.clip(volfrac, 0.05, 1.0))

        self.penalty = float(config.get('penalization', config.get('penalty', 3.0)))
        self.max_iterations = int(config.get('max_iterations_auto', config.get('iterations', 300)))
        self.min_iterations = int(max(10, config.get('min_iterations_auto', 60)))
        self.change_tol = float(config.get('density_change_tolerance', 1e-3))
        self.comp_tol = float(config.get('lsm_compliance_tolerance', config.get('compliance_tolerance', 2.0e-2)))
        self.stall_patience = int(max(3, config.get('stall_patience', 8)))

        # Hamilton-Jacobi PDE parameters
        self.time_step = float(config.get('lsm_time_step', 0.25))
        self.cfl_factor = float(config.get('lsm_cfl_factor', 0.5))  # CFL safety factor
        self.epsilon = float(config.get('lsm_heaviside_epsilon', 0.10))  # Heaviside regularization width
        self.reinit_interval = int(max(0, config.get('lsm_reinit_interval', 10)))
        self.reinit_method = str(config.get('lsm_reinit_method', 'fast_marching')).lower()
        self.init_mode = str(config.get('lsm_init_mode', 'ellipsoid')).lower()
        self.init_noise = float(max(config.get('lsm_init_noise', 0.0), 0.0))
        self.velocity_smooth_passes = int(max(0, config.get('lsm_velocity_smooth_passes', 1)))
        self.nucleation_weight_floor = float(np.clip(config.get('lsm_nucleation_weight_floor', 0.0), 0.0, 1.0))
        self.phi_smooth_weight = float(np.clip(config.get('lsm_phi_smooth_weight', 0.15), 0.0, 0.49))
        self.phi_smooth_passes = int(max(0, config.get('lsm_phi_smooth_passes', 1)))
        self.dt_decay_rate = float(config.get('lsm_dt_decay_rate', 0.995))
        self._velocity_scale = 1.0
        self._osc_damp = 1.0           # persistent oscillation damping factor
        self._osc_recover = float(config.get('lsm_osc_recover', 1.02))  # recovery per iter
        self._osc_reduce = float(config.get('lsm_osc_reduce', 0.75))    # reduction on detect

        self.rho_min = float(config.get('rho_min', getattr(material, 'rho_min', 1e-6)))
        self.rho_min = float(np.clip(self.rho_min, 1e-9, 5e-2))
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
        self.filter = DensityFilter3D(self.fea.nodes, self.fea.elements, self.filter_radius)

        self.use_min_member_projection = bool(config.get('use_min_member_projection_lsm', config.get('use_min_member_projection', False)))
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

        self.use_stress_constraint = bool(config.get('use_stress_constraint', True))
        self.safety_factor = float(config.get('safety_factor', 1.5))
        self.stress_penalty_weight = float(config.get('stress_penalty_weight', 2.0))
        ys = float(getattr(material, 'yield_strength', config.get('yield_strength', 250.0)))
        self.allowable_stress = ys / max(self.safety_factor, 1e-9)

        passive = config.get('passive_solid_elements', None)
        self.passive_solid = np.zeros(self.n_elements, dtype=bool)
        if passive is not None:
            p = np.asarray(passive, dtype=bool).reshape(-1)
            if p.size == self.n_elements:
                self.passive_solid = p.copy()

        passive_frac = float(np.mean(self.passive_solid)) if self.passive_solid.size else 0.0
        min_target = passive_frac + (1.0 / max(self.n_elements, 1))
        self.volfrac_eff = float(np.clip(max(self.volfrac, min_target), 0.05, 1.0))

        # Initialize level-set field as signed distance from initial volume fraction
        # Positive = solid, negative = void
        self.centroids = np.mean(self.fea.nodes[self.fea.elements], axis=1)
        self._build_element_graph()

        # Initialize phi as a smooth contiguous field to avoid speckled early removal.
        bbox_min = np.min(self.centroids, axis=0)
        bbox_max = np.max(self.centroids, axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        bbox_half = 0.5 * (bbox_max - bbox_min)
        bbox_diag = float(np.linalg.norm(bbox_half))

        scale = np.maximum(bbox_half, 1e-12)
        r_norm = np.linalg.norm((self.centroids - bbox_center) / scale, axis=1)
        r0 = float(np.clip(self.volfrac_eff, 0.05, 0.99) ** (1.0 / 3.0))
        self.phi = (r0 - r_norm) * max(bbox_diag, 1e-6)

        if self.init_mode == 'solid':
            self.phi = np.full(self.n_elements, max(bbox_diag, 1e-6), dtype=np.float64)

        if self.init_noise > 0.0:
            rng = np.random.default_rng(7)
            amp = self.init_noise * max(bbox_diag, 1e-6)
            n = amp * rng.standard_normal(self.n_elements)
            for _ in range(2):
                n = self.filter.apply_density(n)
            self.phi += n

        self._enforce_passive_on_phi()
        self._enforce_volume()

        elems = np.asarray(self.fea.elements, dtype=np.int64)
        npe = int(elems.shape[1])
        self.elem_dofs = (3 * elems[:, :, None] + np.arange(3, dtype=np.int64)).reshape(self.n_elements, 3 * npe)
        rho_probe = np.ones(self.n_elements, dtype=np.float64)
        self.ke0 = np.stack([self.fea.get_element_stiffness(i, rho_probe, penalty=1.0) for i in range(self.n_elements)], axis=0)

        # Precompute least-squares gradient operators for each element
        self._precompute_gradient_operators()

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

    def _enforce_passive_on_phi(self):
        if np.any(self.passive_solid):
            self.phi[self.passive_solid] = 20.0

    def _enforce_passive_on_rho(self, rho):
        if np.any(self.passive_solid):
            rho[self.passive_solid] = 1.0
        return rho

    def _enforce_load_path_connectivity(self, fixed_dofs, forces):
        """Ensure phi maintains connected load paths between supports and loads.

        Uses SciPy's compiled C-graph shortest_path to instantly find bridging
        paths across disconnected components.
        """
        elems = self.fea.elements

        fd = np.asarray(fixed_dofs, dtype=np.int64)
        support_nodes = np.unique(fd // 3)
        f3 = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        load_nodes = np.where(np.max(np.abs(f3), axis=1) > 1e-30)[0]

        if support_nodes.size == 0 or load_nodes.size == 0:
            return

        support_mask = np.any(np.isin(elems, support_nodes), axis=1)
        load_mask = np.any(np.isin(elems, load_nodes), axis=1)

        # Ensure support/load elements stay solid
        eps = max(self.epsilon * self.filter_radius, 1e-12)
        self.phi[support_mask] = np.maximum(self.phi[support_mask], eps * 2.0)
        self.phi[load_mask] = np.maximum(self.phi[load_mask], eps * 2.0)

        support_set = np.where(support_mask)[0].astype(np.int32)
        load_set = np.where(load_mask)[0].astype(np.int32)

        rho = self._heaviside_density()
        is_solid = rho >= 0.5
        
        node_weights = np.where(is_solid, 1e-6, 1.0)
        data = node_weights[self._adj_cols]
        
        N = self.n_elements
        ss_rows = np.full(len(support_set), N, dtype=np.int32)
        ss_cols = support_set
        ss_data = node_weights[support_set]
        
        all_rows = np.concatenate([self._adj_rows, ss_rows])
        all_cols = np.concatenate([self._adj_cols, ss_cols])
        all_data = np.concatenate([data, ss_data])
        
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        
        graph = csr_matrix((all_data, (all_rows, all_cols)), shape=(N + 1, N + 1))
        
        dist, pred = shortest_path(graph, directed=True, indices=N, return_predecessors=True)
        
        load_dists = dist[load_set]
        disconnected_loads = load_set[load_dists >= 0.5]
        
        total_restored = 0
        if len(disconnected_loads) == 0:
            return
            
        for target in disconnected_loads:
            curr = target
            while curr != N and curr >= 0:
                if self.phi[curr] < eps:
                    self.phi[curr] = eps * 2.0
                    total_restored += 1
                for nb in self.elem_neighbors[curr]:
                    if self.phi[nb] < eps:
                        self.phi[nb] = eps * 2.0
                        total_restored += 1
                curr = pred[curr]
                
        if total_restored > 0:
            print(f'    [CONNECTIVITY] Restored load paths ({total_restored} elements bridged).')
            
        self._enforce_passive_on_phi()

    # â”€â”€ Regularized Heaviside: maps phi â†’ density â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _heaviside_density(self, phi=None):
        """Regularized Heaviside function for smooth density mapping.

        H_eps(phi) = { 0                                        if phi < -eps
                     { 0.5 * (1 + phi/eps + sin(pi*phi/eps)/pi)  if |phi| <= eps
                     { 1                                        if phi > eps

        This is the standard regularized Heaviside used in level-set topology optimization
        (Allaire et al. 2004, Wang et al. 2003).
        """
        if phi is None:
            phi = self.phi
        h = self.element_size if np.isfinite(self.element_size) and self.element_size > 0.0 else 1.0
        eps = max(self.epsilon * self.filter_radius, 1.5 * h)
        rho = np.zeros(self.n_elements, dtype=np.float64)

        # Region: phi > eps â†’ solid
        solid = phi > eps
        rho[solid] = 1.0

        # Region: |phi| <= eps â†’ transition
        band = np.abs(phi) <= eps
        if np.any(band):
            x = phi[band] / eps
            rho[band] = 0.5 * (1.0 + x + np.sin(np.pi * x) / np.pi)

        # Region: phi < -eps â†’ void (already 0)
        # Clamp to [rho_min, 1]
        rho = np.clip(rho, self.rho_min, 1.0)
        return self._enforce_passive_on_rho(rho)

    def _heaviside_derivative(self, phi=None):
        """Derivative of regularized Heaviside: delta_eps(phi).

        delta_eps(phi) = { 0                              if |phi| > eps
                         { (1 + cos(pi*phi/eps)) / (2*eps) if |phi| <= eps
        """
        if phi is None:
            phi = self.phi
        h = self.element_size if np.isfinite(self.element_size) and self.element_size > 0.0 else 1.0
        eps = max(self.epsilon * self.filter_radius, 1.5 * h)
        delta = np.zeros(self.n_elements, dtype=np.float64)

        band = np.abs(phi) <= eps
        if np.any(band):
            x = phi[band] / eps
            delta[band] = (1.0 + np.cos(np.pi * x)) / (2.0 * eps)

        return delta

    # â”€â”€ Element connectivity graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_element_graph(self):
        """Build element-element adjacency from shared nodes."""
        elems = np.asarray(self.fea.elements, dtype=np.int64)
        n_nodes = int(np.asarray(self.fea.nodes).shape[0])
        node_to_elems = [[] for _ in range(n_nodes)]
        for eidx in range(self.n_elements):
            for nid in elems[eidx]:
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    node_to_elems[ni].append(eidx)

        rows = []
        cols = []
        self.elem_neighbors = []
        for eidx in range(self.n_elements):
            nset = set()
            for nid in elems[eidx]:
                nset.update(node_to_elems[int(nid)])
            nset.discard(eidx)
            nbrs = np.asarray(sorted(nset), dtype=np.int64)
            self.elem_neighbors.append(nbrs)
            rows.extend([eidx] * len(nbrs))
            cols.extend(nbrs)
            
        self._adj_rows = np.array(rows, dtype=np.int32)
        self._adj_cols = np.array(cols, dtype=np.int32)

    # â”€â”€ Gradient computation: least-squares on unstructured mesh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _precompute_gradient_operators(self):
        """Precompute least-squares gradient reconstruction operators.

        For each element i, we fit: phi(j) - phi(i) â‰ˆ grad_phi_i Â· (c_j - c_i)
        using all neighbors j. This gives a 3-component gradient at each element centroid.
        The normal Moore-Penrose pseudoinverse is precomputed for efficiency.
        """
        self._grad_neighbors = []
        self._grad_pinv = []  # Each entry: (3, n_neighbors) pseudoinverse matrix

        for i in range(self.n_elements):
            nbrs = self.elem_neighbors[i]
            if len(nbrs) == 0:
                self._grad_neighbors.append(np.array([], dtype=np.int64))
                self._grad_pinv.append(np.zeros((3, 0), dtype=np.float64))
                continue

            # Delta vectors from centroid i to each neighbor centroid
            dx = self.centroids[nbrs] - self.centroids[i]  # (n_nbr, 3)

            # Compute pseudoinverse: pinv(dx) has shape (3, n_nbr)
            try:
                pinv = np.linalg.pinv(dx)  # (3, n_nbr)
            except np.linalg.LinAlgError:
                pinv = np.zeros((3, len(nbrs)), dtype=np.float64)

            self._grad_neighbors.append(nbrs)
            self._grad_pinv.append(pinv)

    def _compute_gradient(self, field, band=None):
        """Compute gradient of a scalar field using precomputed least-squares operators."""
        field = np.asarray(field, dtype=np.float64)
        grad = np.zeros((self.n_elements, 3), dtype=np.float64)

        if band is None:
            indices = range(self.n_elements)
        else:
            indices = np.where(band)[0]

        for i in indices:
            nbrs = self._grad_neighbors[i]
            if len(nbrs) == 0:
                continue
            dphi = field[nbrs] - field[i]  # (n_nbr,)
            grad[i] = self._grad_pinv[i] @ dphi  # (3, n_nbr) @ (n_nbr,) -> (3,)

        return grad

    def _compute_gradient_magnitude(self, field):
        """Compute |∇field| at each element centroid."""
        grad = self._compute_gradient(field)
        return np.sqrt(np.sum(grad ** 2, axis=1))

    def _smooth_field_on_graph(self, field, passes=1):
        """Neighbor averaging on element graph for mild regularization."""
        if passes <= 0:
            return np.asarray(field, dtype=np.float64)
        src = np.asarray(field, dtype=np.float64).copy()
        for _ in range(passes):
            out = src.copy()
            for i, nbrs in enumerate(self.elem_neighbors):
                if len(nbrs) == 0:
                    continue
                out[i] = float(np.mean(src[nbrs]))
            src = out
        return src

    def _smooth_phi(self):
        """Apply a small diffusion-like smoothing to reduce jagged boundaries."""
        if self.phi_smooth_passes <= 0 or self.phi_smooth_weight <= 0.0:
            return
        lam = self.phi_smooth_weight
        for _ in range(self.phi_smooth_passes):
            avg = self._smooth_field_on_graph(self.phi, passes=1)
            self.phi = (1.0 - lam) * self.phi + lam * avg
            self._enforce_passive_on_phi()

    # ——— Hamilton-Jacobi PDE evolution ——————————————————————————————————

    def _shape_derivative_velocity(self, u, rho):
        """Compute shape derivative velocity field.

        For compliance minimization, the shape derivative at the interface is:
            V_n = -ce (element strain energy density)

        This velocity is defined at the interface and extended to the domain.
        """
        ue = u[self.elem_dofs]
        ce0 = np.einsum('ni,nij,nj->n', ue, self.ke0, ue)

        # Shape derivative: V_n = -strain_energy at interface
        velocity = -ce0

        # Apply stress constraint penalty if enabled
        if self.use_stress_constraint:
            vm = self.fea.calculate_stress_all_gauss(u, rho, self.penalty)
            ratio = vm / max(self.allowable_stress, 1e-12)
            violation = np.maximum(ratio - 1.0, 0.0)
            stress_penalty = 1.0 + self.stress_penalty_weight * (violation ** 2)
            velocity = velocity * stress_penalty

        return velocity

    def _upwind_hj_update(self, velocity):
        """Solve Hamilton-Jacobi PDE: dphi/dt + V_n * |grad_phi| = 0.

        Uses first-order upwind scheme on unstructured mesh restricted to a Narrow Band
        around the structural interface for massive computational savings.
        """
        # Define narrow band: 3x filter radius around zero-level set
        band_width = max(3.0 * self.filter_radius, 1e-6)
        band = np.abs(self.phi) <= band_width
        if not np.any(band):
            band = np.ones(self.n_elements, dtype=bool)

        # Compute gradient only within the narrow band
        grad_phi = self._compute_gradient(self.phi, band=band)
        grad_mag = np.sqrt(np.sum(grad_phi ** 2, axis=1))

        # For upwind stability
        grad_mag_safe = np.maximum(grad_mag, 1e-12)

        # CFL-limited time step based on max velocity IN the band
        max_vel = float(np.max(np.abs(velocity[band])))
        if max_vel < 1e-30:
            return

        # Characteristic mesh size for CFL
        h = self.element_size if np.isfinite(self.element_size) and self.element_size > 0.0 else 1.0
        dt_cfl = self.cfl_factor * h / max(max_vel, 1e-12)
        dt = min(self.time_step * h, dt_cfl) * self._velocity_scale

        # H-J update: phi_new = phi - dt * V_n * |grad_phi|
        dphi = dt * velocity * grad_mag_safe

        # Clamp per-element update to ±0.5*h to prevent overshooting
        # (standard CFL stability guarantee for level-set methods)
        max_step = 0.5 * h
        dphi = np.clip(dphi, -max_step, max_step)

        self.phi = self.phi - dphi

        # Clamp phi to avoid blow-up
        self.phi = np.clip(self.phi, -20.0 * h, 20.0 * h)
        self._enforce_passive_on_phi()

    # â”€â”€ Volume constraint enforcement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _enforce_volume(self):
        """Shift phi uniformly to satisfy the volume constraint."""
        eps = max(self.epsilon * self.filter_radius, 1e-12)

        lo, hi = -200.0 * eps, 200.0 * eps
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rho_trial = self._heaviside_density(self.phi + mid)
            if float(np.mean(rho_trial)) > self.volfrac_eff:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < 1e-12 * eps:
                break
        self.phi += 0.5 * (lo + hi)
        self._enforce_passive_on_phi()

    # â”€â”€ Fast Marching reinitialization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fast_marching_reinit(self):
        """Reinitialize phi to approximate signed-distance function using
        graph-based Fast Marching Method.

        The FMM solves |âˆ‡T| = 1 from the interface (phi â‰ˆ 0) outward,
        giving a distance field. The sign is preserved from the current phi.
        """
        rho = self._heaviside_density()

        # Find interface elements: those near phi=0 (rho near 0.5)
        h = self.element_size if np.isfinite(self.element_size) and self.element_size > 0.0 else 1.0
        eps = max(self.epsilon * self.filter_radius, 1.5 * h)
        interface = np.abs(self.phi) <= eps
        if not np.any(interface):
            # Fallback: find elements closest to rho=0.5
            k = int(np.clip(max(32, int(0.03 * self.n_elements)), 1, self.n_elements))
            idx = np.argpartition(np.abs(rho - 0.5), k - 1)[:k]
            interface = np.zeros(self.n_elements, dtype=bool)
            interface[idx] = True

        # Preserve sign
        sign = np.where(self.phi >= 0.0, 1.0, -1.0)

        # Dijkstra-based FMM: solve for distance from interface
        inf = 1e30
        dist = np.full(self.n_elements, inf, dtype=np.float64)
        heap = []

        seed_ids = np.where(interface)[0]
        for i in seed_ids.tolist():
            # Initialize with absolute phi to preserve sub-element boundary position
            d0 = float(np.abs(self.phi[i]))
            dist[i] = d0
            heapq.heappush(heap, (d0, int(i)))

        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i]:
                continue
            ci = self.centroids[i]
            for j in self.elem_neighbors[i]:
                jj = int(j)
                # Edge weight is the physical distance between centroids
                w = float(np.linalg.norm(ci - self.centroids[jj]))
                nd = d + max(w, 1e-12)
                if nd < dist[jj]:
                    dist[jj] = nd
                    heapq.heappush(heap, (nd, jj))

        # Handle unreached elements (disconnected components)
        missing = dist >= (0.5 * inf)
        if np.any(missing) and np.any(interface):
            tree = cKDTree(self.centroids[interface])
            d_euc, _ = tree.query(self.centroids[missing], k=1)
            dist[missing] = np.asarray(d_euc, dtype=np.float64)

        # Reconstruct signed distance
        phi_new = sign * dist

        # Smooth blend with current phi to avoid discontinuities
        # Use iteration-adaptive blend if available (strong early, gentle late)
        blend = getattr(self, '_current_reinit_blend',
                        float(self.config.get('lsm_reinit_blend', 0.40)))
        self.phi = blend * phi_new + (1.0 - blend) * self.phi
        self.phi = np.clip(self.phi, -20.0 * max(self.filter_radius, 1e-6), 20.0 * max(self.filter_radius, 1e-6))
        self._enforce_passive_on_phi()

    # â”€â”€ Final binary projection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _binary_projection(self):
        """Project to binary field for final result output."""
        n_solid = int(np.clip(round(self.volfrac_eff * self.n_elements), 1, self.n_elements))
        order = np.argsort(self.phi)[::-1]
        rho_bin = np.full(self.n_elements, self.rho_min, dtype=np.float64)
        rho_bin[order[:n_solid]] = 1.0
        self._enforce_passive_on_rho(rho_bin)
        return rho_bin

    # â”€â”€ Main optimization loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def optimize(self, forces, fixed_dofs, n_iterations=None, visualizer=None):
        n_iter = self.max_iterations if n_iterations is None else int(min(max(1, n_iterations), self.max_iterations))
        compliance_history = []
        prev_rho = None
        prev_phi = None
        prev_comp = None
        stable_count = 0
        self._osc_damp = 1.0
        # Rolling window for compliance stabilization detection
        rolling_window = int(max(self.stall_patience, 15))
        # Majority-vote window: how many of the last N windows must be stable
        vote_window = max(3, self.stall_patience // 2)

        print('\n' + '=' * 60)
        print('3D Level-Set Optimization (Hamilton-Jacobi PDE, Auto-Stop)')
        print('=' * 60)
        print(f'  [LSM] H-J PDE evolution with upwind gradient, CFL={self.cfl_factor:.2f}')
        print(f'  [LSM] Regularized Heaviside eps={self.epsilon:.3f} * filter_radius')
        print(f'  [LSM] Reinitialization every {self.reinit_interval} iterations (method: {self.reinit_method})')
        print(f'  [LSM] Adaptive dt decay={self.dt_decay_rate:.4f}, convergence tol={self.comp_tol:.1e}')

        if self.volfrac_eff > self.volfrac + 1e-12:
            print(f'  [CONSTRAINT] Volume target increased from {self.volfrac:.4f} to {self.volfrac_eff:.4f} to keep support/load regions solid.')

        for it in range(n_iter):
            t0 = time.time()
            # 1. Enforce volume constraint by shifting phi
            self._enforce_volume()

            # 2. Compute density from level-set via regularized Heaviside
            rho = self._heaviside_density()

            # Apply optional projection
            if self.member_projector is not None:
                self.member_projector.set_beta(self._projection_beta(it))
                rho = self.member_projector.apply(rho, target_volume=self.volfrac_eff, rho_min=self.rho_min)
            else:
                # Do NOT filter the physical density rho in classical LSM!
                pass
            self._enforce_passive_on_rho(rho)

            # Physical density for FEA (SIMP interpolation for smooth void regions)
            rho_phys = np.clip(self.rho_min + (1.0 - self.rho_min) * rho, self.rho_min, 1.0)
            self._enforce_passive_on_rho(rho_phys)

            # 3. Solve FEA
            u = self.fea.solve(rho_phys, forces, fixed_dofs, self.penalty)
            compliance = float(self.fea.calculate_compliance(u, rho_phys, self.penalty))
            compliance_history.append(compliance)

            # 4. Compute shape derivative velocity
            velocity = self._shape_derivative_velocity(u, rho_phys)

            # Weight velocity by Heaviside derivative (localize to interface)
            delta_h = self._heaviside_derivative()
            delta_max = float(np.max(delta_h))
            if delta_max > 1e-30:
                # Normalize delta to [0, 1] range and use as weight
                weight = delta_h / delta_max
                # Keep update mostly interface-local to avoid speckled void nucleation.
                floor = max(self.nucleation_weight_floor, 0.1)
                weight = np.maximum(weight, floor)
                velocity = velocity * weight

            if self.velocity_smooth_passes > 0:
                velocity = self._smooth_field_on_graph(velocity, passes=self.velocity_smooth_passes)

            # Normalize velocity
            vel_norm = float(np.max(np.abs(velocity)))
            if vel_norm > 1e-30:
                velocity = velocity / vel_norm

            # Zero out velocity on passive elements
            if np.any(self.passive_solid):
                velocity[self.passive_solid] = 0.0

            # 5. Adaptive time-step scaling: gradual decay + persistent oscillation damping
            # Detect zig-zag: compliance alternates direction in last 6 points
            if len(compliance_history) >= 6:
                diffs = np.diff(compliance_history[-6:])
                signs = np.sign(diffs)
                sign_flips = int(np.sum(signs[:-1] * signs[1:] < 0))
                if sign_flips >= 3:                         # oscillating
                    self._osc_damp = max(self._osc_damp * self._osc_reduce, 0.05)
                else:                                       # calming down — slowly recover
                    self._osc_damp = min(self._osc_damp * self._osc_recover, 1.0)
            self._velocity_scale = (self.dt_decay_rate ** it) * self._osc_damp

            # Evolve phi via Hamilton-Jacobi PDE
            self._upwind_hj_update(velocity)
            self._smooth_phi()

            # Connectivity guard: ensure load paths remain connected.
            self._enforce_load_path_connectivity(fixed_dofs, forces)

            # 6. Periodic reinitialization to signed distance
            #    Taper blend: strong early (maintain signed-distance), gentle late (allow convergence)
            if self.reinit_interval > 0 and ((it + 1) % self.reinit_interval == 0):
                progress = float(it) / float(max(n_iter - 1, 1))
                blend_start = float(self.config.get('lsm_reinit_blend', 0.70))
                blend_end = float(self.config.get('lsm_reinit_blend_end', 0.10))
                self._current_reinit_blend = blend_start + (blend_end - blend_start) * progress
                self._fast_marching_reinit()

            # 7. Convergence tracking
            max_vm = 0.0
            violate_frac = 0.0
            if self.use_stress_constraint:
                vm = self.fea.calculate_stress_all_gauss(u, rho_phys, self.penalty)
                max_vm = float(np.max(vm)) if vm.size else 0.0
                violation = np.maximum(vm / max(self.allowable_stress, 1e-12) - 1.0, 0.0)
                violate_frac = float(np.mean(violation > 0.0)) if violation.size else 0.0

            # For LSM, tracking max density change (rho_change) is unreliable because 
            # the Heaviside projection causes boundary elements to violently oscillate 
            # between 0 and 1 even when the structure has converged.
            # Instead, we rely purely on the objective compliance stabilizing.
            rho_change = float(np.max(np.abs(rho - prev_rho))) if prev_rho is not None else np.inf
            phi_change = float(np.max(np.abs(self.phi - prev_phi))) if prev_phi is not None else np.inf
            comp_rel = abs(compliance - prev_comp) / max(abs(compliance), 1e-12) if prev_comp is not None else np.inf
            
            prev_rho = rho.copy()
            prev_phi = self.phi.copy()
            prev_comp = compliance

            # Rolling-window convergence: majority-vote so one spiky iter
            # can't reset all progress.
            if (it + 1) >= self.min_iterations and len(compliance_history) >= rolling_window:
                # Check stability across `vote_window` overlapping sub-windows
                votes_stable = 0
                for lag in range(vote_window):
                    sub = compliance_history[-(rolling_window + lag): len(compliance_history) - lag if lag > 0 else None]
                    if len(sub) >= rolling_window:
                        c_mean = float(np.mean(sub[-rolling_window:]))
                        c_spread = float(np.max(sub[-rolling_window:]) - np.min(sub[-rolling_window:]))
                        if c_spread / max(abs(c_mean), 1e-12) < self.comp_tol:
                            votes_stable += 1
                if votes_stable >= vote_window:   # ALL sub-windows stable
                    stable_count += 1
                elif votes_stable >= max(1, vote_window - 1):  # allow 1 bad window
                    stable_count += 1
                else:
                    stable_count = max(0, stable_count - 1)  # soft reset
            else:
                stable_count = 0

            vol = float(np.mean(rho))
            if visualizer is not None:
                try:
                    rho_vis = np.where(rho >= 0.5, 1.0, self.rho_min)
                    self._enforce_passive_on_rho(rho_vis)
                    visualizer.update(it, rho_vis.copy(), compliance, vol)
                except Exception as e:
                    print(f'  [VIEWER] {e}')

            t_iter = time.time() - t0
            if self.use_stress_constraint:
                print(f'  Iteration {it + 1}: C={compliance:.6e}, V={vol:.4f}, dR={rho_change:.3e}, '
                      f'VMmax={max_vm:.3e} Pa, Viol={100.0 * violate_frac:.1f}%, T={t_iter:.1f}s')
            else:
                print(f'  Iteration {it + 1}: C={compliance:.6e}, V={vol:.4f}, dR={rho_change:.3e}, T={t_iter:.1f}s')

            if stable_count >= self.stall_patience:
                print(f'  [CONVERGED] Objective stabilized for {self.stall_patience} iterations; stopping at {it + 1}.')
                break

        return self._binary_projection(), compliance_history

    def run_verification_fea(self, forces, fixed_dofs, threshold=0.5):
        """Run verification FEA on the binary-thresholded (manufacturable) design."""
        rho_cont = np.copy(self._binary_projection())
        rho_binary = np.where(rho_cont >= threshold, 1.0, self.rho_min)
        self._enforce_passive_on_rho(rho_binary)

        print(f'\n{"="*60}')
        print(f'  Verification FEA on Binary Design (threshold={threshold:.2f})')
        print(f'{"="*60}')
        n_solid = int(np.sum(rho_binary >= threshold))
        print(f'  Solid elements: {n_solid}/{self.n_elements} ({100.0*n_solid/max(self.n_elements,1):.1f}%)')

        u_cont = self.fea.solve(rho_cont, forces, fixed_dofs, self.penalty)
        comp_cont = self.fea.calculate_compliance(u_cont, rho_cont, self.penalty)
        max_disp_cont = float(np.max(np.abs(u_cont)))

        u_bin = self.fea.solve(rho_binary, forces, fixed_dofs, self.penalty)
        comp_bin = self.fea.calculate_compliance(u_bin, rho_binary, self.penalty)

        solid_elem_mask = (rho_binary >= threshold)
        solid_node_indices = np.unique(self.fea.elements[solid_elem_mask].flatten())
        if solid_node_indices.size > 0:
            solid_dofs = np.concatenate([
                3 * solid_node_indices,
                3 * solid_node_indices + 1,
                3 * solid_node_indices + 2,
            ])
            max_disp_bin = float(np.max(np.abs(u_bin[solid_dofs])))
        else:
            max_disp_bin = 0.0

        vm_bin = self.fea.calculate_stress_all_gauss(u_bin, rho_binary, self.penalty)
        max_vm_bin = float(np.max(vm_bin)) if vm_bin.size > 0 else 0.0

        comp_diff = 100.0 * abs(comp_bin - comp_cont) / max(abs(comp_cont), 1e-30)
        disp_diff = 100.0 * abs(max_disp_bin - max_disp_cont) / max(abs(max_disp_cont), 1e-30)

        print(f'\n  {"Metric":<25} {"Continuous":>14} {"Binary":>14} {"Diff %":>8}')
        print(f'  {"-"*63}')
        print(f'  {"Compliance":<25} {comp_cont:>14.4e} {comp_bin:>14.4e} {comp_diff:>7.1f}%')
        print(f'  {"Max displacement":<25} {max_disp_cont:>14.4e} {max_disp_bin:>14.4e} {disp_diff:>7.1f}%')
        print(f'  {"Max von Mises (binary)":<25} {"—":>14} {max_vm_bin:>14.4e}')
        if hasattr(self, 'allowable_stress'):
            ratio = max_vm_bin / max(self.allowable_stress, 1e-30)
            status = 'OK' if ratio <= 1.0 else 'EXCEEDED'
            print(f'  {"Stress ratio (σ/σ_allow)":<25} {"—":>14} {ratio:>14.3f} [{status}]')
        print()

        return {
            'compliance_continuous': comp_cont,
            'compliance_binary': comp_bin,
            'max_disp_continuous': max_disp_cont,
            'max_disp_binary': max_disp_bin,
            'max_vm_binary': max_vm_bin,
            'rho_binary': rho_binary,
            'threshold': threshold,
        }
