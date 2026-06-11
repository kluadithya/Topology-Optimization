import numpy as np
from scipy.spatial import cKDTree
from fea_solver_3d import FEASolver3D
from density_filter_3d import MinimumMemberSizeProjection3D, auto_filter_radius_from_mesh


class MoriTanaka3DOptimizer:
    """3D Mori-Tanaka material interpolation optimizer with anisotropy regularization and auto-stop.

    This optimizer uses the Mori-Tanaka analytical micromechanics formula to compute
    effective elastic moduli for porous materials with void inclusions. Unlike true
    FE-based computational homogenization (which requires solving unit cell BVPs
    with periodic boundary conditions), this method provides a direct analytical
    estimate of effective isotropic modulus as a function of density.

    The resulting E(rho) curve serves as an alternative to the standard SIMP power-law
    interpolation, with the advantage of being physically motivated from micromechanics.

    Note: This is NOT a computational homogenization method. For true anisotropic
    homogenization, implement FE-based unit cell analysis with periodic BCs.
    """

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

        self.penalty = float(config.get('penalization', config.get('penalty', 3.0)))
        self.max_iterations = int(config.get('max_iterations_auto', config.get('iterations', 200)))
        self.min_iterations = int(max(10, config.get('min_iterations_auto', 25)))
        self.change_tol = float(config.get('density_change_tolerance', 1e-3))
        self.comp_tol = float(config.get('compliance_tolerance', 5e-4))
        self.stall_patience = int(max(3, config.get('stall_patience', 8)))

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
        self.move_limit = float(config.get('move_limit', 0.08))
        self.rho_min = float(config.get('rho_min', getattr(material, 'rho_min', 1e-6)))
        self.rho_min = float(np.clip(self.rho_min, 1e-9, 5e-2))

        self.use_min_member_projection = bool(config.get('use_min_member_projection', True))
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
        self._build_mt_lut()

        elems = np.asarray(self.fea.elements, dtype=np.int64)
        npe = int(elems.shape[1])
        self.elem_dofs = (3 * elems[:, :, None] + np.arange(3, dtype=np.int64)).reshape(self.n_elements, 3 * npe)
        rho_probe = np.ones(self.n_elements, dtype=np.float64)
        self.ke0 = np.stack([self.fea.get_element_stiffness(i, rho_probe, penalty=1.0) for i in range(self.n_elements)], axis=0)

    def _build_filter_map(self):
        from scipy.sparse import csr_matrix
        tree = cKDTree(self.centroids)
        data, rows, cols = [], [], []
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
            rows.extend([i] * len(ids))
            cols.extend(ids)
            data.extend(w)
        self.H = csr_matrix((data, (rows, cols)), shape=(self.n_elements, self.n_elements))

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

    def _enforce_passive_solid(self, rho):
        if np.any(self.passive_solid):
            rho[self.passive_solid] = 1.0
        return rho

    def _apply_density_filter(self, rho):
        rf = self.H.dot(rho)
        self._enforce_passive_solid(rf)
        return np.clip(rf, self.rho_min, 1.0)

    def _apply_sensitivity_filter(self, dc, rho):
        return self.H.dot(dc)

    def _build_mt_lut(self):
        self._lut_size = 1000
        self._lut_rho = np.linspace(self.rho_min, 1.0, self._lut_size)
        self._lut_E = self._effective_modulus_mori_tanaka_exact(self._lut_rho)
        h = 1e-4
        rp = np.clip(self._lut_rho + h, self.rho_min, 1.0)
        rm = np.clip(self._lut_rho - h, self.rho_min, 1.0)
        ep = self._effective_modulus_mori_tanaka_exact(rp)
        em = self._effective_modulus_mori_tanaka_exact(rm)
        self._lut_dEdrho = np.maximum((ep - em) / np.maximum(rp - rm, 1e-12), 1e-12)

    def _effective_modulus_mori_tanaka_exact(self, rho):
        """Mori-Tanaka isotropic porous-solid effective modulus estimate.

        Computes effective Young's modulus for a solid matrix containing spherical
        void inclusions using the Mori-Tanaka mean-field method.

        This gives an isotropic scalar modulus — no anisotropic tensor components.
        For anisotropic effective properties, use FE-based computational homogenization
        with periodic unit cell analysis.

        Reference: Mori & Tanaka, Acta Metallurgica, 1973.
        """
        r = np.clip(np.asarray(rho, dtype=np.float64), self.rho_min, 1.0)
        E0 = float(max(self.material.E0, 1e-12))
        nu0 = float(np.clip(self.material.nu, -0.49, 0.49))

        km = E0 / max(3.0 * (1.0 - 2.0 * nu0), 1e-12)
        gm = E0 / max(2.0 * (1.0 + nu0), 1e-12)
        ki = 0.0
        gi = 0.0

        ci = 1.0 - r
        cm = r

        p_k = km + (4.0 / 3.0) * gm
        f_m = gm * (9.0 * km + 8.0 * gm) / max(6.0 * (km + 2.0 * gm), 1e-12)

        den_k = 1.0 + cm * (ki - km) / max(p_k, 1e-12)
        den_g = 1.0 + cm * (gi - gm) / max(f_m, 1e-12)

        keff = km + ci * (ki - km) / np.maximum(den_k, 1e-12)
        geff = gm + ci * (gi - gm) / np.maximum(den_g, 1e-12)

        denom = np.maximum(3.0 * keff + geff, 1e-12)
        e_eff = 9.0 * keff * geff / denom
        e_eff = np.clip(e_eff, self.rho_min * E0, E0)
        return e_eff

    def _effective_modulus_mori_tanaka(self, rho):
        """Fast interpolation from precomputed MT lookup table."""
        r = np.clip(np.asarray(rho, dtype=np.float64), self.rho_min, 1.0)
        return np.interp(r, self._lut_rho, self._lut_E)

    def _effective_modulus_derivative_exact(self, rho):
        r = np.clip(np.asarray(rho, dtype=np.float64), self.rho_min, 1.0)
        h = 1e-4
        rp = np.clip(r + h, self.rho_min, 1.0)
        rm = np.clip(r - h, self.rho_min, 1.0)
        ep = self._effective_modulus_mori_tanaka_exact(rp)
        em = self._effective_modulus_mori_tanaka_exact(rm)
        d = (ep - em) / np.maximum(rp - rm, 1e-12)
        return np.maximum(d, 1e-12)

    def _effective_modulus_derivative(self, rho):
        """Fast interpolation from precomputed MT derivative lookup table."""
        r = np.clip(np.asarray(rho, dtype=np.float64), self.rho_min, 1.0)
        return np.interp(r, self._lut_rho, self._lut_dEdrho)

    def _equivalent_simp_density_from_modulus(self, e_eff):
        # Map effective modulus back to SIMP density so existing FEA routines can be reused.
        e = np.clip(np.asarray(e_eff, dtype=np.float64), self.rho_min * self.material.E0, self.material.E0)
        ratio = np.clip(e / max(self.material.E0, 1e-12), self.rho_min, 1.0)
        base = (ratio - self.rho_min) / max(1.0 - self.rho_min, 1e-12)
        rho_eq = np.power(np.clip(base, 0.0, 1.0), 1.0 / max(self.penalty, 1e-12))
        return np.clip(rho_eq, self.rho_min, 1.0)

    def _compute_homogenized_tensor(self, rho):
        """Compute effective isotropic stiffness tensor from Mori-Tanaka estimate.

        Note: This produces an isotropic tensor (anisotropy ratio = 1.0 always)
        since Mori-Tanaka with spherical voids gives isotropic effective properties.
        A true anisotropic effective tensor would require FE-based homogenization.
        """
        rho_eff = float(np.clip(np.mean(np.clip(rho, self.rho_min, 1.0)), self.rho_min, 1.0))
        e_eff = float(self._effective_modulus_mori_tanaka(np.asarray([rho_eff]))[0])
        nu = float(np.clip(self.material.nu, -0.49, 0.49))

        lam = e_eff * nu / max((1.0 + nu) * (1.0 - 2.0 * nu), 1e-12)
        mu = e_eff / max(2.0 * (1.0 + nu), 1e-12)
        c11 = lam + 2.0 * mu
        c12 = lam

        return {
            'C11': c11,
            'C22': c11,
            'C33': c11,
            'C12': c12,
            'C13': c12,
            'C23': c12,
            'anisotropy': 1.0,
            'E_eff': e_eff,
            'nu_eff': nu,
        }

    def _compute_element_sensitivities(self, u, rho, dEdrho):
        ue = u[self.elem_dofs]
        ce0 = np.einsum('ni,nij,nj->n', ue, self.ke0, ue)
        e0 = max(float(self.material.E0), 1e-12)
        base = -(np.asarray(dEdrho, dtype=np.float64) / e0) * ce0
        vol = np.ones(self.n_elements, dtype=np.float64)

        if np.any(self.passive_solid):
            base[self.passive_solid] = np.minimum(base[self.passive_solid], -1.0)

        return base, vol

    def _oc_update(self, rho, dc, dv):
        l1, l2 = 0.0, 1e9
        move = self.move_limit
        rho = np.clip(rho, self.rho_min, 1.0)
        cand = rho.copy()
        for _ in range(100):
            lmid = 0.5 * (l1 + l2)
            ratio = -dc / np.maximum(dv * lmid, 1e-30)
            ratio = np.maximum(ratio, 1e-12)
            cand = rho * np.sqrt(ratio)
            cand = np.clip(cand, rho - move, rho + move)
            cand = np.clip(cand, self.rho_min, 1.0)
            self._enforce_passive_solid(cand)
            if float(np.mean(cand)) > self.volfrac_eff:
                l1 = lmid
            else:
                l2 = lmid
        return cand

    def optimize(self, forces, fixed_dofs, n_iterations=None, visualizer=None):
        n_iter = self.max_iterations if n_iterations is None else int(min(max(1, n_iterations), self.max_iterations))
        rho = np.full(self.n_elements, self.volfrac_eff, dtype=np.float64)
        self._enforce_passive_solid(rho)
        compliance_history = []
        prev_comp = None
        stable_count = 0

        print('\n' + '=' * 60)
        print('3D Mori-Tanaka Material Interpolation Optimization (Auto-Stop)')
        print('=' * 60)

        if self.volfrac_eff > self.volfrac + 1e-12:
            print(f'  [CONSTRAINT] Volume target increased from {self.volfrac:.4f} to {self.volfrac_eff:.4f} to keep support/load regions solid.')

        for it in range(n_iter):
            # Gradual projection blend to avoid volume spike.
            if self.member_projector is not None:
                blend_start = max(8, int(0.10 * self.max_iterations))
                blend_end = max(blend_start + 5, int(0.25 * self.max_iterations))
                if it < blend_start:
                    rho_phys = self._apply_density_filter(rho)
                elif it < blend_end:
                    alpha = float(it - blend_start) / float(max(blend_end - blend_start, 1))
                    self.member_projector.set_beta(self._projection_beta(it))
                    rho_filt = self._apply_density_filter(rho)
                    rho_proj = self.member_projector.apply(rho, target_volume=self.volfrac_eff, rho_min=self.rho_min)
                    rho_phys = (1.0 - alpha) * rho_filt + alpha * rho_proj
                else:
                    self.member_projector.set_beta(self._projection_beta(it))
                    rho_phys = self.member_projector.apply(rho, target_volume=self.volfrac_eff, rho_min=self.rho_min)
            else:
                rho_phys = self._apply_density_filter(rho)
            self._enforce_passive_solid(rho_phys)

            e_eff_elem = self._effective_modulus_mori_tanaka(rho_phys)
            
            # Pass effective modulus scaling directly to FEA solver
            scale = e_eff_elem / float(max(self.material.E0, 1e-12))
            
            # If there are passive solid elements, ensure their scale is 1.0
            if np.any(self.passive_solid):
                scale[self.passive_solid] = 1.0

            u = self.fea.solve_with_modulus(scale, forces, fixed_dofs)
            compliance = float(self.fea.calculate_compliance_with_modulus(u, scale))
            compliance_history.append(compliance)

            tensor = self._compute_homogenized_tensor(rho_phys)
            dEdrho = self._effective_modulus_derivative(rho_phys)
            dc, dv = self._compute_element_sensitivities(u, rho_phys, dEdrho)
            dc = self._apply_sensitivity_filter(dc, rho_phys)
            dv = self._apply_sensitivity_filter(dv, rho_phys)

            rho_new = self._oc_update(rho, dc, dv)
            self._enforce_passive_solid(rho_new)
            rho_change = float(np.max(np.abs(rho_new - rho)))
            comp_rel = abs(compliance - prev_comp) / max(abs(compliance), 1e-12) if prev_comp is not None else np.inf
            prev_comp = compliance
            rho = rho_new

            if (it + 1) >= self.min_iterations and rho_change < self.change_tol and comp_rel < self.comp_tol:
                stable_count += 1
            else:
                stable_count = 0

            vol = float(np.mean(rho_phys))
            if visualizer is not None:
                try:
                    rho_vis = np.where(rho_phys >= 0.5, 1.0, self.rho_min)
                    self._enforce_passive_solid(rho_vis)
                    visualizer.update(it, rho_vis.copy(), compliance, vol)
                except Exception as e:
                    print(f'  [VIEWER] {e}')

            print(f'  Iteration {it + 1}: C={compliance:.6e}, V={vol:.4f}, dR={rho_change:.3e}, Ani={tensor["anisotropy"]:.3f}')

            if stable_count >= self.stall_patience:
                print(f'  [CONVERGED] Objective stabilized for {self.stall_patience} iterations; stopping at {it + 1}.')
                break

        return rho, compliance_history


# Backward-compatible alias
Homogenization3DOptimizer = MoriTanaka3DOptimizer

