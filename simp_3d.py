import time
import numpy as np
from fea_solver_3d import FEASolver3D
from density_filter_3d import DensityFilter3D, MinimumMemberSizeProjection3D, auto_filter_radius_from_mesh
from mma_solver import MMASolver, pnorm_stress


class SIMP3DOptimizer:
    """3D SIMP optimization with stress constraint, MMA/OC update, p-norm stress aggregation,
    beta-ramped projection, and convergence checks."""

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
        self.objective = str(config.get('objective', 'compliance')).lower()
        self.volume_fraction = float(config.get('volume_fraction', config.get('volfrac', 0.5)))

        wr = config.get('weight_reduction_percent', config.get('target_weight_reduction_percent', None))
        if self.objective == 'weight' and wr is not None:
            try:
                self.volume_fraction = 1.0 - (float(wr) / 100.0)
            except Exception:
                pass

        self.volume_fraction = float(np.clip(self.volume_fraction, 0.05, 1.0))
        self.penalty = float(config.get('penalization', config.get('penalty', 3.0)))
        self.max_iterations = int(config.get('max_iterations_auto', config.get('iterations', 300)))
        self.min_iterations = int(max(10, config.get('min_iterations_auto', 25)))
        self.change_tol = float(config.get('density_change_tolerance', 1e-3))
        self.comp_tol = float(config.get('compliance_tolerance', 5e-4))
        self.stall_patience = int(max(3, config.get('stall_patience', 8)))

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

        # §4.5: Filter radius continuation — start with larger radius to prevent
        # checkerboard, linearly reduce to target over first 50% of iterations.
        self.filter_radius_start_factor = float(config.get('filter_radius_start_factor', 1.5))
        self.filter_radius_target = self.filter_radius
        self.filter_radius_initial = self.filter_radius * self.filter_radius_start_factor
        self._filter_radius_continuation = bool(config.get('filter_radius_continuation', True))
        self._last_filter_radius = None  # Track to avoid unnecessary filter rebuilds
        self.filter = DensityFilter3D(self.fea.nodes, self.fea.elements, self.filter_radius)

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

        self.use_stress_constraint = bool(config.get('use_stress_constraint', True))
        self.safety_factor = float(config.get('safety_factor', 1.5))
        self.stress_penalty_weight = float(config.get('stress_penalty_weight', 2.0))
        self.use_stress_al = bool(config.get('use_stress_augmented_lagrangian', True))
        self.stress_al_mu = float(max(config.get('stress_al_mu0', 8.0), 1e-9))
        self.stress_al_mu_growth = float(max(config.get('stress_al_mu_growth', 1.10), 1.0))
        self.stress_al_mu_max = float(max(config.get('stress_al_mu_max', 1e4), self.stress_al_mu))
        if hasattr(material, 'get_allowable_stress'):
            self.allowable_stress = float(material.get_allowable_stress(self.safety_factor))
        else:
            ys = float(getattr(material, 'yield_strength', config.get('yield_strength', 250e6)))
            self.allowable_stress = ys / max(self.safety_factor, 1e-9)
        self._stress_lambda = np.zeros(self.n_elements, dtype=np.float64)

        # P-norm stress aggregation
        self.use_pnorm_stress = bool(config.get('use_pnorm_stress', True))
        self.pnorm_p = float(config.get('pnorm_exponent', 8.0))

        # MMA solver (alternative to OC for multi-constraint handling)
        self.use_mma = bool(config.get('use_mma', False))
        self.mma_solver = None

        passive = config.get('passive_solid_elements', None)
        self.passive_solid = np.zeros(self.n_elements, dtype=bool)
        if passive is not None:
            p = np.asarray(passive, dtype=bool).reshape(-1)
            if p.size == self.n_elements:
                self.passive_solid = p.copy()

        passive_frac = float(np.mean(self.passive_solid)) if self.passive_solid.size else 0.0
        min_target = passive_frac + (1.0 / max(self.n_elements, 1))
        self.volume_fraction_eff = float(np.clip(max(self.volume_fraction, min_target), 0.05, 1.0))

        self.rho = np.ones(self.n_elements, dtype=np.float64) * self.volume_fraction_eff
        self._enforce_passive_solid(self.rho)
        self.rho_old = self.rho.copy()

        elems = np.asarray(self.fea.elements, dtype=np.int64)
        npe = int(elems.shape[1])
        self.elem_dofs = (3 * elems[:, :, None] + np.arange(3, dtype=np.int64)).reshape(self.n_elements, 3 * npe)
        rho_probe = np.ones(self.n_elements, dtype=np.float64)
        scale_at_1 = float(self.material.get_density_scale(1.0, 1.0))
        self.ke0 = self.fea._ke_unit * scale_at_1

        self.npe = int(elems.shape[1])
        dflt_interval = 5 if self.npe >= 10 else 1
        self.stress_eval_interval = int(max(1, config.get('stress_eval_interval', dflt_interval)))
        self.log_iteration_timing = bool(config.get('log_iteration_timing', True))
        self._cached_stress_scale = np.ones(self.n_elements, dtype=np.float64)
        self._cached_max_vm = 0.0
        self._cached_violate_frac = 0.0
        self._cached_pnorm_value = 0.0

        # Initialize MMA solver if requested
        if self.use_mma:
            n_constraints = 1  # volume constraint
            if self.use_stress_constraint and self.use_pnorm_stress:
                n_constraints = 2  # volume + p-norm stress
            self.mma_solver = MMASolver(
                n=self.n_elements,
                m=n_constraints,
                x_min=np.full(self.n_elements, self.rho_min),
                x_max=np.ones(self.n_elements),
            )

    def _current_penalty(self, iteration):
        """Penalty continuation: ramp p from 1 → target over first 40% of iterations.

        Starting with p=1 allows the optimizer to explore broad material distributions
        before forcing binary-like designs at higher penalties. This dramatically
        improves convergence to good topologies and reduces mesh-dependency artifacts.
        Reference: Bendsøe & Sigmund, Topology Optimization (2003).
        """
        p_start = 1.0
        p_end = self.penalty
        ramp_frac = float(self.config.get('penalty_ramp_fraction', 0.40))
        ramp_iters = int(ramp_frac * self.max_iterations)
        if ramp_iters <= 0 or iteration >= ramp_iters:
            return p_end
        t = float(iteration) / float(ramp_iters)
        return p_start + (p_end - p_start) * t

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

    def _sensitivity_analysis(self, u, rho, p_current=None):
        if p_current is None:
            p_current = self.penalty
        ue = u[self.elem_dofs]
        ce0 = np.einsum('ni,nij,nj->n', ue, self.ke0, ue)
        scale = max(1.0 - float(getattr(self.material, 'rho_min', 0.0)), 1e-12)
        rho_eff = np.maximum(rho, self.rho_min)
        sens = -scale * p_current * (rho_eff ** (p_current - 1.0)) * ce0

        # ke0 is assembled at rho=1.0, so element stiffness scales with
        # (rho_min + (1-rho_min)*rho^penalty) for SIMP material interpolation.
        comp_scale = float(getattr(self.material, 'rho_min', 0.0)) + scale * (rho_eff ** p_current)
        compliance = float(np.sum(comp_scale * ce0))

        sens_f = self.filter.apply_sensitivity(sens, rho, rho_min=self.rho_min)
        return sens_f, compliance

    def _stress_constraint_scale(self, u, rho, p_current=None):
        """Compute stress constraint scaling using Gauss-point evaluation and p-norm aggregation."""
        if p_current is None:
            p_current = self.penalty
        # Use Gauss-point stress evaluation for accurate peak stress capture
        vm = self.fea.calculate_stress_all_gauss(u, rho, p_current)

        # P-norm stress aggregation for smooth differentiable constraint
        if self.use_pnorm_stress:
            pn_value, dpn_dvm, pn_constraint = pnorm_stress(
                vm, self.allowable_stress, pn=self.pnorm_p
            )
            self._cached_pnorm_value = pn_value
            # Use p-norm based scaling: smoother than per-element comparison
            pn_ratio = pn_value / max(self.allowable_stress, 1e-12)
            # Scale sensitivities based on how close p-norm is to allowable
            if pn_ratio > 1.0:
                global_scale = 1.0 + self.stress_penalty_weight * (pn_ratio - 1.0) ** 2
            else:
                global_scale = 1.0
            # Per-element refinement: elements with high stress get extra penalty
            ratio = vm / max(self.allowable_stress, 1e-12)
            violation = np.maximum(ratio - 1.0, 0.0)
            scale = global_scale + self.stress_penalty_weight * (violation ** 2)
        else:
            ratio = vm / max(self.allowable_stress, 1e-12)
            violation = np.maximum(ratio - 1.0, 0.0)
            pn_value = float(np.max(vm))
            self._cached_pnorm_value = pn_value
            scale = np.ones_like(vm)

        if self.use_stress_al:
            mult = np.maximum(self._stress_lambda + self.stress_al_mu * violation, 0.0)
            scale = scale + mult
            self._stress_lambda = np.maximum(0.0, self._stress_lambda + self.stress_al_mu * violation)
            if float(np.max(violation)) > 0.02:
                self.stress_al_mu = min(self.stress_al_mu * self.stress_al_mu_growth, self.stress_al_mu_max)

        max_vm = float(np.max(vm)) if vm.size else 0.0
        violate_frac = float(np.mean(violation > 0.0)) if violation.size else 0.0
        return scale, max_vm, violate_frac

    def _update_density_oc(self, sensitivities, volume_constraint, iteration):
        move = 0.2
        dc = np.minimum(np.asarray(sensitivities, dtype=np.float64), -1e-16)

        l1 = 1e-12
        l2 = 1e12
        rho_current = self.rho
        target_sum = float(volume_constraint) * float(self.n_elements)

        for _ in range(80):
            lam = 0.5 * (l1 + l2)
            factor = np.sqrt(np.maximum(0.0, -dc / lam))
            rho_trial = rho_current * factor
            rho_new = np.minimum(
                1.0,
                np.maximum(
                    self.rho_min,
                    np.minimum(rho_current + move, np.maximum(rho_current - move, rho_trial)),
                ),
            )

            self._enforce_passive_solid(rho_new)

            if float(np.sum(rho_new)) > target_sum:
                l1 = lam
            else:
                l2 = lam

            if (l2 - l1) / max(l2 + l1, 1e-12) < 1e-6:
                break

        # Gradual projection blend: filter-only early, then smoothly transition
        # to member projection to avoid the volume spike from a hard switch.
        if self.member_projector is not None:
            blend_start = max(8, int(0.10 * self.max_iterations))
            blend_end = max(blend_start + 5, int(0.25 * self.max_iterations))
            if iteration < blend_start:
                # Pure density filter — let topology form first
                rho_new = self.filter.apply_density(rho_new)
            elif iteration < blend_end:
                # Linear blend from filter to projection
                alpha = float(iteration - blend_start) / float(max(blend_end - blend_start, 1))
                self.member_projector.set_beta(self._projection_beta(iteration))
                rho_filt = self.filter.apply_density(rho_new)
                rho_proj = self.member_projector.apply(rho_new, target_volume=volume_constraint, rho_min=self.rho_min)
                rho_new = (1.0 - alpha) * rho_filt + alpha * rho_proj
            else:
                # Full projection
                self.member_projector.set_beta(self._projection_beta(iteration))
                rho_new = self.member_projector.apply(rho_new, target_volume=volume_constraint, rho_min=self.rho_min)
        else:
            rho_new = self.filter.apply_density(rho_new)

        self._enforce_passive_solid(rho_new)

        # §4.6: Volume correction after projection — prevent volume drift
        actual_vol = float(np.mean(rho_new))
        if abs(actual_vol - volume_constraint) > 0.005 and actual_vol > 1e-12:
            rho_new *= volume_constraint / actual_vol
            rho_new = np.clip(rho_new, self.rho_min, 1.0)
            self._enforce_passive_solid(rho_new)

        return rho_new

    def _compute_adjoint_stress_sensitivity(self, u, rho, dpn_dvm, vm, p_current=None):
        """Compute exact stress constraint gradient via adjoint method (vectorized).

        Solves K·λ = ∂PN/∂u  (adjoint system), then computes per-element:
          ∂PN/∂ρ_e = dE/dρ_e · [∂σ_vm/∂σ^T · D · B_e · u_e · dpn_dvm_e  −  λ_e^T · K_e0 · u_e]
        Fully vectorized — replaces original Python for-loops over n_elements.
        """
        if p_current is None:
            p_current = self.penalty
        uu = np.asarray(u, dtype=np.float64)
        rho_eff = np.maximum(np.asarray(rho, dtype=np.float64), self.rho_min)
        rho_min_mat = float(getattr(self.material, 'rho_min', 0.0))
        E_all = (rho_min_mat + (1.0 - rho_min_mat) * (rho_eff ** p_current)) * self.material.E0
        D = self.fea._d_base

        valid = (np.abs(self.fea._det_centroid) >= 1e-14) & (dpn_dvm > 1e-30)
        B_all = self.fea._bm_centroid
        ue_all = uu[self.fea.elem_dofs]

        # Step 1: Vectorized adjoint RHS
        strain_all = np.einsum('eij,ej->ei', B_all, ue_all)
        stress_all = np.einsum('ij,ej->ei', D, strain_all) * E_all[:, None]

        sx, sy, sz = stress_all[:, 0], stress_all[:, 1], stress_all[:, 2]
        txy, tyz, tzx = stress_all[:, 3], stress_all[:, 4], stress_all[:, 5]
        vm_all = np.sqrt(np.maximum(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                                           + 6.0*(txy**2 + tyz**2 + tzx**2)), 1e-60))

        inv_2vm = 1.0 / (2.0 * vm_all)
        dvm_ds_all = np.column_stack([
            (2*sx - sy - sz) * inv_2vm,
            (2*sy - sx - sz) * inv_2vm,
            (2*sz - sx - sy) * inv_2vm,
            3.0 * txy / vm_all,
            3.0 * tyz / vm_all,
            3.0 * tzx / vm_all,
        ])

        weighted_dvm = dvm_ds_all * dpn_dvm[:, None]
        inner = np.einsum('ij,ej->ei', D, weighted_dvm) * E_all[:, None]
        rhs_per_elem = np.einsum('eji,ej->ei', B_all, inner)
        rhs_per_elem[~valid] = 0.0

        adjoint_rhs = np.zeros(self.fea.n_dofs, dtype=np.float64)
        np.add.at(adjoint_rhs, self.fea.elem_dofs, rhs_per_elem)

        # Step 2: Solve adjoint system
        lam = self.fea.solve(rho, adjoint_rhs, self._fixed_dofs, p_current)

        # Step 3: Vectorized per-element sensitivity
        mat_scale = max(1.0 - rho_min_mat, 1e-12)
        dE_drho = mat_scale * p_current * (rho_eff ** (p_current - 1.0)) * self.material.E0

        lam_arr = np.asarray(lam, dtype=np.float64)
        lam_all = lam_arr[self.fea.elem_dofs]

        ke0_u = np.einsum('eij,ej->ei', self.ke0, ue_all)
        indirect = -np.einsum('ei,ei->e', lam_all, ke0_u) * dE_drho

        E_safe = np.maximum(E_all, 1e-30)
        dvm_dot_sigma = np.einsum('ei,ei->e', dvm_ds_all, stress_all)
        direct = dpn_dvm * (dE_drho / E_safe) * dvm_dot_sigma
        direct[~valid | (vm_all < 1e-30)] = 0.0

        return direct + indirect

    def _update_density_mma(self, u, sensitivities, rho, volume_constraint, iteration, compliance, p_current=None):
        """MMA-based density update with volume and optional p-norm stress constraints."""
        if p_current is None:
            p_current = self.penalty
        # Objective: minimize compliance
        f0val = compliance  # use actual compliance (exact)
        df0dx = sensitivities.copy()

        constraints = []
        constraint_grads = []

        # Constraint 1: volume - target <= 0
        vol_constraint = float(np.mean(rho)) - volume_constraint
        dvol_drho = np.ones(self.n_elements, dtype=np.float64) / self.n_elements
        constraints.append(vol_constraint)
        constraint_grads.append(dvol_drho)

        # Constraint 2: p-norm stress (if enabled) with adjoint sensitivity
        if self.use_stress_constraint and self.use_pnorm_stress:
            vm = self.fea.calculate_stress_all_gauss(u, rho, p_current)
            pn_value, dpn_dvm, pn_constraint = pnorm_stress(
                vm, self.allowable_stress, pn=self.pnorm_p
            )
            self._cached_pnorm_value = pn_value
            # Adjoint-based stress sensitivity (exact gradient)
            dstress_drho = self._compute_adjoint_stress_sensitivity(u, rho, dpn_dvm, vm, p_current)
            constraints.append(pn_constraint)
            constraint_grads.append(dstress_drho)

        fval = np.array(constraints, dtype=np.float64)
        dfdx = np.array(constraint_grads, dtype=np.float64)

        # MMA update
        rho_new = self.mma_solver.update(rho, f0val, df0dx, fval, dfdx)
        rho_new = np.clip(rho_new, self.rho_min, 1.0)

        # Apply projection
        if self.member_projector is not None:
            blend_start = max(8, int(0.10 * self.max_iterations))
            blend_end = max(blend_start + 5, int(0.25 * self.max_iterations))
            if iteration < blend_start:
                rho_new = self.filter.apply_density(rho_new)
            elif iteration < blend_end:
                alpha = float(iteration - blend_start) / float(max(blend_end - blend_start, 1))
                self.member_projector.set_beta(self._projection_beta(iteration))
                rho_filt = self.filter.apply_density(rho_new)
                rho_proj = self.member_projector.apply(rho_new, target_volume=volume_constraint, rho_min=self.rho_min)
                rho_new = (1.0 - alpha) * rho_filt + alpha * rho_proj
            else:
                self.member_projector.set_beta(self._projection_beta(iteration))
                rho_new = self.member_projector.apply(rho_new, target_volume=volume_constraint, rho_min=self.rho_min)
        else:
            rho_new = self.filter.apply_density(rho_new)

        self._enforce_passive_solid(rho_new)

        # §4.6: Volume correction after projection — prevent volume drift
        actual_vol = float(np.mean(rho_new))
        if abs(actual_vol - volume_constraint) > 0.005 and actual_vol > 1e-12:
            rho_new *= volume_constraint / actual_vol
            rho_new = np.clip(rho_new, self.rho_min, 1.0)
            self._enforce_passive_solid(rho_new)

        return rho_new

    def optimize(self, forces, fixed_dofs, visualizer=None):
        solver_label = 'MMA' if self.use_mma else 'OC'
        stress_label = 'p-norm' if self.use_pnorm_stress else 'per-element'
        print('\n' + '=' * 60)
        print(f'3D SIMP Optimization (Stress: {stress_label}, Update: {solver_label})')
        print('=' * 60)

        if self.volume_fraction_eff > self.volume_fraction + 1e-12:
            print(f'  [CONSTRAINT] Volume target increased from {self.volume_fraction:.4f} to {self.volume_fraction_eff:.4f} to keep support/load regions solid.')

        if self.npe >= 10 and self.n_elements > int(self.config.get('tet10_warning_elements', 12000)):
            print(f'  [PERF] Large Tet10 model ({self.n_elements} elements). Iteration 1 can take several minutes on direct solver.')
            print('  [PERF] Use lower remesh target or set max_3d_elements in config for interactive runs.')

        compliance_history = []
        prev_comp = None
        stable_count = 0
        self._fixed_dofs = fixed_dofs  # store for adjoint stress sensitivity

        for iteration in range(self.max_iterations):
            # Penalty continuation: ramp p from 1 → target over first 40% of iterations
            p_current = self._current_penalty(iteration)

            # §4.5: Filter radius continuation — linearly reduce from initial
            # to target over first 50% of iterations. Rebuild filter at discrete
            # steps (every 10% of max_iterations) to amortize cKDTree cost.
            if self._filter_radius_continuation and self.filter_radius_start_factor > 1.01:
                ramp_end = int(0.50 * self.max_iterations)
                if iteration < ramp_end:
                    t_frac = float(iteration) / max(ramp_end, 1)
                    r_current = self.filter_radius_initial + t_frac * (self.filter_radius_target - self.filter_radius_initial)
                else:
                    r_current = self.filter_radius_target
                # Only rebuild filter at discrete steps to avoid overhead
                step = max(1, int(0.10 * self.max_iterations))
                if self._last_filter_radius is None or (iteration % step == 0 and abs(r_current - self._last_filter_radius) > 1e-9):
                    self.filter = DensityFilter3D(self.fea.nodes, self.fea.elements, r_current)
                    self._last_filter_radius = r_current

            print(f'\nIteration {iteration + 1} (p={p_current:.2f})')

            self._enforce_passive_solid(self.rho)
            t0 = time.perf_counter()
            u = self.fea.solve(self.rho, forces, fixed_dofs, p_current)
            t1 = time.perf_counter()

            sensitivities, compliance = self._sensitivity_analysis(u, self.rho, p_current)
            t2 = time.perf_counter()
            compliance_history.append(compliance)
            volume_fraction = float(self.rho.mean())
            t3 = t2
            if np.any(self.passive_solid):
                sensitivities[self.passive_solid] = np.minimum(sensitivities[self.passive_solid], -1.0)

            max_vm = 0.0
            violate_frac = 0.0
            pnorm_val = 0.0
            if self.use_stress_constraint:
                do_stress = (iteration % self.stress_eval_interval) == 0
                # §3.3: Skip stress evaluation in early iterations when penalty
                # is low — stress constraints are usually inactive during the
                # topology-forming phase and evaluating them wastes compute.
                if iteration < int(0.30 * self.max_iterations) and p_current < 2.0:
                    do_stress = False
                if do_stress:
                    stress_scale, max_vm, violate_frac = self._stress_constraint_scale(u, self.rho, p_current)
                    self._cached_stress_scale = stress_scale
                    self._cached_max_vm = max_vm
                    self._cached_violate_frac = violate_frac
                    pnorm_val = self._cached_pnorm_value
                else:
                    stress_scale = self._cached_stress_scale
                    max_vm = self._cached_max_vm
                    violate_frac = self._cached_violate_frac
                    pnorm_val = self._cached_pnorm_value

                if not self.use_mma:
                    sensitivities = sensitivities * stress_scale

            if self.use_mma and self.mma_solver is not None:
                self.rho = self._update_density_mma(u, sensitivities, self.rho, self.volume_fraction_eff, iteration, compliance, p_current)
            else:
                self.rho = self._update_density_oc(sensitivities, self.volume_fraction_eff, iteration)
            t4 = time.perf_counter()

            max_change = float(np.max(np.abs(self.rho - self.rho_old)))
            comp_rel = abs(compliance - prev_comp) / max(abs(compliance), 1e-12) if prev_comp is not None else np.inf
            self.rho_old = self.rho.copy()
            prev_comp = compliance

            if (iteration + 1) >= self.min_iterations and max_change < self.change_tol and comp_rel < self.comp_tol:
                stable_count += 1
            else:
                stable_count = 0

            print(f'  Compliance: {compliance:.6e}')
            print(f'  Volume: {volume_fraction:.4f} (target={self.volume_fraction_eff:.4f})')
            if self.use_stress_constraint:
                tag = '' if (iteration % self.stress_eval_interval) == 0 else f' [reused x{self.stress_eval_interval}]'
                pn_tag = f', PN({self.pnorm_p:.0f})={pnorm_val:.3e} Pa' if self.use_pnorm_stress else ''
                print(f'  Max von Mises: {max_vm:.3e} Pa (allow={self.allowable_stress:.3e} Pa){pn_tag}, Viol: {100.0 * violate_frac:.1f}%{tag}')
            print(f'  Max change: {max_change:.6f}, dC_rel: {comp_rel:.3e}')
            if self.log_iteration_timing:
                print(f'  Timing [s]: solve={t1-t0:.2f}, comp+sensitivity={t2-t1:.2f}, update={t4-t3:.2f}, total={t4-t0:.2f}')

            if visualizer is not None:
                try:
                    visualizer.update(iteration, self.rho, compliance, volume_fraction)
                except Exception as e:
                    print(f'  [VIEWER] {e}')

            if stable_count >= self.stall_patience:
                print(f'  [CONVERGED] Objective stabilized for {self.stall_patience} iterations; stopping at {iteration + 1}.')
                break

        print('\n3D SIMP Optimization Completed')
        return self.rho, compliance_history

    def run_verification_fea(self, forces, fixed_dofs, threshold=0.5):
        """Run verification FEA on the binary-thresholded (manufacturable) design.

        The optimization produces a continuous density field ρ ∈ [ρ_min, 1].
        The manufactured part is binary: elements with ρ >= threshold are solid,
        the rest are void. This method re-runs a full FEA on that binary design
        to report physically meaningful compliance, displacement, and stress.

        Returns
        -------
        dict with keys:
            'compliance_continuous' : float  — compliance of the gray-density result
            'compliance_binary'     : float  — compliance of the thresholded design
            'max_disp_continuous'   : float  — max displacement (continuous)
            'max_disp_binary'       : float  — max displacement (binary)
            'max_vm_binary'         : float  — max von Mises stress (binary)
            'rho_binary'            : ndarray — the binary density vector
            'threshold'             : float  — threshold used
        """
        rho_cont = np.copy(self.rho)
        rho_binary = np.where(rho_cont >= threshold, 1.0, self.rho_min)
        self._enforce_passive_solid(rho_binary)

        print(f'\n{"="*60}')
        print(f'  Verification FEA on Binary Design (threshold={threshold:.2f})')
        print(f'{"="*60}')
        n_solid = int(np.sum(rho_binary >= threshold))
        print(f'  Solid elements: {n_solid}/{self.n_elements} ({100.0*n_solid/max(self.n_elements,1):.1f}%)')

        # Continuous-density result
        u_cont = self.fea.solve(rho_cont, forces, fixed_dofs, self.penalty)
        comp_cont = self.fea.calculate_compliance(u_cont, rho_cont, self.penalty)
        max_disp_cont = float(np.max(np.abs(u_cont)))

        # Binary-threshold result
        u_bin = self.fea.solve(rho_binary, forces, fixed_dofs, self.penalty)
        comp_bin = self.fea.calculate_compliance(u_bin, rho_binary, self.penalty)

        # Report max displacement from solid nodes only — void elements have
        # near-zero stiffness and can accumulate fictitious large displacements
        # in disconnected regions, inflating np.max(np.abs(u)).
        solid_elem_mask = (rho_binary >= threshold)
        solid_node_indices = np.unique(self.fea.elements[solid_elem_mask].flatten())
        solid_dofs = np.concatenate([
            3 * solid_node_indices,
            3 * solid_node_indices + 1,
            3 * solid_node_indices + 2,
        ])
        max_disp_bin = float(np.max(np.abs(u_bin[solid_dofs])))

        vm_bin = self.fea.calculate_stress_all_gauss(u_bin, rho_binary, self.penalty)
        max_vm_bin = float(np.max(vm_bin)) if vm_bin.size else 0.0

        # Report comparison
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
