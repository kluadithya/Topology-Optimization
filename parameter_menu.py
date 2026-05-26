"""
Parameter Definition Menu System
Handles: mesh settings, objective function, optimization parameters
Supports Standard (current defaults) and Advanced (all params editable) modes.
"""


class ParameterMenu:
    """Interactive parameter definition menu"""

    def __init__(self):
        self.mesh_settings = {}
        self.objective_function = 'stiffness'
        self.opt_parameters = {}

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _prompt(label, default, cast=float, allow_none=False):
        """Prompt for a value with a shown default.  Enter keeps the default."""
        suffix = f" (default {default}): "
        raw = input(label + suffix).strip()
        if not raw:
            if allow_none and default is None:
                return None
            return cast(default) if default is not None else None
        try:
            return cast(raw)
        except Exception:
            print(f"  [WARN] Invalid input, using default: {default}")
            return cast(default) if default is not None else None

    @staticmethod
    def _prompt_yn(label, default_yes=True):
        """Prompt for y/n with a default."""
        tag = "y" if default_yes else "n"
        raw = input(f"{label} (y/n, default {tag}): ").strip().lower()
        if not raw:
            return default_yes
        return raw == 'y'

    @staticmethod
    def _prompt_choice(label, options_map, default_key):
        """Prompt for a choice from a dict like {'auto': 'auto', ...}."""
        keys = list(options_map.keys())
        opts = "/".join(keys)
        raw = input(f"{label} [{opts}] (default {default_key}): ").strip().lower()
        if not raw or raw not in options_map:
            return options_map[default_key]
        return options_map[raw]

    # ── Standard menus (unchanged from before) ──────────────────────────

    def display_mesh_menu(self):
        """Define mesh refinement settings"""
        print("\n" + "=" * 70)
        print("MESH SETTINGS")
        print("=" * 70)

        print("\nMesh Refinement Options:")
        print("  1. Use current mesh (no refinement)")
        print("  2. Refine mesh uniformly")
        print("  3. Adaptive refinement (high-quality zones)")
        print("  4. Custom refinement parameters")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            self.mesh_settings['refinement'] = 'none'
            print("[OK] Using original mesh")
        elif choice == '2':
            factor = int(input("Refinement factor (2-5): "))
            self.mesh_settings['refinement'] = 'uniform'
            self.mesh_settings['refinement_factor'] = factor
            print(f"[OK] Uniform refinement: {factor}x")
        elif choice == '3':
            self.mesh_settings['refinement'] = 'adaptive'
            print("[OK] Adaptive refinement enabled")
        else:
            self.mesh_settings['refinement'] = 'none'
            print("[OK] Using original mesh")

        return True

    def display_objective_menu(self):
        """Define optimization objective"""
        print("\n" + "=" * 70)
        print("OBJECTIVE FUNCTION")
        print("=" * 70)

        print("\nSelect optimization objective:")
        print("  1. Stiffness Maximization")
        print("  2. Weight Minimization (under stress constraint)")
        print("  3. Stress Minimization (uniform stress)")

        choice = input("\nSelect objective (1-3): ").strip()

        if choice == '1':
            self.objective_function = 'stiffness'
            print("[OK] Objective: Maximize stiffness")
        elif choice == '2':
            self.objective_function = 'weight'
            self.opt_parameters['stress_limit'] = float(input("Stress limit (MPa): "))
            print(f"[OK] Objective: Minimize weight with {self.opt_parameters['stress_limit']} MPa limit")
        elif choice == '3':
            self.objective_function = 'stress'
            print("[OK] Objective: Minimize peak stress")
        else:
            self.objective_function = 'stiffness'
            print("[OK] Objective: Maximize stiffness")

        return True

    # ── Core optimization menu (standard section) ───────────────────────

    def display_optimization_menu(self, objective_function='stiffness',
                                  preset_volume_fraction=None,
                                  defer_volume_fraction=False,
                                  settings_mode='standard',
                                  optimization_method='SIMP'):
        """Define optimization parameters (auto-iteration mode).

        When *settings_mode* is ``'advanced'``, additional sub-menus are
        presented after the standard prompts so the user can override every
        hardcoded default.
        """
        print("\n" + "=" * 70)
        print("OPTIMIZATION PARAMETERS")
        print("=" * 70)

        # ── Volume fraction (same for both modes) ───────────────────────
        print("\nMaterial Usage Target:")
        if defer_volume_fraction:
            self.opt_parameters['volume_fraction'] = 0.3  # placeholder default
            print("  [INFO] Volume fraction / mass reduction will be set AFTER stress analysis in the GUI.")
            print("  [INFO] You will see a stress heatmap first, then choose how much material to remove.")
        elif objective_function == 'weight' and preset_volume_fraction is not None:
            self.opt_parameters['volume_fraction'] = max(0.01, min(1.0, float(preset_volume_fraction)))
            print(f"[OK] Volume fraction auto-set from weight target: {self.opt_parameters['volume_fraction']:.3f}")
        else:
            print("  Volume fraction = fraction of the full design space kept as material")
            vf = float(input("Target volume fraction (0.0-1.0, default 0.3): "))
            self.opt_parameters['volume_fraction'] = max(0.01, min(1.0, vf))

        if self.opt_parameters['volume_fraction'] > 0.55:
            print("  [WARN] High volume fraction (>0.55) can produce bulky topologies with localized voids.")
            print("  [WARN] For clearer load paths, consider 0.30-0.45 when possible.")

        # ── Standard auto-settings ──────────────────────────────────────
        print("\nIteration mode:")
        print("  [AUTO] Solver will iterate until convergence (objective completion), with internal safety cap.")

        print("\nFilter Settings:")
        self.opt_parameters['auto_filter_radius'] = True
        self.opt_parameters['filter_radius_factor'] = 3.5
        print("  [AUTO] Filter radius is computed from mesh element size (3.5x median tet edge length)")

        self.opt_parameters['threshold'] = 0.5
        print("  [AUTO] Density threshold is derived from target volume/weight (no manual input)")

        self.opt_parameters['penalization'] = 3.0
        print("  [AUTO] Penalization factor fixed at 3.0")

        # ── Manufacturability (standard prompts) ────────────────────────
        print("\nManufacturability:")
        nd_dist_in = input("Protect support/load region distance [mm] (>=0, default 1.0): ").strip()
        try:
            nd_dist_mm = float(nd_dist_in) if nd_dist_in else 1.0
        except Exception:
            nd_dist_mm = 1.0
        self.opt_parameters['non_design_distance_mm'] = max(0.0, nd_dist_mm)
        self.opt_parameters['non_design_node_layers'] = max(1, int(self.opt_parameters['non_design_distance_mm'] + 0.999999))

        min_mm_in = input("Minimum member size in model units (optional, Enter to skip): ").strip()
        if min_mm_in:
            try:
                self.opt_parameters['min_member_size_mm'] = max(float(min_mm_in), 0.0)
            except Exception:
                pass

        use_proj = input("Enable minimum member size projection? (y/n, default y): ").strip().lower()
        self.opt_parameters['use_min_member_projection'] = (use_proj != 'n')
        if self.opt_parameters['use_min_member_projection']:
            factor_in = input("Minimum member size factor (>=1.0, default 1.5): ").strip()
            try:
                factor = float(factor_in) if factor_in else 1.5
            except Exception:
                factor = 1.5
            self.opt_parameters['min_member_size_factor'] = max(1.0, factor)
            self.opt_parameters['min_member_projection_beta_start'] = 1.0
            self.opt_parameters['min_member_projection_beta_end'] = 32.0
            # Backward-compatible single-beta field.
            self.opt_parameters['min_member_projection_beta'] = 32.0
            print('  [AUTO] Projection beta schedule fixed: start=1.0, end=32.0')

        ovh_in = input("Enable overhang self-support filter? (y/n, default n): ").strip().lower()
        self.opt_parameters['enable_overhang_filter'] = (ovh_in == 'y')
        if self.opt_parameters['enable_overhang_filter']:
            ang_in = input("Critical overhang angle in deg (default 45): ").strip()
            axis_in = input("Build axis [x/y/z], default z: ").strip().lower()
            try:
                self.opt_parameters['overhang_angle_deg'] = float(ang_in) if ang_in else 45.0
            except Exception:
                self.opt_parameters['overhang_angle_deg'] = 45.0
            self.opt_parameters['build_axis'] = axis_in if axis_in in ('x', 'y', 'z') else 'z'

        encl_in = input("Run enclosed-void check on final result? (y/n, default y): ").strip().lower()
        self.opt_parameters['check_enclosed_voids'] = (encl_in != 'n')

        # Tet10-only runtime policy.

        print("[OK] Optimization parameters set")

        # ── ADVANCED sub-menus ──────────────────────────────────────────
        if settings_mode == 'advanced':
            self._display_advanced_menus(optimization_method)

        return True

    # ── Advanced sub-menus ──────────────────────────────────────────────

    def _display_advanced_menus(self, optimization_method='SIMP'):
        """Show grouped advanced parameter sub-menus."""
        print("\n" + "=" * 70)
        print("  ADVANCED SETTINGS")
        print("  Press Enter on any prompt to keep the default value")
        print("=" * 70)

        self._adv_solver_iteration()
        self._adv_penalization()
        self._adv_filter_projection()
        self._adv_stress_constraint()
        self._adv_post_processing()
        self._adv_method_specific(optimization_method)
        self._adv_mesh_limits()

        print("\n[OK] Advanced settings applied")

    # ·· Sub-menu: Solver & Iteration ····································

    def _adv_solver_iteration(self):
        print("\n── Solver & Iteration " + "─" * 48)

        self.opt_parameters['max_iterations_auto'] = int(
            self._prompt("  Max iterations", 300, cast=int))

        solver_map = {'auto': 'auto', 'direct': 'direct', 'iterative': 'iterative'}
        self.opt_parameters['linear_solver'] = self._prompt_choice(
            "  Linear solver", solver_map, 'auto')

        self.opt_parameters['iterative_solver_tol'] = self._prompt(
            "  Iterative solver tolerance", 1e-8)

        self.opt_parameters['iterative_solver_maxiter'] = int(
            self._prompt("  Iterative solver max iters", 2000, cast=int))

    # ·· Sub-menu: SIMP / Penalization ···································

    def _adv_penalization(self):
        print("\n── SIMP / Penalization " + "─" * 48)

        val = self._prompt("  Penalization exponent", 3.0)
        self.opt_parameters['penalization'] = max(1.0, float(val))
        print(f"  [SET] penalization = {self.opt_parameters['penalization']:.2f}")

    # ·· Sub-menu: Filter & Projection ···································

    def _adv_filter_projection(self):
        print("\n── Filter & Projection " + "─" * 47)

        auto = self._prompt_yn("  Auto filter radius?", default_yes=True)
        self.opt_parameters['auto_filter_radius'] = auto

        if auto:
            self.opt_parameters['filter_radius_factor'] = self._prompt(
                "  Filter radius factor (× median edge length)", 3.5)
        else:
            self.opt_parameters['filter_radius'] = self._prompt(
                "  Manual filter radius", 1.5)

        # Projection beta (only if projection is enabled)
        if self.opt_parameters.get('use_min_member_projection', True):
            self.opt_parameters['min_member_projection_beta_start'] = self._prompt(
                "  Projection beta start", 1.0)
            self.opt_parameters['min_member_projection_beta_end'] = self._prompt(
                "  Projection beta end", 32.0)
            # Backward-compatible single-beta = end value
            self.opt_parameters['min_member_projection_beta'] = \
                self.opt_parameters['min_member_projection_beta_end']

        # Density threshold override
        thr = self._prompt("  Density threshold override (Enter=auto)", None,
                           allow_none=True)
        if thr is not None:
            self.opt_parameters['threshold'] = float(thr)
            self.opt_parameters['threshold_source'] = 'manual'
        else:
            self.opt_parameters['threshold_source'] = 'auto'

    # ·· Sub-menu: Stress Constraint ·····································

    def _adv_stress_constraint(self):
        print("\n── Stress Constraint " + "─" * 49)

        enabled = self._prompt_yn("  Enable stress constraint?", default_yes=True)
        self.opt_parameters['use_stress_constraint'] = enabled

        if enabled:
            self.opt_parameters['safety_factor'] = max(
                0.1, self._prompt("  Safety factor", 1.5))

            self.opt_parameters['stress_penalty_weight'] = max(
                0.0, self._prompt("  Stress penalty weight", 2.0))

            use_al = self._prompt_yn("  Use augmented Lagrangian?", default_yes=True)
            self.opt_parameters['use_stress_augmented_lagrangian'] = use_al

            if use_al:
                self.opt_parameters['stress_al_mu0'] = self._prompt(
                    "  AL initial mu", 8.0)
                self.opt_parameters['stress_al_mu_growth'] = self._prompt(
                    "  AL mu growth rate", 1.10)
                self.opt_parameters['stress_al_mu_max'] = self._prompt(
                    "  AL mu max", 1e4)

            self.opt_parameters['stress_eval_interval'] = int(
                self._prompt("  Stress eval interval (iterations)", 5, cast=int))

    # ·· Sub-menu: Post-Processing ·······································

    def _adv_post_processing(self):
        print("\n── Post-Processing " + "─" * 51)

        self.opt_parameters['final_density_smoothing_passes'] = int(
            self._prompt("  Final density smoothing passes", 2, cast=int))

        self.opt_parameters['final_density_smoothing_blend'] = self._prompt(
            "  Final density smoothing blend (0-1)", 0.60)

        smooth = self._prompt_yn("  Smooth result surface?", default_yes=True)
        self.opt_parameters['smooth_result_surface'] = smooth

        if smooth:
            self.opt_parameters['result_surface_smooth_iterations'] = int(
                self._prompt("  Surface smooth iterations", 25, cast=int))

            self.opt_parameters['result_surface_relaxation'] = self._prompt(
                "  Surface relaxation factor", 0.08)

    # ·· Sub-menu: Method-Specific ·······································

    def _adv_method_specific(self, method='SIMP'):
        method_upper = str(method).upper()

        if method_upper == 'LSM':
            print("\n── LSM-Specific " + "─" * 54)

            self.opt_parameters['lsm_reinit_interval'] = int(
                self._prompt("  LSM reinit interval", 12, cast=int))

            self.opt_parameters['lsm_reinit_band'] = self._prompt(
                "  LSM reinit band", 0.04)

            self.opt_parameters['lsm_reinit_blend'] = self._prompt(
                "  LSM reinit blend", 0.70)

        elif method_upper == 'BESO':
            print("\n── BESO-Specific " + "─" * 53)

            self.opt_parameters['beso_addback_alpha'] = self._prompt(
                "  BESO add-back alpha", 0.95)

        # SIMP and MORI_TANAKA have no extra method-specific advanced params

    # ·· Sub-menu: Mesh Limits ···········································

    def _adv_mesh_limits(self):
        print("\n── Mesh Limits " + "─" * 55)

        self.opt_parameters['min_3d_elements'] = int(
            self._prompt("  Min tet elements", 40, cast=int))

        self.opt_parameters['max_3d_elements'] = int(
            self._prompt("  Max tet elements", 100000, cast=int))

    # ── Solver menu (unchanged) ─────────────────────────────────────────

    def display_solver_menu(self):
        """Define solver settings (direct solver only)."""
        print("\n" + "=" * 70)
        print("SOLVER SETTINGS")
        print("=" * 70)

        self.opt_parameters["solver_type"] = "direct"
        print("\nSolver Mode: Direct sparse solver (fixed)")
        print("[OK] Using direct solver")

        return True

    # ── Full menu runner ────────────────────────────────────────────────

    def run_full_menu(self, include_objective=True,
                      objective_function='compliance',
                      preset_volume_fraction=None,
                      include_mesh=True,
                      defer_volume_fraction=False,
                      settings_mode='standard',
                      optimization_method='SIMP'):
        """Run complete parameter definition menu"""
        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#" + "  TOPOLOGY OPTIMIZATION - PARAMETER CONFIGURATION".center(68) + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)

        if include_mesh:
            self.display_mesh_menu()
        else:
            self.mesh_settings['refinement'] = 'deferred_to_gui_remesh'
            print("[INFO] Mesh settings are deferred to GUI Remesh stage after BC definition")

        if include_objective:
            self.display_objective_menu()
        else:
            self.objective_function = str(objective_function).lower()
            print(f"[INFO] Objective locked from previous step: {self.objective_function}")

        self.display_solver_menu()
        self.display_optimization_menu(
            self.objective_function,
            preset_volume_fraction,
            defer_volume_fraction=defer_volume_fraction,
            settings_mode=settings_mode,
            optimization_method=optimization_method,
        )

        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Mesh Refinement: {self.mesh_settings.get('refinement', 'none')}")
        print(f"Objective: {self.objective_function}")
        if defer_volume_fraction:
            print("Volume Fraction: Deferred (set after stress analysis)")
        else:
            print(f"Volume Fraction: {self.opt_parameters.get('volume_fraction', 0.3):.2f}")
        print("Iterations: Auto (until convergence)")
        if self.opt_parameters.get('auto_filter_radius', True):
            print(f"Filter Radius: Auto ({self.opt_parameters.get('filter_radius_factor', 3.5)}x median element edge length)")
        else:
            print(f"Filter Radius: Manual = {self.opt_parameters.get('filter_radius', 1.5)}")
        print(f"Solver: {self.opt_parameters.get('solver_type', 'direct')}")
        print(f"Min member projection: {self.opt_parameters.get('use_min_member_projection', True)}")
        thr_src = self.opt_parameters.get('threshold_source', 'auto')
        if thr_src == 'manual':
            print(f"Threshold: {self.opt_parameters.get('threshold', 0.5)} (manual)")
        else:
            print("Threshold: Auto (derived from target volume/weight)")
        print(f"Protected support/load distance [mm]: {self.opt_parameters.get('non_design_distance_mm', 1.0):.2f}")

        if settings_mode == 'advanced':
            print(f"Penalization: {self.opt_parameters.get('penalization', 3.0)}")
            print(f"Max Iterations: {self.opt_parameters.get('max_iterations_auto', 300)}")
            print(f"Safety Factor: {self.opt_parameters.get('safety_factor', 1.5)}")
            print(f"Stress Constraint: {self.opt_parameters.get('use_stress_constraint', True)}")
            print(f"Linear Solver: {self.opt_parameters.get('linear_solver', 'auto')}")

        print("=" * 70)

        return self.opt_parameters, self.objective_function
