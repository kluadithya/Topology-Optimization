"""
TOPOLOGY OPTIMIZATION FRAMEWORK
3D-only interactive workflow with unified single-window GUI support
"""

import numpy as np
from pathlib import Path
import sys
import os
import tempfile
from core_components import MaterialModel, DEFAULT_CONFIG
from cad_importer import CADImporter, Mesh3DGenerator
from fea_solver_3d import FEASolver3D
from simp_3d import SIMP3DOptimizer
from beso_3d import BESO3DOptimizer
from lsm_3d import LSM3DOptimizer
from homogenization_3d import MoriTanaka3DOptimizer
from parameter_menu import ParameterMenu
from materials_library import MATERIAL_LIBRARY
from am_processes import AM_PROCESSES, list_am_process_keys, get_am_process, apply_am_process


class InteractiveTopologyOptimizer:
    """Interactive 3D topology optimization framework with unified GUI"""
    
    def __init__(self):
        self.mode = '3D'
        self.nodes = None
        self.elements = None
        self.material = None
        self.config = DEFAULT_CONFIG.copy()
        self._standard_defaults = DEFAULT_CONFIG.copy()  # immutable Standard baseline
        self.forces = None
        self.fixed_dofs = None
        self.results = None
        self.objective_function = 'stiffness'
        self.optimization_method = 'SIMP'
        self.settings_mode = 'standard'  # 'standard' or 'advanced'
        self.surf_faces = None
        self.mesh_gen = None  # Keep reference for remeshing
        self.face_ids = None
        self.edge_segments = None
        self._cad_surface_nodes = None
        self._cad_surface_elements = None
    
    def print_banner(self):
        """Display the main banner"""
        print('''
        
          TOPOLOGY OPTIMIZATION FRAMEWORK
          
        Enhanced Interactive Experience for 3D Optimization
        
        Features:
           - Unified single-window GUI for entire workflow
           - BC definition, mesh review, live optimization in one window
           - Static stress analysis before optimization (informed VF selection)
           - Post-optimization stress analysis (verify optimized design)
           - Re-optimization loop (adjust VF and re-run if needed)
           - Multiple optimization methods (SIMP, BESO, LSM, Mori-Tanaka)
           - Support for 3D CAD geometries
           - Custom objective function selection
           - Full parameter customization
        ''')

    def print_section(self, title):
        """Print a formatted section header"""
        print('\n' + '='*75)
        print(f'  {title}')
        print('='*75)

    def select_settings_mode(self):
        """Let user choose Standard or Advanced settings mode.

        Standard: current workflow with recommended defaults (quick setup).
        Advanced: all hardcoded parameters become editable via grouped sub-menus.
        The two modes maintain completely isolated parameter sets.
        """
        self.print_section('SETTINGS MODE')
        print('\n  1. Standard  - Quick setup with recommended defaults')
        print('  2. Advanced  - Full parameter control (expert users)')
        print()
        choice = self.safe_input('Choice (1-2, default 1): ', allow_none=True)
        if choice == '2':
            self.settings_mode = 'advanced'
            # Advanced starts from a fresh copy of defaults — never shares
            # state with Standard defaults.
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.settings_mode = 'standard'
            # Standard starts from the immutable baseline copy.
            self.config = self._standard_defaults.copy()
        print(f'[OK] Settings mode: {self.settings_mode.upper()}')

    def safe_input(self, prompt, input_type=str, allow_none=False):
        """Safely get user input with type checking"""
        while True:
            try:
                user_input = input(prompt).strip()
                if not user_input and allow_none:
                    return None
                if input_type == int:
                    return int(user_input)
                elif input_type == float:
                    return float(user_input)
                else:
                    return user_input
            except ValueError:
                print(f'Invalid input. Please enter a valid {input_type.__name__}.')
            except KeyboardInterrupt:
                print('\nExiting...')
                sys.exit(0)

    def select_mode(self):
        """Retained for compatibility: runtime is fixed to 3D mode."""
        self.mode = '3D'
        return True

    def setup_3d_from_cad(self):
        """Set up 3D domain from CAD file - loads surface mesh only (fast).
        Tet meshing is deferred to the GUI's Remesh (R key) stage."""
        self.print_section('3D CAD IMPORT')
        
        while True:
            cad_file = self.safe_input('\nCAD file path: ')
            if not Path(cad_file).exists():
                print('[ERROR] File not found')
                continue
            
            try:
                importer = CADImporter()
                if not importer.import_file(cad_file, auto_heal=True):
                    print('[ERROR] Import failed')
                    continue
                
                # Use surface mesh directly - NO tet meshing here (it can take minutes)
                self.nodes = importer.vertices
                self.elements = importer.elements
                
                # Store surface faces for GUI display
                self.surf_faces = importer.elements  # triangles ARE the surface
                self._cad_surface_nodes = importer.vertices.copy()
                self._cad_surface_elements = importer.elements.copy()
                self.face_ids = getattr(importer, 'face_ids', None)
                self.edge_segments = getattr(importer, 'edge_segments', None)
                
                # Also keep reference for later remeshing
                self._cad_importer = importer
                
                print(f'[OK] Surface mesh loaded: {self.elements.shape[0]} triangles, {self.nodes.shape[0]} nodes')
                print('[INFO] Use Remesh (R key) in the GUI to generate tet mesh for optimization.')
                return True
            except Exception as e:
                print(f'[ERROR] {e}')
                continue

    def select_optimization_method(self):
        """Select optimization method"""
        self.print_section('OPTIMIZATION METHOD')
        print('\n  1. SIMP')
        print('  2. BESO')
        print('  3. LSM (Hamilton-Jacobi)')
        print('  4. Mori-Tanaka (analytical micromechanics)')
        
        choice = self.safe_input('\nChoice (1-4): ')
        methods = {'1': 'SIMP', '2': 'BESO', '3': 'LSM', '4': 'MORI_TANAKA'}
        
        if choice in methods:
            self.optimization_method = methods[choice]
            print(f'[OK] {self.optimization_method}')
            return True
        return False

    def setup_material_properties(self):
        """Set up material properties using presets or manual custom values."""
        self.print_section('MATERIAL PROPERTIES')

        print('\nMaterial selection:')
        print('  1. Choose from library')
        print('  2. Enter custom values manually')

        mode_choice = self.safe_input('Choice (1-2): ')

        selected = None
        if mode_choice == '1':
            keys = list(MATERIAL_LIBRARY.keys())
            print('\nAvailable materials (edit materials_library.py to add more):')
            for i, key in enumerate(keys, start=1):
                m = MATERIAL_LIBRARY[key]
                print(f"  {i}. {m.get('display_name', key)}")

            idx = self.safe_input('Select material number: ', int)
            if 1 <= idx <= len(keys):
                selected = MATERIAL_LIBRARY[keys[idx - 1]]
                self.config['material_name'] = selected.get('display_name', keys[idx - 1])
                self.config['young_modulus'] = float(selected['young_modulus_mpa'])
                self.config['poisson_ratio'] = float(selected['poisson_ratio'])
                self.config['material_density'] = float(selected['density_kg_mm3'])
                self.config['yield_strength'] = float(selected['yield_strength_mpa'])
                for src_key, cfg_key in (
                    ('thermal_expansion_1_K', 'thermal_expansion'),
                    ('fatigue_limit_mpa', 'fatigue_limit'),
                    ('fracture_toughness_mpa_sqrt_mm', 'fracture_toughness'),
                ):
                    if src_key in selected and selected[src_key] is not None:
                        self.config[cfg_key] = float(selected[src_key])
                print(f"[OK] Selected material: {self.config['material_name']}")
            else:
                print('[WARNING] Invalid selection, switching to manual input.')

        if selected is None:
            self.config['material_name'] = 'Custom'
            self.config['young_modulus'] = self.safe_input("Young's Modulus / Elastic Stiffness [MPa]: ", float)
            self.config['poisson_ratio'] = self.safe_input("Poisson's Ratio [-]: ", float)
            self.config['material_density'] = self.safe_input('Mass Density [kg/mm^3]: ', float)
            self.config['yield_strength'] = self.safe_input('Yield Strength [MPa]: ', float)
            te = self.safe_input('Thermal expansion coeff [1/K] (optional): ', float, allow_none=True)
            fl = self.safe_input('Fatigue limit [MPa] (optional): ', float, allow_none=True)
            fk = self.safe_input('Fracture toughness [MPa*sqrt(mm)] (optional): ', float, allow_none=True)
            if te is not None:
                self.config['thermal_expansion'] = float(te)
            if fl is not None:
                self.config['fatigue_limit'] = float(fl)
            if fk is not None:
                self.config['fracture_toughness'] = float(fk)

        # Optional AM process modifiers
        use_am = self.safe_input('Apply additive manufacturing (AM) process modifiers? (y/n): ').lower()
        if use_am == 'y':
            keys = list_am_process_keys()
            print('\nAvailable AM processes:')
            for i, key in enumerate(keys, start=1):
                proc = get_am_process(key) or {}
                name = proc.get('display_name', key)
                notes = proc.get('notes', '')
                note_text = f' - {notes}' if notes else ''
                print(f'  {i}. {name}{note_text}')

            idx = self.safe_input('Select AM process number: ', int)
            if 1 <= idx <= len(keys):
                process_key = keys[idx - 1]
                axis_in = self.safe_input('Build axis [x/y/z] (default z): ', allow_none=True)
                build_axis = axis_in.lower().strip() if axis_in else 'z'
                if build_axis not in ('x', 'y', 'z'):
                    build_axis = 'z'

                if self.config.get('am_process_key') and self.config.get('base_young_modulus') is not None:
                    print('[INFO] AM modifiers already applied; using base material values to avoid double-scaling.')
                base_props = {
                    'young_modulus': self.config.get('base_young_modulus', self.config['young_modulus']),
                    'material_density': self.config.get('base_material_density', self.config['material_density']),
                    'yield_strength': self.config.get('base_yield_strength', self.config['yield_strength']),
                }
                am_result = apply_am_process(base_props, process_key, build_axis=build_axis)

                self.config['base_young_modulus'] = base_props['young_modulus']
                self.config['base_material_density'] = base_props['material_density']
                self.config['base_yield_strength'] = base_props['yield_strength']

                self.config['young_modulus'] = am_result['effective_young_modulus']
                self.config['material_density'] = am_result['effective_density']
                self.config['yield_strength'] = am_result['effective_yield_strength']
                self.config['am_process_key'] = am_result['process_key']
                self.config['am_process_name'] = am_result['process_name']
                self.config['am_build_axis'] = am_result['build_axis']
                self.config['am_anisotropy_factors'] = am_result['anisotropy_factors']
                self.config['am_yield_anisotropy_factors'] = am_result.get('yield_anisotropy_factors', None)
                self.config['am_stiffness_scale'] = am_result['stiffness_scale']
                self.config['am_yield_scale'] = am_result['yield_scale']
                self.config['am_density_scale'] = am_result['density_scale']

                e_eff = float(am_result.get('effective_young_modulus', 0.0))
                if not np.isfinite(e_eff) or e_eff <= 0.0:
                    print("[ERROR] Effective Young's modulus is non-physical after AM scaling.")
                elif e_eff < 1.0 or e_eff > 1e6:
                    print("[WARNING] Effective Young's modulus is outside a typical range. Check AM scaling.")

                print(f"[OK] AM process: {am_result['process_name']} (build axis {am_result['build_axis']})")
            else:
                print('[WARNING] Invalid AM process selection, skipping AM modifiers.')

        print('\nActive material properties for TO:')
        print(f"  Material: {self.config.get('material_name', 'Custom')}")
        print(f"  Young's Modulus [MPa]: {self.config['young_modulus']}")
        print(f"  Poisson's Ratio [-]: {self.config['poisson_ratio']}")
        print(f"  Density [kg/mm^3]: {self.config['material_density']}")
        print(f"  Yield Strength [MPa]: {self.config['yield_strength']}")
        if self.config.get('thermal_expansion', None) is not None:
            print(f"  Thermal Expansion [1/K]: {self.config['thermal_expansion']}")
        if self.config.get('fatigue_limit', None) is not None:
            print(f"  Fatigue Limit [MPa]: {self.config['fatigue_limit']}")
        if self.config.get('fracture_toughness', None) is not None:
            print(f"  Fracture Toughness [MPa*sqrt(mm)]: {self.config['fracture_toughness']}")

        if self.config.get('am_process_key', None):
            print(f"  AM Process: {self.config.get('am_process_name', self.config['am_process_key'])}")
            print(f"  AM Build Axis: {self.config.get('am_build_axis', 'z')}")
            print(f"  AM Scales: stiffness={self.config.get('am_stiffness_scale', 1.0):.2f}, yield={self.config.get('am_yield_scale', 1.0):.2f}, density={self.config.get('am_density_scale', 1.0):.2f}")
            anis = self.config.get('am_anisotropy_factors', None)
            if isinstance(anis, dict):
                print(f"  AM Anisotropy Factors: x={anis.get('x', 1.0):.2f}, y={anis.get('y', 1.0):.2f}, z={anis.get('z', 1.0):.2f}")
            yanis = self.config.get('am_yield_anisotropy_factors', None)
            if isinstance(yanis, dict):
                print(f"  AM Yield Anisotropy: x={yanis.get('x', 1.0):.2f}, y={yanis.get('y', 1.0):.2f}, z={yanis.get('z', 1.0):.2f}")

        try:
            self.material = MaterialModel(
                E0=self.config['young_modulus'],
                nu=self.config['poisson_ratio'],
                density=self.config.get('material_density', 7.8e-6),
                yield_strength=self.config.get('yield_strength', 250.0),
                thermal_expansion=self.config.get('thermal_expansion', None),
                fatigue_limit=self.config.get('fatigue_limit', None),
                fracture_toughness=self.config.get('fracture_toughness', None),
                anisotropy_factors=self.config.get('am_anisotropy_factors', None),
                build_axis=self.config.get('am_build_axis', 'z'),
                process_name=self.config.get('am_process_name', None),
                yield_anisotropy_factors=self.config.get('am_yield_anisotropy_factors', None),
            )
            print('[OK] Material properties set')
            return True
        except Exception as e:
            print(f'[ERROR] {e}')
            return False

    def select_objective_function(self):
        """Select objective type only. Volume fraction / mass reduction is
        deferred to the GUI stress-analysis stage so the user can make an
        informed choice after seeing the stress heatmap."""
        self.print_section('OBJECTIVE')
        print('\n1. Stiffness  2. Weight')
        choice = self.safe_input('Choice: ')

        if choice == '2':
            self.objective_function = 'weight'
            print('  [INFO] Weight reduction % will be set after stress analysis in the GUI.')
        else:
            self.objective_function = 'stiffness'
            print('  [INFO] Volume fraction will be set after stress analysis in the GUI.')

        # Mark volume fraction as deferred — the GUI will prompt after stress viz
        self.config['defer_volume_fraction'] = True
        self.config['volume_fraction'] = 0.3  # placeholder default
        self.config['objective_function'] = self.objective_function
        print(f'[OK] {self.objective_function}')
        return True

    def configure_optimization(self):
        """Configure optimization parameters and stress-constraint controls.

        In Standard mode the hardcoded defaults are applied silently.
        In Advanced mode the ParameterMenu presents grouped sub-menus so the
        user can override every parameter.  Advanced values are per-run and
        never contaminate Standard defaults.
        """
        try:
            param_menu = ParameterMenu()
            preset_vf = self.config.get('volume_fraction', 0.3)
            defer_vf = bool(self.config.get('defer_volume_fraction', False))
            opt_params, _ = param_menu.run_full_menu(
                include_objective=False,
                objective_function=self.objective_function,
                preset_volume_fraction=preset_vf,
                include_mesh=False,
                defer_volume_fraction=defer_vf,
                settings_mode=self.settings_mode,
                optimization_method=self.optimization_method,
            )

            # ── Values from the standard parameter menu prompts ─────────
            if self.objective_function != 'weight':
                self.config['volume_fraction'] = opt_params.get('volume_fraction', self.config.get('volume_fraction', 0.3))
            self.config['auto_filter_radius'] = bool(opt_params.get('auto_filter_radius', self.config.get('auto_filter_radius', True)))
            self.config['filter_radius_factor'] = float(opt_params.get('filter_radius_factor', self.config.get('filter_radius_factor', 3.5)))
            self.config['support_non_design_distance_mm'] = float(opt_params.get('support_non_design_distance_mm', self.config.get('support_non_design_distance_mm', 1.0)))
            self.config['force_non_design_distance_mm'] = float(opt_params.get('force_non_design_distance_mm', self.config.get('force_non_design_distance_mm', 1.0)))
            self.config['use_min_member_projection'] = bool(opt_params.get('use_min_member_projection', self.config.get('use_min_member_projection', True)))
            self.config['min_member_size_factor'] = float(opt_params.get('min_member_size_factor', self.config.get('min_member_size_factor', 1.5)))
            self.config['min_member_projection_beta'] = float(opt_params.get('min_member_projection_beta', self.config.get('min_member_projection_beta', 8.0)))
            self.config['min_member_size_mm'] = opt_params.get('min_member_size_mm', self.config.get('min_member_size_mm', None))
            self.config['enable_overhang_filter'] = bool(opt_params.get('enable_overhang_filter', self.config.get('enable_overhang_filter', False)))
            self.config['overhang_angle_deg'] = float(opt_params.get('overhang_angle_deg', self.config.get('overhang_angle_deg', 45.0)))
            self.config['build_axis'] = str(opt_params.get('build_axis', self.config.get('build_axis', 'z'))).lower()
            self.config['check_enclosed_voids'] = bool(opt_params.get('check_enclosed_voids', self.config.get('check_enclosed_voids', True)))

            # ── Penalization / stress / iterations ──────────────────────
            if self.settings_mode == 'advanced':
                # Use values that the user entered in the Advanced sub-menus
                pen = float(opt_params.get('penalization', 3.0))
                self.config['penalization'] = pen
                self.config['penalty'] = pen
                self.config['penalization_factor'] = pen

                # Threshold: manual override or auto
                if opt_params.get('threshold_source') == 'manual':
                    self.config['threshold'] = float(opt_params['threshold'])
                else:
                    self.config['threshold'] = self._compute_auto_density_threshold(None)

                # Solver & iteration
                self.config['max_iterations_auto'] = int(opt_params.get('max_iterations_auto', 300))
                self.config['linear_solver'] = str(opt_params.get('linear_solver', 'auto'))
                self.config['iterative_solver_tol'] = float(opt_params.get('iterative_solver_tol', 1e-8))
                self.config['iterative_solver_maxiter'] = int(opt_params.get('iterative_solver_maxiter', 2000))

                # Filter / projection (may have been overridden)
                if not self.config['auto_filter_radius']:
                    self.config['filter_radius'] = float(opt_params.get('filter_radius', 1.5))
                if 'min_member_projection_beta_start' in opt_params:
                    self.config['min_member_projection_beta_start'] = float(opt_params['min_member_projection_beta_start'])
                if 'min_member_projection_beta_end' in opt_params:
                    self.config['min_member_projection_beta_end'] = float(opt_params['min_member_projection_beta_end'])
                    self.config['min_member_projection_beta'] = float(opt_params['min_member_projection_beta_end'])

                # Stress constraint
                self.config['use_stress_constraint'] = bool(opt_params.get('use_stress_constraint', True))
                self.config['safety_factor'] = float(opt_params.get('safety_factor', 1.5))
                self.config['stress_penalty_weight'] = float(opt_params.get('stress_penalty_weight', 2.0))
                self.config['use_stress_augmented_lagrangian'] = bool(opt_params.get('use_stress_augmented_lagrangian', True))
                if self.config['use_stress_augmented_lagrangian']:
                    self.config['stress_al_mu0'] = float(opt_params.get('stress_al_mu0', 8.0))
                    self.config['stress_al_mu_growth'] = float(opt_params.get('stress_al_mu_growth', 1.10))
                    self.config['stress_al_mu_max'] = float(opt_params.get('stress_al_mu_max', 1e4))
                self.config['stress_eval_interval'] = int(opt_params.get('stress_eval_interval', 5))

                # Post-processing
                self.config['final_density_smoothing_passes'] = int(opt_params.get('final_density_smoothing_passes', 2))
                self.config['final_density_smoothing_blend'] = float(opt_params.get('final_density_smoothing_blend', 0.60))
                self.config['smooth_result_surface'] = bool(opt_params.get('smooth_result_surface', True))
                if self.config['smooth_result_surface']:
                    self.config['result_surface_smooth_iterations'] = int(opt_params.get('result_surface_smooth_iterations', 25))
                    self.config['result_surface_relaxation'] = float(opt_params.get('result_surface_relaxation', 0.08))

                # Method-specific
                method_upper = str(self.optimization_method).upper()
                if method_upper == 'LSM':
                    self.config['lsm_reinit_interval'] = int(opt_params.get('lsm_reinit_interval', 12))
                    self.config['lsm_reinit_band'] = float(opt_params.get('lsm_reinit_band', 0.04))
                    self.config['lsm_reinit_blend'] = float(opt_params.get('lsm_reinit_blend', 0.70))
                elif method_upper == 'BESO':
                    self.config['beso_addback_alpha'] = float(opt_params.get('beso_addback_alpha', 0.95))

                # Mesh limits
                self.config['min_3d_elements'] = int(opt_params.get('min_3d_elements', 40))
                self.config['max_3d_elements'] = int(opt_params.get('max_3d_elements', 100000))

            else:
                # ── Standard mode: silent hardcoded defaults ────────────
                self.config['penalization'] = 3.0
                self.config['penalty'] = 3.0
                self.config['penalization_factor'] = 3.0
                self.config['threshold'] = self._compute_auto_density_threshold(None)
                self.config['max_iterations_auto'] = int(self.config.get('max_iterations_auto', 300))

            # ── Tet element type (both modes) ──────────────────────────
            tet_default = str(self.config.get('tet_element_type', 'tet10')).lower()
            print('\nTetrahedral element type for remeshing:')
            print('  1. Tet4 (faster, lower accuracy)')
            print('  2. Tet10 (slower, higher accuracy)')
            default_choice = '2' if tet_default == 'tet10' else '1'
            tet_in = self.safe_input(f'Choice (1-2, default {default_choice}): ', allow_none=True)
            if tet_in is None or str(tet_in).strip() == '':
                tet_choice = 'tet10' if tet_default == 'tet10' else 'tet4'
            else:
                tet_choice = 'tet10' if str(tet_in).strip() == '2' else 'tet4'
            self.config['tet_element_type'] = tet_choice
            self.config['gmsh_element_order'] = 2 if tet_choice == 'tet10' else 1

            self.config['objective_function'] = self.objective_function
            self.config['iterations'] = int(self.config['max_iterations_auto'])

            # ── Stress constraint summary ──────────────────────────────
            if self.mode == '3D':
                if self.settings_mode == 'standard':
                    # Standard: silently apply fixed stress constraint
                    self.config['use_stress_constraint'] = True
                    self.config['safety_factor'] = float(self.config.get('safety_factor', 1.5))
                    self.config['stress_penalty_weight'] = float(self.config.get('stress_penalty_weight', 2.0))

                if self.config.get('use_stress_constraint', True):
                    ys = float(self.config.get('yield_strength', 250.0))
                    allowable = ys / max(float(self.config['safety_factor']), 1e-9)
                    status = 'ACTIVE' if self.config['use_stress_constraint'] else 'DISABLED'
                    print(f'\nStress constraint: {status}')
                    print('  [STRESS] Yield={:.3e} Pa, SF={:.3f}, Allowable={:.3e} Pa'.format(ys, self.config['safety_factor'], allowable))
                else:
                    print('\nStress constraint: DISABLED')

            print('[OK] Configuration complete')
            return True
        except Exception as e:
            print(f'[WARNING] Parameter menu failed: {e}')
            if self.safe_input('Customize? (y/n): ').lower() == 'y':
                if self.objective_function != 'weight':
                    self.config['volume_fraction'] = self.safe_input('Volume fraction: ', float)

            self.config['objective_function'] = self.objective_function
            self.config['max_iterations_auto'] = int(self.config.get('max_iterations_auto', 300))
            self.config['iterations'] = int(self.config['max_iterations_auto'])
            self.config['use_stress_constraint'] = True
            print('[OK] Config set')
            return True

    def _make_optimizer_factory(self):
        """Return a factory function that creates the selected optimizer."""
        method = self.optimization_method
        def factory(nodes, elements, material, config):
            cfg3d = dict(config)
            cfg3d.setdefault('penalty', cfg3d.get('penalization', cfg3d.get('penalization_factor', 3.0)))
            cfg3d.setdefault('penalization', cfg3d.get('penalty', 3.0))
            cfg3d.setdefault('max_iterations_auto', int(cfg3d.get('max_iterations_auto', cfg3d.get('iterations', 300))))

            obj = str(self.objective_function).lower()
            if obj == 'compliance':
                obj = 'stiffness'
            # Force objective from current session selection (do not allow stale config keys to override).
            cfg3d['objective'] = obj

            if obj == 'weight':
                wr = float(self.config.get('weight_reduction_percent', self.config.get('target_weight_reduction_percent', 0.0)))
                wr = float(np.clip(wr, 1.0, 90.0))
                vf = float(np.clip(1.0 - wr / 100.0, 0.05, 0.99))
                cfg3d['weight_reduction_percent'] = wr
                cfg3d['target_weight_reduction_percent'] = wr
                cfg3d['volume_fraction'] = vf
                cfg3d['volfrac'] = vf
                print(f"  [CFG] Objective=weight, target reduction={wr:.1f}% -> vf={vf:.4f}")
            else:
                vf = float(np.clip(cfg3d.get('volume_fraction', cfg3d.get('volfrac', 0.3)), 0.05, 0.99))
                cfg3d['volume_fraction'] = vf
                cfg3d['volfrac'] = vf
                print(f"  [CFG] Objective={obj}, vf={vf:.4f}")
            fair_cmp = bool(cfg3d.get('fair_method_comparison', True))
            if fair_cmp and method == 'BESO':
                cfg3d.setdefault('use_min_member_projection_beso', False)
            if fair_cmp and method == 'LSM':
                cfg3d.setdefault('use_min_member_projection_lsm', True)

            if method == 'SIMP':
                return SIMP3DOptimizer(nodes, elements, material, cfg3d)
            elif method == 'BESO':
                return BESO3DOptimizer(nodes, elements, material, cfg3d)
            elif method == 'LSM':
                return LSM3DOptimizer(nodes, elements, material, cfg3d)
            else:
                return MoriTanaka3DOptimizer(nodes, elements, material, cfg3d)

        return factory

    def _remesh_3d(self, nodes, elements, progress_cb=None):
        """Callback for 3D mesh refinement.
        User enters only target tetra count; meshing parameters are auto-tuned."""
        print('\n  [REMESH 3D] Generating tetrahedral mesh from CAD surface...')

        def emit(percent, message):
            if progress_cb is not None:
                try:
                    progress_cb(float(percent), message)
                except Exception:
                    pass

        emit(2.0, 'Waiting for target element count...')

        try:
            min_tets = int(self.config.get('min_3d_elements', 1000))
            max_tets = int(self.config.get('max_3d_elements', 100000))
            default_tets = int(np.clip(10000, min_tets, max_tets))
            raw_target = input(f'  Target tetra elements ({min_tets}-{max_tets}): ').strip()
            if not raw_target:
                target_tets = default_tets
                print(f'  [REMESH 3D] Using default target: {default_tets}')
            else:
                target_tets = int(raw_target)
            target_tets = max(min_tets, min(max_tets, target_tets))
            tol = 0.10
            low_target = int(np.floor(target_tets * (1.0 - tol)))
            high_target = int(np.ceil(target_tets * (1.0 + tol)))
            max_attempts = 8
            quality_order = 2.0
            use_gmsh = True
            gmsh_order = int(self.config.get('gmsh_element_order', 2))
            tet_label = 'Tet10' if gmsh_order >= 2 else 'Tet4'
            # §3.6: Tet10 mesh size guidance — Tet10 needs 4-8× fewer elements
            # than Tet4 for equivalent accuracy due to quadratic shape functions.
            if gmsh_order >= 2 and target_tets > 15000:
                suggested = int(target_tets / 5)
                print(f'  [TIP] Tet10 elements capture bending 4-8× better than Tet4.')
                print(f'  [TIP] Consider using ~{suggested} Tet10 elements instead of {target_tets} for similar accuracy but 3-4× faster iterations.')
            emit(10.0, f'Target set to {target_tets} tetra elements')
        except (ValueError, KeyboardInterrupt):
            min_tets = int(self.config.get('min_3d_elements', 1000))
            max_tets = int(self.config.get('max_3d_elements', 100000))
            default_tets = int(np.clip(10000, min_tets, max_tets))
            print(f'  [REMESH 3D] Invalid input, using default target: {default_tets}')
            target_tets = default_tets
            tol = 0.10
            low_target = int(np.floor(target_tets * (1.0 - tol)))
            high_target = int(np.ceil(target_tets * (1.0 + tol)))
            max_attempts = 8
            quality_order = 2.0
            use_gmsh = True
            gmsh_order = int(self.config.get('gmsh_element_order', 2))
            tet_label = 'Tet10' if gmsh_order >= 2 else 'Tet4'

        try:
            from cad_importer import Mesh3DGenerator

            source_nodes = self._cad_surface_nodes if self._cad_surface_nodes is not None else nodes
            source_elements = self._cad_surface_elements if self._cad_surface_elements is not None else elements
            emit(18.0, 'Preparing source surface...')

            # Keep geometry unchanged by default: no smoothing subdivision.
            # If input surface is extremely coarse, one linear subdivision helps mesher robustness.
            surface_refine = 1 if (source_elements is not None and len(source_elements) < 80) else 0
            if surface_refine > 0:
                emit(25.0, 'Refining very coarse source surface...')
                try:
                    import pyvista as pv
                    tri = np.asarray(source_elements, dtype=np.int32)
                    if tri.ndim == 2 and tri.shape[1] == 3 and len(tri) > 0:
                        faces_pv = np.hstack([
                            np.full((tri.shape[0], 1), 3, dtype=np.int64),
                            tri.astype(np.int64)
                        ]).ravel()
                        surf = pv.PolyData(np.asarray(source_nodes, dtype=np.float64), faces_pv).triangulate()
                        surf = surf.subdivide(1, subfilter='linear')
                        source_nodes = np.asarray(surf.points, dtype=np.float64)
                        source_elements = surf.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
                        print(f'  [REMESH 3D] Surface refined -> {source_nodes.shape[0]} nodes, {source_elements.shape[0]} triangles')
                except Exception as e:
                    print(f'  [REMESH 3D] Surface refinement skipped: {e}')

            bmin = np.min(source_nodes, axis=0)
            bmax = np.max(source_nodes, axis=0)
            bbox_vol = float(np.prod(np.maximum(bmax - bmin, 1e-9)))
            current_max_volume = max(bbox_vol / float(max(target_tets, 1)), 1e-9)

            best_nodes, best_elements = None, None
            best_mesh_gen = None
            best_diff = None

            for attempt in range(1, max_attempts + 1):
                phase = 30.0 + (attempt - 1) * (60.0 / max_attempts)
                emit(phase, f'Meshing attempt {attempt}/{max_attempts}...')

                mesh_gen = Mesh3DGenerator(source_nodes, source_elements)
                ok = False
                if use_gmsh:
                    ok = mesh_gen._mesh_with_gmsh(current_max_volume, element_order=gmsh_order)
                    if not ok:
                        print(f'  [REMESH 3D] gmsh {tet_label} meshing failed on this attempt.')
                else:
                    ok = False

                if not ok:
                    continue

                new_nodes, new_elements = mesh_gen.get_mesh()
                n_tet = int(new_elements.shape[0]) if new_elements is not None else 0
                print(f'  [REMESH 3D] Attempt {attempt}: {new_nodes.shape[0]} nodes, {n_tet} tetra elements')
                emit(min(95.0, phase + 5.0), f'Attempt {attempt}: {n_tet} tetra elements')

                diff = abs(n_tet - target_tets)
                if best_diff is None or diff < best_diff:
                    best_nodes, best_elements = new_nodes, new_elements
                    best_mesh_gen = mesh_gen
                    best_diff = diff

                if low_target <= n_tet <= high_target:
                    print(f'  [REMESH 3D] Target met within +/-10%: {n_tet} (target {target_tets})')
                    break

                # Update max volume using approximate inverse relationship: count ~ 1/volume
                ratio = float(n_tet) / float(max(target_tets, 1))
                ratio = max(0.2, min(5.0, ratio))
                current_max_volume = max(current_max_volume * ratio, 1e-12)

            if best_nodes is not None and best_elements is not None:
                self.nodes = best_nodes
                self.elements = best_elements
                self.face_ids = None
                self.edge_segments = None
                if best_mesh_gen is not None:
                    self.surf_faces = best_mesh_gen.surface_faces

                final_n = int(best_elements.shape[0])
                if not (low_target <= final_n <= high_target):
                    print(f'  [REMESH 3D] Closest mesh achieved: {final_n} tetra (target {target_tets}, tolerance +/-10%).')
                print(f'  [REMESH 3D] Success! {best_nodes.shape[0]} nodes, {final_n} tet elements')
                emit(100.0, f'Remesh complete: {final_n} tetra elements')
                return best_nodes, best_elements

            emit(100.0, 'Remesh failed')
            print('  [REMESH 3D] Tetrahedral meshing failed.')
            return None, None

        except Exception as e:
            print(f'  [REMESH 3D] Error: {e}')
            import traceback; traceback.print_exc()
            return None, None

    def run_unified_gui_workflow(self):
        """Run the unified single-window GUI workflow."""
        from unified_gui import UnifiedWorkflowGUI

        # Show a warning if the mesh is surface-only, but still open the GUI
        if self.mode == '3D':
            elem_nodes = self.elements.shape[1] if self.elements.ndim == 2 else 0
            if elem_nodes < 4:
                print('\n[INFO] Your mesh has {}-node elements (surface triangles).'.format(elem_nodes))
                print('  The GUI will open in Mesh Review stage first.')
                print('  Press R to remesh into Tet4/Tet10 elements, then N to define BCs.')
                print('  To enable remeshing, install gmsh: pip install gmsh\n')

        gui = UnifiedWorkflowGUI(
            nodes=self.nodes,
            elements=self.elements,
            mode=self.mode,
            surf_faces=self.surf_faces,
            material=self.material,
            config=self.config,
            optimizer_factory=self._make_optimizer_factory(),
            face_ids=self.face_ids,
            edge_segments=self.edge_segments
        )

        # Register 3D remesh callback
        gui.set_remesh_callback(self._remesh_3d)

        # Run the unified GUI (blocks until user quits)
        fixed_dofs, forces, rho, compliance = gui.run()

        if rho is not None:
            self.fixed_dofs = fixed_dofs
            self.forces = forces
            self.results = {
                'method': self.optimization_method,
                'objective': self.objective_function,
                'mode': self.mode,
                'rho_optimized': rho,
                'compliance': compliance,
                'config': self.config.copy(),
                'reopt_count': getattr(gui, '_reopt_count', 0),
            }
            return True
        else:
            print('[INFO] No optimization results (user may have quit early)')
            # Still store BC data if available
            if fixed_dofs is not None and len(fixed_dofs) > 0:
                self.fixed_dofs = fixed_dofs
                self.forces = forces
            return False

    def show_results(self):
        """Display results"""
        if not self.results:
            return
        
        self.print_section('RESULTS')
        print(f'Mode: {self.results["mode"]}')
        print(f'Method: {self.results["method"]}')
        print(f'Objective: {self.results["objective"]}')
        comp = self.results['compliance']
        if isinstance(comp, (list, tuple)):
            try:
                comp = float(comp[-1]) if len(comp) > 0 else 0.0
            except:
                comp = float(comp)
        else:
            comp = float(comp)
        print(f'Compliance: {comp:.6e}')
        
        rho = self.results['rho_optimized']
        print(f'Volume: {rho.mean():.4f}')
        print(f'Min: {rho.min():.6e}, Max: {rho.max():.6e}')
        reopt = self.results.get('reopt_count', 0)
        if reopt > 0:
            print(f'Re-optimizations: {reopt}')
    
    def _build_export_surface(self, rho, threshold=0.5):
        """Build an exportable triangle surface from optimized result."""
        nodes = np.asarray(self.nodes, dtype=np.float64)
        if nodes.ndim != 2:
            raise ValueError('Invalid node array for export')
        if nodes.shape[1] == 2:
            nodes = np.column_stack([nodes, np.zeros(nodes.shape[0], dtype=np.float64)])

        elems = np.asarray(self.elements, dtype=np.int64)
        if elems.ndim != 2:
            raise ValueError('Invalid element array for export')

        n_elem = elems.shape[0]
        if rho is None or len(rho) != n_elem:
            active = np.ones(n_elem, dtype=bool)
        else:
            active = np.asarray(rho, dtype=np.float64) >= float(threshold)

        n_per = elems.shape[1]
        if n_per in (4, 10):
            tets = elems[active]
            if tets.size == 0:
                return nodes, np.zeros((0, 3), dtype=np.int64)

            corners = tets[:, :4] if n_per == 10 else tets
            faces = np.vstack([
                corners[:, [0, 1, 2]],
                corners[:, [0, 1, 3]],
                corners[:, [0, 2, 3]],
                corners[:, [1, 2, 3]],
            ])
            faces_sorted = np.sort(faces, axis=1)
            uniq, counts = np.unique(faces_sorted, axis=0, return_counts=True)
            boundary = uniq[counts == 1]
            return nodes, boundary.astype(np.int64)

        if n_per == 3:
            tris = elems[active]
            return nodes, tris.astype(np.int64)

        raise ValueError(f'Unsupported element topology for export: {n_per} nodes per element')

    @staticmethod
    def _smooth_export_surface(nodes, triangles, iterations=2, alpha=0.30):
        """Light Laplacian smoothing for export-only faceting cleanup."""
        tri = np.asarray(triangles, dtype=np.int64)
        if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0:
            return np.asarray(nodes, dtype=np.float64)

        pts = np.asarray(nodes, dtype=np.float64).copy()
        n_pts = pts.shape[0]
        if n_pts == 0:
            return pts

        alpha = float(np.clip(alpha, 0.0, 1.0))
        iters = int(max(0, iterations))
        if iters == 0 or alpha <= 0.0:
            return pts

        neighbors = [set() for _ in range(n_pts)]
        for a, b, c in tri:
            ia, ib, ic = int(a), int(b), int(c)
            neighbors[ia].add(ib); neighbors[ia].add(ic)
            neighbors[ib].add(ia); neighbors[ib].add(ic)
            neighbors[ic].add(ia); neighbors[ic].add(ib)

        edge_use = {}
        for a, b, c in tri:
            e1 = tuple(sorted((int(a), int(b))))
            e2 = tuple(sorted((int(b), int(c))))
            e3 = tuple(sorted((int(c), int(a))))
            edge_use[e1] = edge_use.get(e1, 0) + 1
            edge_use[e2] = edge_use.get(e2, 0) + 1
            edge_use[e3] = edge_use.get(e3, 0) + 1
        boundary = np.zeros(n_pts, dtype=bool)
        for (i, j), c in edge_use.items():
            if c == 1:
                boundary[i] = True
                boundary[j] = True

        for _ in range(iters):
            new_pts = pts.copy()
            for i in range(n_pts):
                if boundary[i] or not neighbors[i]:
                    continue
                nbr = np.fromiter(neighbors[i], dtype=np.int64)
                centroid = np.mean(pts[nbr], axis=0)
                new_pts[i] = (1.0 - alpha) * pts[i] + alpha * centroid
            pts = new_pts

        return pts

    @staticmethod
    def _decimate_export_surface(nodes, triangles, target_faces=5000):
        """Reduce face count for faster STEP import in CAD tools."""
        tri = np.asarray(triangles, dtype=np.int64)
        pts = np.asarray(nodes, dtype=np.float64)
        if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] <= int(target_faces):
            return pts, tri

        try:
            import pyvista as pv
            faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
            mesh = pv.PolyData(pts, faces)
            red = float(np.clip(1.0 - (float(target_faces) / float(max(tri.shape[0], 1))), 0.0, 0.99))
            dec = mesh.decimate_pro(red)
            if dec is None or dec.n_cells < 4:
                return pts, tri
            faces_dec = np.asarray(dec.faces).reshape(-1, 4)
            tri_dec = np.asarray(faces_dec[:, 1:4], dtype=np.int64)
            pts_dec = np.asarray(dec.points, dtype=np.float64)
            return pts_dec, tri_dec
        except Exception:
            return pts, tri

    @staticmethod
    def _write_ascii_stl(stl_path, nodes, triangles):
        with open(stl_path, 'w', encoding='utf-8') as f:
            f.write('solid topology_optimized\n')
            for tri in triangles:
                p1 = nodes[int(tri[0])]
                p2 = nodes[int(tri[1])]
                p3 = nodes[int(tri[2])]
                n = np.cross(p2 - p1, p3 - p1)
                nn = float(np.linalg.norm(n))
                if nn > 1e-20:
                    n = n / nn
                else:
                    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                f.write(f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
                f.write('    outer loop\n')
                f.write(f'      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n')
                f.write(f'      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n')
                f.write(f'      vertex {p3[0]:.6e} {p3[1]:.6e} {p3[2]:.6e}\n')
                f.write('    endloop\n')
                f.write('  endfacet\n')
            f.write('endsolid topology_optimized\n')

    def _export_stl(self, stl_path, nodes, triangles):
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=nodes, faces=triangles, process=False)
            mesh.export(stl_path)
            return True, 'Exported using trimesh'
        except Exception:
            try:
                self._write_ascii_stl(stl_path, nodes, triangles)
                return True, 'Exported using ASCII STL fallback'
            except Exception as e:
                return False, f'STL export failed: {e}'

    def _export_step(self, step_path, nodes, triangles):
        # Primary path (Gmsh) bypassed: Gmsh surface classification often hangs in an infinite
        # tolerance loop when trying to fit analytical surfaces over complex/organic topology 
        # optimization meshes. We skip straight to the safer pythonocc faceted fallback.
        gmsh_err = "Gmsh analytical reconstruction bypassed to prevent infinite tolerance loops."

        # Fallback path: write faceted STEP directly with pythonocc.
        try:
            from OCC.Core.BRep import BRep_Builder  # type: ignore
            from OCC.Core.TopoDS import TopoDS_Compound  # type: ignore
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace  # type: ignore
            from OCC.Core.gp import gp_Pnt  # type: ignore
            from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs  # type: ignore
            from OCC.Core.IFSelect import IFSelect_RetDone  # type: ignore

            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)

            for tri in triangles:
                p1 = nodes[int(tri[0])]
                p2 = nodes[int(tri[1])]
                p3 = nodes[int(tri[2])]

                poly = BRepBuilderAPI_MakePolygon()
                poly.Add(gp_Pnt(float(p1[0]), float(p1[1]), float(p1[2])))
                poly.Add(gp_Pnt(float(p2[0]), float(p2[1]), float(p2[2])))
                poly.Add(gp_Pnt(float(p3[0]), float(p3[1]), float(p3[2])))
                poly.Close()
                wire = poly.Wire()
                face = BRepBuilderAPI_MakeFace(wire).Face()
                builder.Add(compound, face)

            writer = STEPControl_Writer()
            writer.Transfer(compound, STEPControl_AsIs)
            status = writer.Write(step_path)
            if status == IFSelect_RetDone:
                return True, 'Exported using pythonocc faceted STEP fallback'
            return False, f'STEP write returned status {status}'
        except Exception as e_occ:
            return False, f'STEP export failed (gmsh primary: {gmsh_err}; pythonocc fallback: {e_occ})'

    def export_optimized_design(self):
        """Export optimized design with selectable output format."""
        if not self.results or 'rho_optimized' not in self.results:
            return

        choice = self.safe_input('\nExport optimized design? (y/n): ').lower()
        if choice != 'y':
            return

        print('\nExport format:')
        print('  1. STL only')
        print('  2. STEP only')
        print('  3. Both STL and STEP')
        fmt_choice = self.safe_input('Choice (1-3): ')

        do_stl = fmt_choice in ('1', '3')
        do_step = fmt_choice in ('2', '3')
        if not (do_stl or do_step):
            print('[EXPORT] Invalid choice; export cancelled.')
            return

        base = self.safe_input('Output file base path (default: optimized_design): ', allow_none=True)
        if not base:
            base = 'optimized_design'
        base, _ = os.path.splitext(base)
        if not base:
            base = 'optimized_design'

        stl_path = base + '.stl'
        step_path = base + '.step'

        rho = np.asarray(self.results['rho_optimized'], dtype=np.float64)
        threshold = float(self.config.get('threshold', 0.5))
        smooth_iters = int(self.config.get('export_smoothing_iterations', 0))
        smooth_alpha = float(self.config.get('export_smoothing_alpha', 0.30))

        smooth_in = self.safe_input('Apply light export smoothing? (y/n, default n): ', allow_none=True)
        if smooth_in is not None and smooth_in.strip().lower() == 'y':
            it_in = self.safe_input('Smoothing iterations (default 2): ', allow_none=True)
            al_in = self.safe_input('Smoothing alpha 0-1 (default 0.30): ', allow_none=True)
            try:
                smooth_iters = int(it_in) if it_in else 2
            except Exception:
                smooth_iters = 2
            try:
                smooth_alpha = float(al_in) if al_in else 0.30
            except Exception:
                smooth_alpha = 0.30

        try:
            nodes, triangles = self._build_export_surface(rho, threshold=threshold)
        except Exception as e:
            print(f'[EXPORT] Failed to build export surface: {e}')
            return

        if triangles.shape[0] == 0:
            print('[EXPORT] No solid surface found to export (try lower threshold).')
            return

        if smooth_iters > 0 and smooth_alpha > 0.0:
            nodes = self._smooth_export_surface(nodes, triangles, iterations=smooth_iters, alpha=smooth_alpha)
            print(f'[EXPORT] Applied smoothing: iterations={smooth_iters}, alpha={smooth_alpha:.2f}')

        if do_stl:
            stl_ok, stl_msg = self._export_stl(stl_path, nodes, triangles)
            if stl_ok:
                print(f'[EXPORT] STL saved: {stl_path}')
            else:
                print(f'[EXPORT] STL failed: {stl_msg}')

        if do_step:
            # Faceted STEP can be slow in CAD import; allow explicit face-budget profiles.
            default_faces = int(self.config.get('step_export_target_faces', 3000))
            print('\nSTEP import speed profile:')
            print('  1. Fast (1500 faces, smallest/fastest loading)')
            print('  2. Balanced (3000 faces, recommended)')
            print('  3. Quality (5000 faces)')
            print('  4. High quality (8000 faces, slower)')
            p_in = self.safe_input(f'Profile (1-4, Enter=Balanced, current={default_faces}): ', allow_none=True)
            profile_map = {'1': 1500, '2': 3000, '3': 5000, '4': 8000}
            if p_in is None or str(p_in).strip() == '':
                dec_target = 3000
            else:
                dec_target = profile_map.get(str(p_in).strip(), default_faces)

            dec_target_in = self.safe_input(f'Custom STEP target faces (Enter to keep {dec_target}): ', allow_none=True)
            if dec_target_in is not None and str(dec_target_in).strip() != '':
                try:
                    dec_target = max(500, int(float(dec_target_in)))
                except Exception:
                    pass

            nodes_step, tri_step = self._decimate_export_surface(nodes, triangles, target_faces=dec_target)
            if tri_step.shape[0] < triangles.shape[0]:
                print(f'[EXPORT] STEP decimation: {triangles.shape[0]} -> {tri_step.shape[0]} faces (target={dec_target})')
            else:
                print(f'[EXPORT] STEP decimation: unchanged at {tri_step.shape[0]} faces (target={dec_target})')
            step_ok, step_msg = self._export_step(step_path, nodes_step, tri_step)
            if step_ok:
                print(f'[EXPORT] STEP saved: {step_path}')
                print('[EXPORT] Tip: choose Fast/Balanced profile for quickest SolidWorks opening.')
            else:
                print(f'[EXPORT] STEP failed: {step_msg}')
                print('[EXPORT] Install pythonocc-core for robust STEP export support.')

    def _target_volume_fraction(self):
        """Return target material fraction derived from objective settings."""
        obj = str(self.objective_function).lower()
        if obj == 'weight':
            wr = self.config.get('weight_reduction_percent', self.config.get('target_weight_reduction_percent', 0.0))
            try:
                wr = float(wr)
            except Exception:
                wr = 0.0
            return float(np.clip(1.0 - wr / 100.0, 0.05, 0.99))
        vf = self.config.get('volume_fraction', self.config.get('volfrac', 0.3))
        try:
            vf = float(vf)
        except Exception:
            vf = 0.3
        return float(np.clip(vf, 0.05, 0.99))

    def _compute_auto_density_threshold(self, rho=None):
        """Compute cutoff from target volume fraction (and rho field when available)."""
        vf = self._target_volume_fraction()
        if rho is None:
            # Pre-run estimate: denser targets use lower cutoff, lighter targets use higher cutoff.
            return float(np.clip(1.0 - vf, 0.30, 0.80))

        arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(np.clip(1.0 - vf, 0.30, 0.80))

        n = int(arr.size)
        n_solid = int(np.clip(round(vf * n), 1, n))
        k = max(0, n - n_solid)
        thr = float(np.partition(arr, k)[k])
        return float(np.clip(thr, 0.30, 0.80))

    def _sync_auto_density_threshold(self):
        """Set post-optimization display/export threshold automatically from objective target."""
        rho = None
        if self.results is not None and 'rho_optimized' in self.results:
            rho = self.results['rho_optimized']
        thr = self._compute_auto_density_threshold(rho)
        self.config['threshold'] = thr
        self.config['threshold_source'] = 'auto'
        print('\nPost-optimization result cutoff:')
        print('  [AUTO] Density threshold is derived from target volume/weight objective.')
        print(f'  [AUTO] Threshold set to {thr:.3f}')

    def _run_verification_fea_if_available(self):
        """Run verification FEA on binary-thresholded design if optimizer supports it."""
        if not self.results or 'rho_optimized' not in self.results:
            return
        if self.forces is None or self.fixed_dofs is None:
            return

        # Only SIMP optimizer has run_verification_fea
        try:
            from simp_3d import SIMP3DOptimizer
        except ImportError:
            return

        if self.optimization_method != 'SIMP':
            print('[INFO] Verification FEA is currently available for SIMP method only.')
            return

        try:
            threshold = float(self.config.get('threshold', 0.5))
            # Reconstruct a lightweight optimizer to access verification
            optimizer = SIMP3DOptimizer(self.nodes, self.elements, self.material, self.config)
            optimizer.rho = np.asarray(self.results['rho_optimized'], dtype=np.float64)
            verification = optimizer.run_verification_fea(self.forces, self.fixed_dofs, threshold=threshold)
            self.results['verification'] = verification
        except Exception as e:
            print(f'[WARNING] Verification FEA failed: {e}')

    def _check_unit_consistency(self):
        """Heuristic check for geometry/material unit consistency (§2.9).

        If the bounding box extent is very large relative to E0/yield_strength
        (which has units of length), the user likely modelled geometry in mm
        while using Pa for material properties — causing 10^6 displacement errors.
        """
        if self.nodes is None or self.material is None:
            return
        try:
            bb_min = np.min(self.nodes, axis=0)
            bb_max = np.max(self.nodes, axis=0)
            L_bb = float(np.linalg.norm(bb_max - bb_min))
            E0 = float(self.material.E0)
            ys = float(self.material.yield_strength)
            if L_bb < 1e-15 or E0 < 1e-6 or ys < 1e-6:
                return
            L_char = E0 / ys  # dimensionless if units are consistent
            ratio = L_bb / L_char
            if ratio > 1e3:
                print(f'\n  [WARNING] Bounding box diagonal = {L_bb:.3e}')
                print(f'  [WARNING] E0 / yield_strength = {L_char:.3e}')
                print(f'  [WARNING] Ratio = {ratio:.1e} >> 1 — geometry may be in mm while material is in Pa.')
                print(f'  [WARNING] Ensure geometry is in metres (SI) or scale material properties to match.\n')
            elif ratio < 1e-3:
                print(f'\n  [WARNING] Bounding box diagonal = {L_bb:.3e}')
                print(f'  [WARNING] E0 / yield_strength = {L_char:.3e}')
                print(f'  [WARNING] Ratio = {ratio:.1e} << 1 — geometry may be in metres while material is in MPa.')
                print(f'  [WARNING] Ensure consistent unit system.\n')
        except Exception:
            pass

    def run_interactive_workflow(self):
        """Run workflow"""
        self.print_banner()

        # Select Standard or Advanced settings mode first
        self.select_settings_mode()

        print('[INFO] 3D-only runtime active.')
        while not self.setup_3d_from_cad():
            pass
        
        while not self.select_optimization_method():
            pass
        
        while not self.setup_material_properties():
            pass
        
        # §2.9: Unit consistency check
        self._check_unit_consistency()

        # Define all optimization parameters BEFORE the GUI
        self.select_objective_function()
        while not self.configure_optimization():
            pass
        
        # Run the unified GUI (mesh review -> BC definition -> optimization -> results)
        if self.run_unified_gui_workflow():
            self._sync_auto_density_threshold()
            self.show_results()
            self._run_verification_fea_if_available()
            self.export_optimized_design()
        
        self.print_section('WORKFLOW COMPLETE')
        print('\nDone!')

def main():
    """Main entry point"""
    try:
        optimizer = InteractiveTopologyOptimizer()
        optimizer.run_interactive_workflow()
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


