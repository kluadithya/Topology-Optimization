"""
unified_gui.py - Single-window PyVista GUI for topology optimization.

All stages (BC definition -> mesh review -> live optimization -> final result)
happen inside one persistent plotter window.
"""

import numpy as np
import threading
import queue
import time
import inspect
from pathlib import Path

try:
    import pyvista as pv
    import warnings
    if hasattr(pv, 'PyVistaFutureWarning'):
        warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)
    warnings.filterwarnings('ignore', message='.*extract_surface.*')
    _HAS_PYVISTA = True
except ImportError:
    pv = None
    _HAS_PYVISTA = False


class UnifiedWorkflowGUI:
    """Single-window PyVista GUI that walks through the full optimization workflow."""

    _COL_BG = '#1e1e2e'
    _COL_SURFACE = '#8ea7c9'
    _COL_EDGE = '#5f7ea8'
    _COL_NODE = '#d5d8de'
    _COL_FIXED = '#22cc55'
    _COL_LOAD = '#ee3333'
    _COL_PICK = '#ffee00'
    _COL_MESH_EDGE = '#2f8cff'
    _COL_FACE_SEL = '#ffd166'
    _COL_EDGE_SEL = '#00d4ff'

    STAGE_BC = 'BC_DEFINITION'
    STAGE_MESH = 'MESH_REVIEW'
    STAGE_STRESS = 'STRESS_ANALYSIS'
    STAGE_OPT = 'OPTIMIZING'
    STAGE_POST_STRESS = 'POST_OPT_STRESS'
    STAGE_RESULTS = 'RESULTS'

    def __init__(self, nodes, elements, mode='3D', surf_faces=None,
                 material=None, config=None, optimizer_factory=None,
                 face_ids=None, edge_segments=None):
        self.mode = mode
        self._original_nodes = np.asarray(nodes, dtype=np.float64)
        self._original_elements = np.asarray(elements, dtype=np.int32)

        if self._original_nodes.shape[1] == 2:
            z = np.zeros((self._original_nodes.shape[0], 1))
            self.nodes = np.hstack([self._original_nodes, z])
        else:
            self.nodes = self._original_nodes.copy()

        self.elements = self._original_elements.copy()
        self.surf_faces = (np.asarray(surf_faces, dtype=np.int32)
                           if surf_faces is not None else None)

        self.material = material
        self.config = config or {}
        self.optimizer_factory = optimizer_factory

        self.face_ids = np.asarray(face_ids, dtype=np.int32) if face_ids is not None else None
        self.edge_segments = np.asarray(edge_segments, dtype=np.int32) if edge_segments is not None else None

        self.fixed_nodes = set()
        self.loads = {}
        self._undo_stack = []
        self._bc_mode = 'SUPPORTS'
        self._pick_scope = 'NODE'

        self._load_direction = 2 if mode == '3D' else 1
        self._load_magnitude = 0.0

        self.rho_result = None
        self.compliance_result = None
        self.compliance_history = []
        self.volume_history = []
        self._passive_solid_mask = None
        self._last_pick_meta = {'scope': 'NODE', 'n_nodes': 0, 'n_faces': 0, 'n_edges': 0, 'face_indices': []}
        self._support_face_indices = set()
        self._load_face_indices = set()

        self._remesh_callback = None
        self._remesh_queue = queue.Queue()
        self._remesh_in_progress = False
        self._remesh_thread = None
        self._remesh_progress = 0.0
        self._remesh_message = 'Idle'

        self._input_queue = queue.Queue()
        self._awaiting_input = False

        self._plotter = None
        self._stage = self.STAGE_BC
        self._status_actor = None
        self._help_actor = None
        self._info_actor = None
        self._quality_actor = None
        self._pick_actor = None
        self._pick_geom_actor = None
        self._indicator_actors = []
        self._mesh_actor = None
        self._node_actor = None
        self._density_actor = None
        self._scalar_bar_actor = None
        self._mask_actor = None

        # Stress analysis stage state
        self._stress_vm_all = None          # per-element von Mises stress
        self._stress_displacement = None    # displacement vector from static FEA
        self._stress_actor = None           # PyVista actor for stress heatmap
        self._deformed_actor = None         # PyVista actor for deformed shape
        self._vf_entered = False            # True after user enters volume fraction
        self._stress_running = False        # True while FEA is computing

        # Post-optimization stress analysis state
        self._post_stress_vm_all = None     # per-element von Mises on optimized design
        self._post_stress_displacement = None  # displacement on optimized design
        self._post_stress_actor = None      # PyVista actor for post-opt stress heatmap
        self._post_deformed_actor = None    # PyVista actor for post-opt deformed shape
        self._reopt_count = 0               # how many times user has re-optimized
        self._cached_fea_solver = None       # Cached FEA solver to avoid redundant precomputation

        self._opt_queue = queue.Queue()
        self._opt_done = threading.Event()
        self._abort = False
        self._closing = False

        self._boundary_faces_cache = None
        self._surface_edges_cache = None

        bbox = self.nodes.max(axis=0) - self.nodes.min(axis=0)
        self._ind_size = float(np.mean(bbox)) * 0.02
        self._arrow_scale = float(np.mean(bbox)) * 0.08

    def set_remesh_callback(self, cb):
        self._remesh_callback = cb

    def run(self):
        if not _HAS_PYVISTA:
            print('[ERROR] PyVista is required for the unified GUI.')
            return None, None, None, None

        self._print_welcome()
        try:
            self._create_plotter()
            self._enter_stage(self.STAGE_MESH)
            self._main_loop()
        except Exception as e:
            print(f'[ERROR] GUI crashed: {e}')
            import traceback
            traceback.print_exc()

        fixed_dofs, forces = self._build_bc_outputs()
        return fixed_dofs, forces, self.rho_result, self.compliance_result

    def _create_plotter(self):
        p = pv.Plotter(title='Topology Optimization - Unified Workflow', window_size=(1600, 950))
        # Patch for PyVista strict-mode crash during picking
        try:
            pv.set_new_attribute(p, 'pickpoint', None)
        except Exception:
            pass
        p.background_color = self._COL_BG
        try:
            axes_actor = p.add_axes(line_width=3, xlabel='X', ylabel='Y', zlabel='Z')
        except TypeError:
            axes_actor = p.add_axes(line_width=3)
        # Keep triad axis colors (RGB), but make X/Y/Z label text white and slightly larger.
        try:
            for cap_getter in ('GetXAxisCaptionActor2D', 'GetYAxisCaptionActor2D', 'GetZAxisCaptionActor2D'):
                cap = getattr(axes_actor, cap_getter)()
                tprop = cap.GetCaptionTextProperty()
                tprop.SetColor(1.0, 1.0, 1.0)
                tprop.SetBold(True)
                tprop.SetFontSize(18)
                cap.SetCaptionTextProperty(tprop)
        except Exception:
            pass
        p.show_grid(color='#555555', xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis')
        self._plotter = p

        for key in ('s', 'S'):
            p.add_key_event(key, lambda: self._key_S())
        for key in ('l', 'L'):
            p.add_key_event(key, lambda: self._key_L())
        for key in ('u', 'U'):
            p.add_key_event(key, lambda: self._key_U())
        for key in ('n', 'N'):
            p.add_key_event(key, lambda: self._key_N())
        for key in ('b', 'B'):
            p.add_key_event(key, lambda: self._key_B())
        for key in ('r', 'R'):
            p.add_key_event(key, lambda: self._key_R())
        for key in ('t', 'T'):
            p.add_key_event(key, lambda: self._key_T())
        for key in ('q', 'Q'):
            p.add_key_event(key, lambda: self._key_Q())
        for key in ('f', 'F'):
            p.add_key_event(key, lambda: self._key_F())
        for key in ('d', 'D'):
            p.add_key_event(key, lambda: self._key_D())

        for key in ('p', 'P'):
            p.add_key_event(key, lambda: self._key_pick_scope('NODE'))
        for key in ('g', 'G'):
            p.add_key_event(key, lambda: self._key_pick_scope('FACE'))
        for key in ('e', 'E'):
            p.add_key_event(key, lambda: self._key_pick_scope('EDGE'))

        for key in ('x', 'X'):
            p.add_key_event(key, lambda: self._key_dir(0))
        for key in ('y', 'Y'):
            p.add_key_event(key, lambda: self._key_dir(1))
        for key in ('z', 'Z'):
            p.add_key_event(key, lambda: self._key_dir(2))

        for key in ('m', 'M'):
            p.add_key_event(key, lambda: self._key_flip_mag())
        for key in ('v', 'V'):
            p.add_key_event(key, lambda: self._key_set_mag())
        for key in ('plus', 'equal'):
            p.add_key_event(key, lambda: self._key_mag_scale(1.25))
        for key in ('minus', 'underscore'):
            p.add_key_event(key, lambda: self._key_mag_scale(0.8))

        p.add_key_event('1', lambda: self._set_view('front'))
        p.add_key_event('2', lambda: self._set_view('back'))
        p.add_key_event('3', lambda: self._set_view('top'))
        p.add_key_event('4', lambda: self._set_view('bottom'))
        p.add_key_event('5', lambda: self._set_view('left'))
        p.add_key_event('6', lambda: self._set_view('right'))
        p.add_key_event('7', lambda: self._set_view('isometric'))
    def _main_loop(self):
        p = self._plotter
        p.show(interactive_update=True, auto_close=False)

        try:
            while True:
                if self._abort:
                    break
                if not p.render_window or not p.render_window.GetInteractor():
                    break

                if self._stage == self.STAGE_OPT:
                    self._process_opt_queue()
                    if self._opt_done.is_set() and not self._abort:
                        self._enter_stage(self.STAGE_POST_STRESS)

                if self._stage == self.STAGE_MESH:
                    self._process_remesh_queue()

                self._process_input_queue()

                try:
                    p.update()
                except Exception:
                    break
                time.sleep(0.02)
        except Exception:
            pass

        self._closing = True
        self._abort = True
        try:
            p.close()
        except Exception:
            pass

    def _enter_stage(self, stage):
        self._stage = stage
        print(f'\n  -- STAGE: {stage} --')
        self._clear_all_actors()

        if stage == self.STAGE_BC:
            self._setup_bc_stage()
        elif stage == self.STAGE_MESH:
            self._setup_mesh_stage()
        elif stage == self.STAGE_STRESS:
            self._setup_stress_stage()
        elif stage == self.STAGE_OPT:
            self._setup_opt_stage()
        elif stage == self.STAGE_POST_STRESS:
            self._setup_post_stress_stage()
        elif stage == self.STAGE_RESULTS:
            self._setup_results_stage()

        self._update_overlays()
        try:
            self._plotter.view_isometric()
            self._plotter.reset_camera()
            self._plotter.render()
        except Exception:
            pass

    def _clear_all_actors(self):
        actors = self._indicator_actors + [
            self._mesh_actor, self._node_actor, self._density_actor,
            self._status_actor, self._help_actor, self._info_actor, self._quality_actor,
            self._pick_actor, self._pick_geom_actor, self._scalar_bar_actor, self._mask_actor,
            self._stress_actor, self._deformed_actor,
            self._post_stress_actor, self._post_deformed_actor,
        ]
        for a in actors:
            if a is not None:
                try:
                    self._plotter.remove_actor(a)
                except Exception:
                    pass

        self._indicator_actors = []
        self._mesh_actor = None
        self._node_actor = None
        self._density_actor = None
        self._status_actor = None
        self._help_actor = None
        self._info_actor = None
        self._quality_actor = None
        self._pick_actor = None
        self._pick_geom_actor = None
        self._scalar_bar_actor = None
        self._stress_actor = None
        self._deformed_actor = None
        self._post_stress_actor = None
        self._post_deformed_actor = None

    def _setup_bc_stage(self):
        self._bc_mode = 'SUPPORTS'
        self._pick_scope = 'FACE'
        # Keep BC stage visually clean: no mesh facet edges.
        self._add_surface_mesh(opacity=0.98, show_edges=False)

        node_pd = pv.PolyData(self.nodes)
        self._node_actor = self._plotter.add_mesh(
            node_pd,
            color=self._COL_NODE,
            point_size=4,
            render_points_as_spheres=True,
            pickable=True,
            opacity=0.18,
            label='Nodes')

        self._set_bc_pickability()
        self._enable_picking()
        self._redraw_bc_indicators()

    def _setup_stress_stage(self):
        """Set up the stress analysis visualization stage.

        Runs a static FEA with full-density elements (rho=1) using the BCs
        defined in the previous stage, then visualizes per-element von Mises
        stress as a heatmap and optionally shows the deformed shape.
        """
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

        self._stress_running = True
        self._vf_entered = False
        self._update_overlays()

        # Show a waiting message
        self._info_actor = self._plotter.add_text(
            'Running static FEA...\nPlease wait',
            position='upper_left', font_size=14, color='white')
        try:
            self._plotter.render()
        except Exception:
            pass

        # Run FEA synchronously (it's fast for a single static solve)
        try:
            self._run_stress_analysis()
        except Exception as e:
            print(f'  [STRESS] FEA failed: {e}')
            import traceback; traceback.print_exc()
            self._stress_running = False
            if self._info_actor is not None:
                try:
                    self._plotter.remove_actor(self._info_actor)
                except Exception:
                    pass
            self._info_actor = self._plotter.add_text(
                f'STRESS ANALYSIS FAILED\n{e}\n\nPress B to go back',
                position='upper_left', font_size=14, color='#ff4444')
            self._update_overlays()
            return

        self._stress_running = False

        # Remove waiting text
        if self._info_actor is not None:
            try:
                self._plotter.remove_actor(self._info_actor)
            except Exception:
                pass
            self._info_actor = None

        # Render stress heatmap and deformed shape
        self._draw_stress_heatmap()

        # Build info overlay with stress statistics
        vm = self._stress_vm_all
        if vm is not None and len(vm) > 0:
            ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
            sf = float(self.config.get('safety_factor', 1.5))
            allowable = self._get_allowable_stress(sf)
            max_vm = float(np.max(vm))
            mean_vm = float(np.mean(vm))
            min_vm = float(np.min(vm))
            median_vm = float(np.median(vm))
            util = max_vm / max(allowable, 1e-12) * 100.0

            # Compute displacement stats
            disp_info = ''
            if self._stress_displacement is not None:
                u = np.asarray(self._stress_displacement, dtype=np.float64)
                u3 = u.reshape(-1, 3)
                disp_mag = np.linalg.norm(u3, axis=1)
                max_disp = float(np.max(disp_mag))
                disp_info = f'\nMax displacement: {max_disp:.3e} mm'

            # Suggest volume fraction based on absolute Yield Limits and stress utilization
            if util >= 100.0:
                suggest_vf = 1.0
                suggest_vf_str = '1.00 (WARNING: Fails yield limit)'
                vf_note = '(Solid structure is yielding, no material removal suggested)'
            elif util > 80.0:
                low_stress_frac = float(np.sum(vm < 0.20 * allowable)) / max(len(vm), 1)
                suggest_vf = float(np.clip(1.0 - low_stress_frac * 0.5, 0.70, 0.95))
                suggest_vf_str = f'~{suggest_vf:.2f}'
                vf_note = f'(~{low_stress_frac*100:.0f}% elements <20% allowable. High utilization)'
            else:
                low_stress_frac = float(np.sum(vm < 0.20 * allowable)) / max(len(vm), 1)
                suggest_vf = float(np.clip(1.0 - low_stress_frac * 0.8, 0.15, 0.85))
                suggest_vf_str = f'~{suggest_vf:.2f}'
                vf_note = f'(~{low_stress_frac*100:.0f}% elements <20% allowable)'

            info = (
                f'STATIC STRESS ANALYSIS\n'
                f'Max von Mises: {max_vm:.3e} MPa\n'
                f'Mean von Mises: {mean_vm:.3e} MPa\n'
                f'Median von Mises: {median_vm:.3e} MPa\n'
                f'Min von Mises: {min_vm:.3e} MPa\n'
                f'\nYield: {ys:.3e} MPa  SF: {sf:.1f}\n'
                f'Allowable: {allowable:.3e} MPa\n'
                f'Utilization: {util:.1f}%'
                f'{disp_info}\n'
                f'\nSuggested VF: {suggest_vf_str}\n'
                f'{vf_note}\n'
                f'\nPress V to set volume fraction'
            )
            self._info_actor = self._plotter.add_text(info, position='upper_left', font_size=11, color='white')
        else:
            self._info_actor = self._plotter.add_text(
                'STRESS ANALYSIS\nNo stress data available\n\nPress B to go back',
                position='upper_left', font_size=14, color='#ff4444')

        self._update_overlays()

        # Auto-prompt for volume fraction input
        print('\n  ========================================================')
        print('  STRESS ANALYSIS COMPLETE')
        print('  ========================================================')
        if vm is not None and len(vm) > 0:
            print(f'  Max von Mises stress: {float(np.max(vm)):.3e} MPa')
            print(f'  Suggested volume fraction: ~{suggest_vf:.2f}')
        print('  Press V in the GUI to set volume fraction / mass reduction.')
        print('  Press N to proceed to optimization, B to go back to BCs.')
        print('  ========================================================\n')

    def _get_or_create_fea_solver(self):
        """Return a cached FEA solver, creating one if needed.

        Avoids redundant precomputation of unit stiffness matrices,
        centroid B-matrices, and assembly scatter maps between stages.
        """
        from fea_solver_3d import FEASolver3D

        if self._cached_fea_solver is not None:
            # Verify the cached solver still matches the current mesh
            if (self._cached_fea_solver.n_nodes == self._original_nodes.shape[0] and
                    self._cached_fea_solver.n_elements == self._original_elements.shape[0]):
                return self._cached_fea_solver

        fea = FEASolver3D(self._original_nodes, self._original_elements, self.material)
        # Configure solver with user settings (was previously missing,
        # causing CG to fail and spsolve to hang on large meshes)
        try:
            fea.configure_linear_solver(
                mode=self.config.get('linear_solver', 'auto'),
                iterative_threshold_dofs=int(self.config.get('iterative_solver_dof_threshold', 50000)),
                cg_tol=float(self.config.get('iterative_solver_tol', 1e-8)),
                cg_maxiter=int(self.config.get('iterative_solver_maxiter', 2000)),
                use_pyamg=bool(self.config.get('use_pyamg_preconditioner', True)),
            )
        except Exception:
            pass
        self._cached_fea_solver = fea
        return fea

    def _run_stress_analysis(self):
        """Run static FEA with full-density elements and compute per-element stress."""
        import time as _time
        t0 = _time.perf_counter()
        print('  [STRESS] Running static FEA with full-density elements...')

        # Build BCs
        fixed_dofs, forces = self._build_bc_outputs()
        if forces is None or np.sum(np.abs(forces)) < 1e-10:
            raise ValueError('No loads defined — cannot run stress analysis')

        n_elem = int(self._original_elements.shape[0])
        rho_full = np.ones(n_elem, dtype=np.float64)

        # Use cached FEA solver (avoids redundant precomputation)
        fea = self._get_or_create_fea_solver()
        t1 = _time.perf_counter()
        pen = float(self.config.get('penalization', self.config.get('penalty', 3.0)))

        # Solve
        u = fea.solve(rho_full, forces, fixed_dofs, pen)
        self._stress_displacement = u.copy()
        t2 = _time.perf_counter()

        # Compute per-element von Mises stress
        print('  [STRESS] Computing per-element von Mises stress...')
        vm_all = fea.calculate_stress_all_gauss(u, rho_full, pen)
        self._stress_vm_all = vm_all
        t3 = _time.perf_counter()

        max_vm = float(np.max(vm_all)) if len(vm_all) > 0 else 0.0
        mean_vm = float(np.mean(vm_all)) if len(vm_all) > 0 else 0.0
        print(f'  [STRESS] Done. Max VM={max_vm:.3e} MPa, Mean VM={mean_vm:.3e} MPa')
        print(f'  [STRESS] Timing: init={t1-t0:.2f}s, solve={t2-t1:.2f}s, stress={t3-t2:.2f}s, total={t3-t0:.2f}s')

    def _draw_stress_heatmap(self):
        """Draw per-element von Mises stress heatmap with deformed shape overlay."""
        p = self._plotter
        if p is None or self._stress_vm_all is None:
            return

        vm = self._stress_vm_all
        n_elem = self.elements.shape[0]
        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0

        if n_per not in (4, 10):
            print('  [STRESS] Cannot display stress heatmap for non-tet elements')
            self._add_surface_mesh(opacity=1.0, show_edges=True)
            return

        try:
            # Build unstructured grid for stress visualization
            cells = np.hstack([
                np.full((n_elem, 1), n_per, dtype=np.int64),
                self.elements.astype(np.int64)
            ]).ravel()
            vtk_tet_type = 24 if n_per == 10 else 10
            celltypes = np.full(n_elem, vtk_tet_type, dtype=np.uint8)

            # Use original (undeformed) geometry for the stress heatmap
            grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
            vm_clipped = np.clip(np.asarray(vm, dtype=np.float64)[:n_elem], 0.0, None)
            grid['von_mises'] = vm_clipped

            # Threshold to show only boundary (extract surface for cleaner viz)
            try:
                surface = grid.extract_surface()
            except Exception:
                surface = grid

            # Professional engineering colormap: blue → cyan → green → yellow → red
            # This is more intuitive than a simple green-red diverging map
            stress_cmap = [
                '#0d47a1',   # deep blue (zero/minimal stress)
                '#1565c0',   # blue
                '#0097a7',   # teal
                '#00897b',   # green-teal
                '#2e7d32',   # green (moderate-low stress)
                '#558b2f',   # lime-green
                '#9e9d24',   # olive
                '#f9a825',   # amber/yellow
                '#ef6c00',   # orange (moderate-high stress)
                '#d32f2f',   # red (high stress)
                '#b71c1c',   # dark red (critical stress)
            ]

            max_vm = float(np.max(vm_clipped)) if len(vm_clipped) > 0 else 1.0
            self._stress_actor = p.add_mesh(
                surface,
                scalars='von_mises',
                cmap=stress_cmap,
                clim=[0.0, max_vm],
                show_edges=False,
                opacity=0.92,
                pickable=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Von Mises Stress (MPa)',
                    'color': 'white',
                    'fmt': '%.2e',
                    'n_labels': 7,
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'width': 0.08,
                    'height': 0.6,
                },
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.75,
                specular=0.15,
            )

            # --- Deformed shape overlay (wireframe) ---
            if self._stress_displacement is not None:
                try:
                    u = np.asarray(self._stress_displacement, dtype=np.float64)
                    u3 = u.reshape(-1, 3)

                    # Auto-scale: make max displacement ~2% of model bounding box
                    # Capped at 200x to avoid extreme distortion that looks "crashed"
                    bbox = np.max(self.nodes, axis=0) - np.min(self.nodes, axis=0)
                    model_size = float(np.linalg.norm(bbox))
                    max_disp = float(np.max(np.linalg.norm(u3, axis=1)))
                    if max_disp > 1e-20:
                        scale_factor = min(0.02 * model_size / max_disp, 200.0)
                    else:
                        scale_factor = 1.0

                    deformed_nodes = self.nodes + u3[:self.nodes.shape[0]] * scale_factor

                    # Build deformed grid
                    grid_def = pv.UnstructuredGrid(cells, celltypes, deformed_nodes)
                    try:
                        surface_def = grid_def.extract_surface()
                    except Exception:
                        surface_def = grid_def

                    self._deformed_actor = p.add_mesh(
                        surface_def,
                        color='#ffffff',
                        style='wireframe',
                        line_width=1.0,
                        opacity=0.15,
                        pickable=False,
                        lighting=False,
                    )
                    print(f'  [STRESS] Deformed shape overlay: scale factor = {scale_factor:.2f}x')
                except Exception as e:
                    print(f'  [STRESS] Deformed shape overlay skipped: {e}')

        except Exception as e:
            print(f'  [STRESS] Heatmap display error: {e}')
            import traceback; traceback.print_exc()
            self._add_surface_mesh(opacity=1.0, show_edges=True)

    # ---- POST-OPTIMIZATION STRESS ANALYSIS ----

    def _setup_post_stress_stage(self):
        """Set up the post-optimization stress analysis stage.

        Runs static FEA on the optimized (binarized) density field to show
        whether the optimized design meets stress requirements.  Provides
        comparison with the pre-optimization full-density stress results and
        gives the user Accept / Re-optimize options.
        """
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

        # Show waiting message
        self._info_actor = self._plotter.add_text(
            'Running post-optimization FEA...\nAnalysing optimized design',
            position='upper_left', font_size=14, color='white')
        self._update_overlays()
        try:
            self._plotter.render()
        except Exception:
            pass

        # Run FEA on optimized design synchronously
        try:
            self._run_post_opt_stress_analysis()
        except Exception as e:
            print(f'  [POST-STRESS] FEA failed: {e}')
            import traceback; traceback.print_exc()
            if self._info_actor is not None:
                try:
                    self._plotter.remove_actor(self._info_actor)
                except Exception:
                    pass
            self._info_actor = self._plotter.add_text(
                f'POST-OPT STRESS ANALYSIS FAILED\n{e}\n\nPress N to skip to results\nPress R to re-optimize',
                position='upper_left', font_size=14, color='#ff4444')
            self._update_overlays()
            return

        # Remove waiting text
        if self._info_actor is not None:
            try:
                self._plotter.remove_actor(self._info_actor)
            except Exception:
                pass
            self._info_actor = None

        # Draw post-opt stress heatmap
        self._draw_post_stress_heatmap()

        # Build comparison info overlay
        vm_post = self._post_stress_vm_all
        vm_pre = self._stress_vm_all
        if vm_post is not None and len(vm_post) > 0:
            ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
            sf = float(self.config.get('safety_factor', 1.5))
            allowable = self._get_allowable_stress(sf)

            max_vm_post = float(np.max(vm_post))
            mean_vm_post = float(np.mean(vm_post))
            util_post = max_vm_post / max(allowable, 1e-12) * 100.0

            # Displacement stats
            disp_info = ''
            if self._post_stress_displacement is not None:
                u = np.asarray(self._post_stress_displacement, dtype=np.float64)
                u3 = u.reshape(-1, 3)
                disp_mag = np.linalg.norm(u3, axis=1)
                max_disp = float(np.max(disp_mag))
                disp_info = f'\nMax displacement: {max_disp:.3e} mm'

            # Safety margin
            sf_margin = allowable / max(max_vm_post, 1e-12)

            # Mass report
            mass_info = ''
            mass_report = self._compute_mass_report(self.rho_result)
            if mass_report is not None:
                mass_info = (
                    f'\n\nMASS REPORT\n'
                    f'Original: {mass_report["mass_baseline"]:.3f} kg\n'
                    f'Optimized: {mass_report["mass_current"]:.3f} kg\n'
                    f'Saved: {mass_report["saved_pct"]:.1f}%'
                )

            # Pre-opt comparison
            compare_info = ''
            if vm_pre is not None and len(vm_pre) > 0:
                max_vm_pre = float(np.max(vm_pre))
                util_pre = max_vm_pre / max(allowable, 1e-12) * 100.0
                stress_change = ((max_vm_post - max_vm_pre) / max(max_vm_pre, 1e-12)) * 100.0
                compare_info = (
                    f'\n\nPRE vs POST COMPARISON\n'
                    f'Pre-opt max VM: {max_vm_pre:.3e} MPa\n'
                    f'Post-opt max VM: {max_vm_post:.3e} MPa\n'
                    f'Change: {stress_change:+.1f}%\n'
                    f'Pre-opt util: {util_pre:.1f}%\n'
                    f'Post-opt util: {util_post:.1f}%'
                )

            # Status verdict
            if util_post <= 100.0:
                verdict = 'PASS - Design is within allowable stress'
                verdict_color = '#00d26a'
            else:
                verdict = 'FAIL - Design exceeds allowable stress!'
                verdict_color = '#ff4444'

            reopt_label = f'  (re-optimization #{self._reopt_count})' if self._reopt_count > 0 else ''

            info = (
                f'POST-OPTIMIZATION STRESS ANALYSIS{reopt_label}\n'
                f'Status: {verdict}\n\n'
                f'Max von Mises: {max_vm_post:.3e} MPa\n'
                f'Mean von Mises: {mean_vm_post:.3e} MPa\n'
                f'Yield: {ys:.3e} MPa  SF: {sf:.1f}\n'
                f'Allowable: {allowable:.3e} MPa\n'
                f'Utilization: {util_post:.1f}%\n'
                f'Safety margin: {sf_margin:.2f}x'
                f'{disp_info}'
                f'{mass_info}'
                f'{compare_info}\n\n'
                f'N = Accept & view results\n'
                f'R = Re-optimize (change VF)'
            )
            self._info_actor = self._plotter.add_text(info, position='upper_left', font_size=10, color='white')

            # Print summary to terminal
            print(f'\n  ========================================================')
            print(f'  POST-OPTIMIZATION STRESS ANALYSIS{reopt_label}')
            print(f'  ========================================================')
            print(f'  Verdict: {verdict}')
            print(f'  Max von Mises: {max_vm_post:.3e} MPa')
            print(f'  Utilization: {util_post:.1f}%')
            print(f'  Safety margin: {sf_margin:.2f}x')
            if mass_report is not None:
                print(f'  Mass saved: {mass_report["saved_pct"]:.1f}%')
            print(f'  Press N to accept, R to re-optimize with different VF')
            print(f'  ========================================================\n')
        else:
            self._info_actor = self._plotter.add_text(
                'POST-OPTIMIZATION STRESS\nNo stress data available\n\nPress N for results\nPress R to re-optimize',
                position='upper_left', font_size=14, color='#ff4444')

        self._update_overlays()

    def _run_post_opt_stress_analysis(self):
        """Run static FEA on the optimized (binarized) design."""
        print('  [POST-STRESS] Running static FEA on optimized design...')

        if self.rho_result is None:
            raise ValueError('No optimization result available')

        # Build BCs
        fixed_dofs, forces = self._build_bc_outputs()
        if forces is None or np.sum(np.abs(forces)) < 1e-10:
            raise ValueError('No loads defined — cannot run stress analysis')

        # Binarize the density field using display cutoff
        thr = self._compute_display_cutoff(self.rho_result)
        rho_min = float(self.config.get('rho_min', 1e-6))
        rho_bin = np.where(
            np.asarray(self.rho_result, dtype=np.float64) >= thr,
            1.0, rho_min
        )
        # Enforce passive solid regions
        if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(rho_bin):
            rho_bin[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0

        n_solid = int(np.sum(rho_bin > 0.5))
        n_total = len(rho_bin)
        print(f'  [POST-STRESS] Binary design: {n_solid}/{n_total} solid elements (cutoff={thr:.3f})')

        # Reuse cached FEA solver (avoids redundant precomputation)
        fea = self._get_or_create_fea_solver()
        pen = float(self.config.get('penalization', self.config.get('penalty', 3.0)))
        u = fea.solve(rho_bin, forces, fixed_dofs, pen)
        self._post_stress_displacement = u.copy()

        # Compute per-element von Mises stress
        print('  [POST-STRESS] Computing per-element von Mises stress...')
        vm_all = fea.calculate_stress_all_gauss(u, rho_bin, pen)
        self._post_stress_vm_all = vm_all

        max_vm = float(np.max(vm_all)) if len(vm_all) > 0 else 0.0
        mean_vm = float(np.mean(vm_all)) if len(vm_all) > 0 else 0.0
        print(f'  [POST-STRESS] Done. Max VM={max_vm:.3e} MPa, Mean VM={mean_vm:.3e} MPa')

    def _draw_post_stress_heatmap(self):
        """Draw per-element von Mises stress on optimized (solid) elements only."""
        p = self._plotter
        if p is None or self._post_stress_vm_all is None or self.rho_result is None:
            return

        vm = self._post_stress_vm_all
        n_elem = self.elements.shape[0]
        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0

        if n_per not in (4, 10):
            print('  [POST-STRESS] Cannot display stress heatmap for non-tet elements')
            self._add_surface_mesh(opacity=1.0, show_edges=True)
            return

        try:
            # Build unstructured grid
            cells = np.hstack([
                np.full((n_elem, 1), n_per, dtype=np.int64),
                self.elements.astype(np.int64)
            ]).ravel()
            vtk_tet_type = 24 if n_per == 10 else 10
            celltypes = np.full(n_elem, vtk_tet_type, dtype=np.uint8)

            grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
            vm_clipped = np.clip(np.asarray(vm, dtype=np.float64)[:n_elem], 0.0, None)
            grid['von_mises'] = vm_clipped

            # Add density to threshold out void elements
            thr = self._compute_display_cutoff(self.rho_result)
            rho_arr = np.asarray(self.rho_result, dtype=np.float64)[:n_elem]
            grid['density'] = rho_arr

            # Threshold to show only solid elements
            try:
                solid_grid = grid.threshold(value=thr, scalars='density')
            except Exception:
                solid_grid = grid

            # Extract surface for cleaner visualization
            try:
                surface = solid_grid.extract_surface()
            except Exception:
                surface = solid_grid

            # Professional engineering colormap
            stress_cmap = [
                '#0d47a1', '#1565c0', '#0097a7', '#00897b',
                '#2e7d32', '#558b2f', '#9e9d24', '#f9a825',
                '#ef6c00', '#d32f2f', '#b71c1c',
            ]

            max_vm = float(np.max(vm_clipped)) if len(vm_clipped) > 0 else 1.0
            self._post_stress_actor = p.add_mesh(
                surface,
                scalars='von_mises',
                cmap=stress_cmap,
                clim=[0.0, max_vm],
                show_edges=False,
                opacity=0.92,
                pickable=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Von Mises Stress - Optimized (MPa)',
                    'color': 'white',
                    'fmt': '%.2e',
                    'n_labels': 7,
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'width': 0.08,
                    'height': 0.6,
                },
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.75,
                specular=0.15,
            )

            # Deformed shape overlay
            if self._post_stress_displacement is not None:
                try:
                    u = np.asarray(self._post_stress_displacement, dtype=np.float64)
                    u3 = u.reshape(-1, 3)

                    bbox = np.max(self.nodes, axis=0) - np.min(self.nodes, axis=0)
                    model_size = float(np.linalg.norm(bbox))
                    max_disp = float(np.max(np.linalg.norm(u3, axis=1)))
                    if max_disp > 1e-20:
                        scale_factor = min(0.02 * model_size / max_disp, 200.0)
                    else:
                        scale_factor = 1.0

                    deformed_nodes = self.nodes + u3[:self.nodes.shape[0]] * scale_factor

                    grid_def = pv.UnstructuredGrid(cells, celltypes, deformed_nodes)
                    grid_def['density'] = rho_arr
                    try:
                        solid_def = grid_def.threshold(value=thr, scalars='density')
                    except Exception:
                        solid_def = grid_def
                    try:
                        surface_def = solid_def.extract_surface()
                    except Exception:
                        surface_def = solid_def

                    self._post_deformed_actor = p.add_mesh(
                        surface_def,
                        color='#ffffff',
                        style='wireframe',
                        line_width=1.0,
                        opacity=0.15,
                        pickable=False,
                        lighting=False,
                    )
                    print(f'  [POST-STRESS] Deformed shape overlay: scale factor = {scale_factor:.2f}x')
                except Exception as e:
                    print(f'  [POST-STRESS] Deformed shape overlay skipped: {e}')

        except Exception as e:
            print(f'  [POST-STRESS] Heatmap display error: {e}')
            import traceback; traceback.print_exc()
            # Fallback: show density mesh
            if self.rho_result is not None:
                self._draw_density_mesh(self.rho_result)

    def _set_bc_pickability(self):
        """Prioritize face clicks in FACE/EDGE mode by disabling node picking."""
        if self._node_actor is None:
            return
        try:
            if self._pick_scope == 'NODE':
                self._node_actor.SetPickable(True)
                self._node_actor.GetProperty().SetOpacity(0.35)
            else:
                self._node_actor.SetPickable(False)
                self._node_actor.GetProperty().SetOpacity(0.05)
        except Exception:
            pass

    def _enable_picking(self):
        # Prefer surface point picking so clicking on a face works directly.
        try:
            self._plotter.enable_surface_point_picking(
                callback=self._on_pick,
                show_point=False,
                show_message=False,
                left_clicking=True,
                pickable_window=False,
            )
            return
        except Exception:
            pass

        pick_kwargs = dict(
            callback=self._on_pick,
            show_message=False,
            point_size=1,
            color='yellow',
        )
        try:
            self._plotter.enable_point_picking(
                use_picker='point', left_clicking=True,
                pickable_window=False, **pick_kwargs)
        except TypeError:
            try:
                self._plotter.enable_point_picking(
                    use_mesh=True, left_clicking=True, **pick_kwargs)
            except TypeError:
                self._plotter.enable_point_picking(use_mesh=True, **pick_kwargs)

    def _on_pick(self, point_data, *_args):
        if self._stage != self.STAGE_BC:
            return
        try:
            coords = self._extract_coords(point_data)
            if coords is None:
                return

            picked_nodes = self._resolve_pick_targets(coords)
            if not picked_nodes:
                return

            if self._bc_mode == 'SUPPORTS':
                self._apply_supports(picked_nodes)
            else:
                self._apply_loads(picked_nodes)

            self._redraw_bc_indicators()
            self._update_overlays()
        except Exception as e:
            print(f'  [WARNING] Pick error: {e}')

    def _extract_coords(self, point_data):
        coords = None
        if isinstance(point_data, np.ndarray) and point_data.shape == (3,):
            coords = point_data.astype(np.float64)
        elif hasattr(point_data, 'points'):
            pts = np.asarray(point_data.points)
            if pts.ndim == 2 and pts.shape[0] == 1:
                coords = pts[0].astype(np.float64)
            elif pts.ndim == 2 and pts.shape[0] > 1:
                coords = np.asarray(point_data.center, dtype=np.float64)
            elif pts.ndim == 1 and len(pts) == 3:
                coords = pts.astype(np.float64)

        if coords is None and self._plotter is not None:
            try:
                pp = self._plotter.picked_point
                if pp is not None:
                    coords = np.asarray(pp, dtype=np.float64)
            except Exception:
                pass

        if coords is None:
            try:
                coords = np.asarray(point_data, dtype=np.float64).flatten()[:3]
            except Exception:
                pass

        if coords is not None and len(coords) == 3 and np.isfinite(coords).all():
            return coords
        return None

    def _resolve_pick_targets(self, coords):
        if self._pick_scope == 'NODE':
            dists = np.linalg.norm(self.nodes - coords.reshape(1, 3), axis=1)
            nid = int(np.argmin(dists))
            c = self.nodes[nid]
            self._last_pick_meta = {'scope': 'NODE', 'n_nodes': 1, 'n_faces': 0, 'n_edges': 0}
            print(f'  Picked node {nid}  @ ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})')
            self._show_pick_highlight(c)
            return [nid]

        faces = self._get_display_faces()
        if faces is None or len(faces) == 0:
            print('  [WARNING] No surface faces available for face/edge picking')
            return []

        if self._pick_scope == 'FACE':
            centers = self.nodes[faces].mean(axis=1)
            fidx = int(np.argmin(np.linalg.norm(centers - coords.reshape(1, 3), axis=1)))
            face_idx = self._group_face_region(faces, fidx)
            
            # Use full topological boundary faces (which include midside nodes) if available
            if self._boundary_faces_cache is not None and face_idx.max() < self._boundary_faces_cache.shape[0]:
                face_group = self._boundary_faces_cache[face_idx]
            else:
                face_group = faces[face_idx]

            picked_nodes = sorted(set(int(n) for n in face_group.reshape(-1).tolist()))
            self._last_pick_meta = {'scope': 'FACE', 'n_nodes': len(picked_nodes), 'n_faces': int(face_group.shape[0]), 'n_edges': 0, 'face_indices': [int(i) for i in face_idx.tolist()]}
            cg = self.nodes[face_group.reshape(-1)].reshape(-1, 3).mean(axis=0)
            print(f'  Picked face region with {face_group.shape[0]} triangle(s), {len(picked_nodes)} nodes @ ({cg[0]:.2f}, {cg[1]:.2f}, {cg[2]:.2f})')
            self._show_face_highlight(faces[face_idx])  # Highlight only corners for PyVista
            return picked_nodes

        edges = self._get_surface_edges()
        if edges is None or len(edges) == 0:
            print('  [WARNING] No edges available for edge picking')
            return []

        seg_a = self.nodes[edges[:, 0]]
        seg_b = self.nodes[edges[:, 1]]
        mid = 0.5 * (seg_a + seg_b)
        eidx = int(np.argmin(np.linalg.norm(mid - coords.reshape(1, 3), axis=1)))
        edge_group = self._group_edge_region(edges, eidx)

        group_nodes = np.unique(edge_group.reshape(-1))
        self._last_pick_meta = {'scope': 'EDGE', 'n_nodes': int(group_nodes.size), 'n_faces': 0, 'n_edges': int(edge_group.shape[0])}
        cg = self.nodes[group_nodes].mean(axis=0)
        print(f'  Picked edge region with {edge_group.shape[0]} segment(s), {group_nodes.size} nodes @ ({cg[0]:.2f}, {cg[1]:.2f}, {cg[2]:.2f})')
        self._show_edge_highlight(edge_group[0])
        return sorted(int(n) for n in group_nodes.tolist())

    def _group_face_region(self, faces, seed_idx):
        """Return connected face-region indices; prefer CAD face IDs, fallback to planar region growth."""
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            return np.asarray([int(seed_idx)], dtype=np.int64)

        if self.face_ids is not None:
            face_ids = np.asarray(self.face_ids, dtype=np.int64).reshape(-1)
            if face_ids.size == faces.shape[0]:
                fid = int(face_ids[seed_idx])
                mask = (face_ids == fid)
                if np.sum(mask) > 1:
                    return np.where(mask)[0].astype(np.int64)

        tri_pts = self.nodes[faces]
        v1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        v2 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
        normals = np.cross(v1, v2)
        nrm = np.linalg.norm(normals, axis=1)
        valid = nrm > 1e-12
        if not np.any(valid) or not valid[seed_idx]:
            return np.asarray([int(seed_idx)], dtype=np.int64)

        normals[valid] /= nrm[valid][:, None]
        n0 = normals[seed_idx]
        c0 = tri_pts[seed_idx].mean(axis=0)

        centers = tri_pts.mean(axis=1)
        ang_cos = np.abs(normals @ n0)
        diag = float(np.linalg.norm(np.max(self.nodes, axis=0) - np.min(self.nodes, axis=0)))
        plane_tol = max(1e-6, 0.005 * diag)
        plane_dist = np.abs((centers - c0) @ n0)
        candidate = valid & (ang_cos >= 0.9396926) & (plane_dist <= plane_tol)

        if np.sum(candidate) < 12:
            plane_tol2 = max(plane_tol, 0.02 * diag)
            candidate = valid & (ang_cos >= 0.8660254) & (plane_dist <= plane_tol2)

        edge_to_faces = {}
        for i, tri in enumerate(faces):
            if not candidate[i]:
                continue
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                edge_to_faces.setdefault(e, []).append(i)

        visited = np.zeros(faces.shape[0], dtype=bool)
        stack = [int(seed_idx)]
        visited[seed_idx] = True
        region = []

        while stack:
            i = stack.pop()
            if not candidate[i]:
                continue
            region.append(i)
            tri = faces[i]
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                for j in edge_to_faces.get(e, []):
                    if not visited[j]:
                        visited[j] = True
                        stack.append(j)

        if not region:
            return np.asarray([int(seed_idx)], dtype=np.int64)
        return np.asarray(region, dtype=np.int64)

    def _group_edge_region(self, edges, seed_idx):
        """Return connected, near-collinear edge chain for uniform edge selection."""
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
            return edges[seed_idx:seed_idx + 1]

        pts_a = self.nodes[edges[:, 0]]
        pts_b = self.nodes[edges[:, 1]]
        vec = pts_b - pts_a
        ln = np.linalg.norm(vec, axis=1)
        valid = ln > 1e-12
        if not np.any(valid) or not valid[seed_idx]:
            return edges[seed_idx:seed_idx + 1]

        dirs = np.zeros_like(vec)
        dirs[valid] = vec[valid] / ln[valid][:, None]
        d0 = dirs[seed_idx]
        collinear = np.abs(np.einsum('ij,j->i', dirs, d0)) >= 0.9848078  # cos(10deg)

        node_to_edges = {}
        for i, e in enumerate(edges):
            if not collinear[i]:
                continue
            n0, n1 = int(e[0]), int(e[1])
            node_to_edges.setdefault(n0, []).append(i)
            node_to_edges.setdefault(n1, []).append(i)

        visited = np.zeros(edges.shape[0], dtype=bool)
        stack = [int(seed_idx)]
        visited[seed_idx] = True
        region = []

        while stack:
            i = stack.pop()
            if not collinear[i]:
                continue
            region.append(i)
            n0, n1 = int(edges[i, 0]), int(edges[i, 1])
            for n in (n0, n1):
                for j in node_to_edges.get(n, []):
                    if not visited[j]:
                        visited[j] = True
                        stack.append(j)

        if not region:
            return edges[seed_idx:seed_idx + 1]
        return edges[np.asarray(region, dtype=np.int64)]

    def _apply_supports(self, node_ids):
        prev = {int(n): (int(n) in self.fixed_nodes) for n in node_ids}
        prev_support_faces = set(self._support_face_indices)
        prev_load_faces = set(self._load_face_indices)
        scope = str(self._last_pick_meta.get('scope', 'NODE'))
        if all(prev.values()):
            for n in node_ids:
                self.fixed_nodes.discard(int(n))
            if scope == 'FACE':
                print(f"  x Support REMOVED from FACE region: {self._last_pick_meta.get('n_faces', 0)} triangles, {len(node_ids)} nodes")
            elif scope == 'EDGE':
                print(f"  x Support REMOVED from EDGE region: {self._last_pick_meta.get('n_edges', 0)} segments, {len(node_ids)} nodes")
            else:
                print(f'  x Support REMOVED from {len(node_ids)} node(s)')
        else:
            for n in node_ids:
                self.fixed_nodes.add(int(n))
            if scope == 'FACE':
                print(f"  + Support ADDED on FACE region: {self._last_pick_meta.get('n_faces', 0)} triangles, {len(node_ids)} nodes")
            elif scope == 'EDGE':
                print(f"  + Support ADDED on EDGE region: {self._last_pick_meta.get('n_edges', 0)} segments, {len(node_ids)} nodes")
            else:
                print(f'  + Support ADDED to {len(node_ids)} node(s)')
        if scope == 'FACE':
            face_idx = [int(i) for i in self._last_pick_meta.get('face_indices', [])]
            if all(prev.values()):
                self._support_face_indices.difference_update(face_idx)
            else:
                self._support_face_indices.update(face_idx)
        self._sync_face_region_sets()
        self._undo_stack.append(('restore_supports', {'node_prev': prev, 'support_faces_prev': prev_support_faces, 'load_faces_prev': prev_load_faces}))

    def _apply_loads(self, node_ids):
        ndim = 2 if self.mode == '2D' else 3
        direction = int(np.clip(self._load_direction, 0, ndim - 1))
        mag = float(self._load_magnitude)
        mag_each = mag / float(len(node_ids)) if len(node_ids) > 1 else mag

        prev = {int(n): self.loads.get(int(n), None) for n in node_ids}
        prev_support_faces = set(self._support_face_indices)
        prev_load_faces = set(self._load_face_indices)
        for n in node_ids:
            self.loads[int(n)] = {'dir': direction, 'mag': mag_each}

        labels = ['X', 'Y', 'Z']
        scope = str(self._last_pick_meta.get('scope', 'NODE'))
        if scope == 'FACE':
            print(f"  + Load {mag_each:.2f} N in {labels[direction]} applied on FACE region: {self._last_pick_meta.get('n_faces', 0)} triangles, {len(node_ids)} nodes")
        elif scope == 'EDGE':
            print(f"  + Load {mag_each:.2f} N in {labels[direction]} applied on EDGE region: {self._last_pick_meta.get('n_edges', 0)} segments, {len(node_ids)} nodes")
        else:
            print(f'  + Load {mag_each:.2f} N in {labels[direction]} applied to {len(node_ids)} node(s)')
        if scope == 'FACE':
            face_idx = [int(i) for i in self._last_pick_meta.get('face_indices', [])]
            self._load_face_indices.update(face_idx)
        self._sync_face_region_sets()
        self._undo_stack.append(('restore_loads', {'load_prev': prev, 'support_faces_prev': prev_support_faces, 'load_faces_prev': prev_load_faces}))

    def _show_pick_highlight(self, coords):
        if self._plotter is None:
            return
        try:
            if self._pick_actor is not None:
                self._plotter.remove_actor(self._pick_actor)
            sphere = pv.Sphere(radius=self._ind_size * 1.5, center=coords)
            self._pick_actor = self._plotter.add_mesh(
                sphere, color=self._COL_PICK, opacity=0.85, pickable=False)
        except Exception:
            pass

    def _show_face_highlight(self, face_nodes):
        if self._plotter is None:
            return
        try:
            if self._pick_geom_actor is not None:
                self._plotter.remove_actor(self._pick_geom_actor)
            tri = np.asarray(face_nodes, dtype=np.int64)
            if tri.ndim == 1:
                tri = tri.reshape(1, 3)
            if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0:
                return
            faces_pv = np.hstack([
                np.full((tri.shape[0], 1), 3, dtype=np.int64),
                tri
            ]).ravel()
            pd = pv.PolyData(self.nodes, faces_pv)
            self._pick_geom_actor = self._plotter.add_mesh(
                pd, color=self._COL_FACE_SEL, opacity=0.6,
                show_edges=False, pickable=False)
        except Exception:
            pass

    def _show_edge_highlight(self, edge_nodes):
        if self._plotter is None:
            return
        try:
            if self._pick_geom_actor is not None:
                self._plotter.remove_actor(self._pick_geom_actor)
            a = self.nodes[int(edge_nodes[0])]
            b = self.nodes[int(edge_nodes[1])]
            line = pv.Line(a, b, resolution=1)
            self._pick_geom_actor = self._plotter.add_mesh(
                line.tube(radius=max(self._ind_size * 0.3, 1e-6)),
                color=self._COL_EDGE_SEL, opacity=0.95, pickable=False)
        except Exception:
            pass



    def _key_Q(self):
        if self._closing:
            return
        print('  [QUIT] Closing GUI...')
        self._abort = True
        self._closing = True

    def _key_F(self):
        try:
            self._plotter.reset_camera()
            self._plotter.render()
        except Exception:
            pass

    def _key_pick_scope(self, scope):
        if self._stage != self.STAGE_BC:
            return
        self._pick_scope = scope
        self._set_bc_pickability()
        print(f'  [PICK] Scope -> {scope}')
        self._update_overlays()

    def _key_dir(self, d):
        if self._stage != self.STAGE_BC:
            return
        ndim = 2 if self.mode == '2D' else 3
        if d >= ndim:
            return
        self._load_direction = int(d)
        for nid in list(self.loads.keys()):
            self.loads[nid]['dir'] = self._load_direction
        labels = ['X', 'Y', 'Z']
        print(f'  [LOAD] Direction -> {labels[d]} (applied to existing loads)')
        self._redraw_bc_indicators()
        self._update_overlays()

    def _key_flip_mag(self):
        if self._stage != self.STAGE_BC:
            return
        self._load_magnitude *= -1.0
        for nid in list(self.loads.keys()):
            self.loads[nid]['mag'] = -float(self.loads[nid]['mag'])
        print(f'  [LOAD] Magnitude sign flipped -> {self._load_magnitude:.2f} N (applied to existing loads)')
        self._redraw_bc_indicators()
        self._update_overlays()

    def _clear_scalar_bars(self):
        try:
            if hasattr(self._plotter, 'scalar_bars'):
                for title in list(self._plotter.scalar_bars.keys()):
                    self._plotter.remove_scalar_bar(title)
            else:
                self._plotter.remove_scalar_bar()
        except Exception:
            pass

    def _safe_remove_actor(self, actor):
        if self._stage != self.STAGE_BC:
            return
        f = float(factor)
        self._load_magnitude *= f
        for nid in list(self.loads.keys()):
            self.loads[nid]['mag'] = float(self.loads[nid]['mag']) * f
        print(f'  [LOAD] Magnitude -> {self._load_magnitude:.2f} N (scaled existing loads)')
        self._redraw_bc_indicators()
        self._update_overlays()

    def _key_set_mag(self):
        if self._stage == self.STAGE_STRESS:
            # In stress analysis stage, V sets volume fraction
            if self._awaiting_input:
                print('  [VF] Already waiting for input in terminal...')
                return
            self._awaiting_input = True
            print('  [VF] Type volume fraction or mass reduction in the terminal window...')
            t = threading.Thread(target=self._read_vf_input, daemon=True)
            t.start()
            return
        if self._stage != self.STAGE_BC:
            return
        if self._awaiting_input:
            print('  [LOAD] Already waiting for input in terminal...')
            return
        self._awaiting_input = True
        print('  [LOAD] Type magnitude in the terminal window...')
        t = threading.Thread(target=self._read_magnitude_input, daemon=True)
        t.start()

    def _key_D(self):
        if self._stage != self.STAGE_BC:
            return
        if self._awaiting_input:
            return
        self._awaiting_input = True
        print('\n  [INPUT AWAITING IN TERMINAL]')
        t = threading.Thread(target=self._read_distance_input, daemon=True)
        t.start()

    def _read_distance_input(self):
        try:
            mode_str = 'SUPPORTS' if self._bc_mode == 'SUPPORTS' else 'LOADS'
            raw = input(f'  Enter protection distance for {mode_str} [mm]: ').strip()
            if raw:
                self._input_queue.put(('dist', (self._bc_mode, float(raw))))
            else:
                self._input_queue.put(('dist_cancel', None))
        except (ValueError, KeyboardInterrupt):
            self._input_queue.put(('dist_cancel', None))

    def _read_magnitude_input(self):
        try:
            raw = input('  Enter load magnitude [N] (signed, e.g. -1000): ').strip()
            if raw:
                self._input_queue.put(('mag', float(raw)))
            else:
                self._input_queue.put(('mag_cancel', None))
        except (ValueError, KeyboardInterrupt):
            self._input_queue.put(('mag_cancel', None))

    def _read_vf_input(self):
        """Background thread: read volume fraction or mass reduction from terminal."""
        try:
            obj = str(self.config.get('objective_function', 'stiffness')).lower()
            if obj == 'weight':
                prompt = '  Enter mass reduction % (e.g. 30 for 30% reduction, 1-90): '
            else:
                prompt = '  Enter volume fraction (0.0-1.0) or mass reduction with %  (e.g. 0.3 or 70%): '
            raw = input(prompt).strip()
            if raw:
                self._input_queue.put(('vf', raw))
            else:
                self._input_queue.put(('vf_cancel', None))
        except (ValueError, KeyboardInterrupt):
            self._input_queue.put(('vf_cancel', None))

    def _process_input_queue(self):
        while not self._input_queue.empty():
            try:
                kind, val = self._input_queue.get_nowait()
            except Exception:
                break
            self._awaiting_input = False
            if kind == 'mag' and val is not None:
                new_mag = float(val)
                old_mag = self._load_magnitude if abs(self._load_magnitude) > 1e-12 else None
                self._load_magnitude = new_mag
                if old_mag is None:
                    mag_each = new_mag / max(1, len(self.loads))
                    for nid in list(self.loads.keys()):
                        self.loads[nid]['mag'] = mag_each
                else:
                    scale = new_mag / old_mag
                    for nid in list(self.loads.keys()):
                        self.loads[nid]['mag'] = float(self.loads[nid]['mag']) * scale
                print(f'  [LOAD] Magnitude set -> {self._load_magnitude:.2f} N (applied to existing loads)')
                self._redraw_bc_indicators()
                self._update_overlays()
            elif kind == 'mag_cancel':
                print('  [LOAD] Magnitude input cancelled')
            elif kind == 'dist' and val is not None:
                mode, new_dist = val
                if mode == 'SUPPORTS':
                    self.config['support_non_design_distance_mm'] = new_dist
                else:
                    self.config['force_non_design_distance_mm'] = new_dist
                print(f'  [{mode}] Protection distance set -> {new_dist:.2f} mm')
                self._redraw_bc_indicators()
                self._update_overlays()
            elif kind == 'dist_cancel':
                print('  [BC] Distance input cancelled')
            elif kind == 'new_mask':
                self._draw_non_design_mask(val)
            elif kind == 'vf' and val is not None:
                raw_str = str(val).strip()
                obj = str(self.config.get('objective_function', 'stiffness')).lower()
                try:
                    if raw_str.endswith('%') or obj == 'weight':
                        # Interpret as mass reduction percentage
                        num_str = raw_str.rstrip('%').strip()
                        reduction = float(np.clip(float(num_str), 1.0, 90.0))
                        vf = float(np.clip(1.0 - reduction / 100.0, 0.05, 0.99))
                        self.config['weight_reduction_percent'] = reduction
                        self.config['target_weight_reduction_percent'] = reduction
                        self.config['volume_fraction'] = vf
                        self.config['volfrac'] = vf
                        self._vf_entered = True
                        print(f'  [VF] Mass reduction: {reduction:.1f}% -> volume fraction {vf:.3f}')
                    else:
                        vf = float(np.clip(float(raw_str), 0.05, 0.99))
                        self.config['volume_fraction'] = vf
                        self.config['volfrac'] = vf
                        self._vf_entered = True
                        print(f'  [VF] Volume fraction set to {vf:.3f}')
                except (ValueError, TypeError) as e:
                    print(f'  [VF] Invalid input: {e}')
                self._update_overlays()
            elif kind == 'vf_cancel':
                print('  [VF] Volume fraction input cancelled')

    def _set_view(self, direction):
        if self._plotter is None:
            return
        try:
            views = {
                'front': self._plotter.view_xz,
                'back': lambda: self._plotter.view_xz(negative=True),
                'top': self._plotter.view_xy,
                'bottom': lambda: self._plotter.view_xy(negative=True),
                'left': self._plotter.view_yz,
                'right': lambda: self._plotter.view_yz(negative=True),
                'isometric': self._plotter.view_isometric,
            }
            if direction in views:
                views[direction]()
                self._plotter.render()
        except Exception:
            pass

    def _update_overlays(self):
        if self._abort or self._closing:
            return
        p = self._plotter
        if p is None:
            return

        if self._status_actor is not None:
            try:
                p.remove_actor(self._status_actor)
            except Exception:
                pass
        
        if self._stage == self.STAGE_BC:
            self._status_actor = p.add_text(self._status_text(), position='upper_left', font_size=14, color='white')
        else:
            self._status_actor = None

        if self._help_actor is not None:
            try:
                p.remove_actor(self._help_actor)
            except Exception:
                pass
        self._help_actor = p.add_text(self._help_text(), position='lower_right', font_size=11, color='#b8b8b8')

    def _status_text(self):
        if self._stage == self.STAGE_BC:
            load_line = ''
            if len(self.loads) > 0:
                axis = ['X', 'Y', 'Z'][int(self._load_direction)] if 0 <= self._load_direction <= 2 else '?'
                sign = '+' if self._load_magnitude >= 0 else '-'
                load_line = f'Load tool: [{axis}{sign}] mag={abs(self._load_magnitude):.1f} N\n'
            return (
                f'STAGE: Support and Force Define\n'
                f'Mode : {self._bc_mode}\n'
                f'Pick : {self._pick_scope}\n'
                f'{load_line}'
                f'Fixed: {len(self.fixed_nodes)} nodes\n'
                f'Loads: {len(self.loads)} nodes')
        if self._stage == self.STAGE_MESH:
            remesh_state = 'REMESHING' if self._remesh_in_progress else 'READY'
            pct = float(np.clip(self._remesh_progress, 0.0, 100.0))
            filled = int(round(pct / 5.0))
            bar = '#' * filled + '-' * (20 - filled)
            order = int(self.config.get('gmsh_element_order', 2))
            tet_label = 'Tet10' if order >= 2 else 'Tet4'
            return (
                f'STAGE: Mesh Review\n'
                f'Mesh: {remesh_state}  {pct:5.1f}%\n'
                f'Next remesh: {tet_label}\n'
                f'[{bar}]\n'
                f'{self._remesh_message}\n'
                f'Nodes: {self.nodes.shape[0]:,}\n'
                f'Elements: {self.elements.shape[0]:,}')
        if self._stage == self.STAGE_STRESS:
            if self._stress_running:
                return 'STAGE: Stress Analysis\nRunning static FEA...'
            vm = self._stress_vm_all
            if vm is not None and len(vm) > 0:
                ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
                sf = float(self.config.get('safety_factor', 1.5))
                allowable = self._get_allowable_stress(sf)
                max_vm = float(np.max(vm))
                mean_vm = float(np.mean(vm))
                util = max_vm / max(allowable, 1e-12) * 100.0
                vf_status = f'VF = {self.config.get("volume_fraction", 0.3):.3f}' if self._vf_entered else 'VF = Not set yet (press V)'
                return (
                    f'STAGE: Stress Analysis\n'
                    f'Max VM: {max_vm:.3e} MPa\n'
                    f'Mean VM: {mean_vm:.3e} MPa\n'
                    f'Yield: {ys:.3e} MPa  SF: {sf:.1f}\n'
                    f'Allowable: {allowable:.3e} MPa\n'
                    f'Utilization: {util:.1f}%\n'
                    f'{vf_status}')
            return 'STAGE: Stress Analysis\nNo stress data'
        if self._stage == self.STAGE_OPT:
            return 'STAGE: Optimizing...'
        if self._stage == self.STAGE_POST_STRESS:
            vm = self._post_stress_vm_all
            if vm is not None and len(vm) > 0:
                ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
                sf = float(self.config.get('safety_factor', 1.5))
                allowable = self._get_allowable_stress(sf)
                max_vm = float(np.max(vm))
                util = max_vm / max(allowable, 1e-12) * 100.0
                verdict = 'PASS' if util <= 100.0 else 'FAIL'
                reopt_label = f'  (#{self._reopt_count})' if self._reopt_count > 0 else ''
                return (
                    f'STAGE: Post-Opt Stress{reopt_label}\n'
                    f'Status: {verdict}\n'
                    f'Max VM: {max_vm:.3e} MPa\n'
                    f'Utilization: {util:.1f}%')
            return 'STAGE: Post-Opt Stress\nAnalysing...'
        if self._stage == self.STAGE_RESULTS:
            return 'STAGE: Results'
        return ''

    def _help_text(self):
        if self._stage == self.STAGE_MESH:
            return 'N=Accept & Define BCs  R=Remesh (target tetra count)\nT=Toggle Tet4/Tet10  Q=Quit  F=Fit  1-7=Views'
        if self._stage == self.STAGE_BC:
            return (
                'S=Supports  L=Loads  U=Undo\n'
                'P=Node  G=Face  E=Edge pick\n'
                'Supports lock translations in X,Y,Z\n'
                'X/Y/Z=Load dir  V=Set magnitude  M=Flip sign  +/- = scale\n'
                'D=Set protect distance\n'
                'N=Accept & Stress Analysis  B=Back to Mesh Review\n'
                'Q=Quit  F=Fit  1-7=Views  Click=Pick')
        if self._stage == self.STAGE_STRESS:
            return (
                'STRESS ANALYSIS RESULTS\n'
                'Green = low stress (safe to remove)\n'
                'Red = high stress (keep)\n\n'
                'V=Set volume fraction / mass reduction\n'
                'N=Accept & Start Optimization  B=Back to BCs\n'
                'Q=Quit  F=Fit  1-7=Views')
        if self._stage == self.STAGE_OPT:
            return 'Optimization running...\nGUI stays interactive.'
        if self._stage == self.STAGE_POST_STRESS:
            return (
                'POST-OPTIMIZATION STRESS ANALYSIS\n'
                'Stress heatmap on optimized design\n\n'
                'N=Accept & view final results\n'
                'R=Re-optimize (adjust VF & re-run)\n'
                'Q=Quit  F=Fit  1-7=Views')
        if self._stage == self.STAGE_RESULTS:
            return 'R=Re-optimize  Q=Quit & EXPORT  F=Fit  1-7=Views'
        return ''

    def _add_surface_mesh(self, opacity=0.97, show_edges=True, edge_color=None):
        p = self._plotter
        ec = edge_color or self._COL_EDGE
        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
        faces_to_use = self._get_display_faces()

        try:
            use_smooth = bool(self.elements.shape[0] < 200000)
            if faces_to_use is not None and len(faces_to_use) > 0:
                n_tri = faces_to_use.shape[0]
                faces_pv = np.hstack([
                    np.full((n_tri, 1), 3, dtype=np.int64),
                    faces_to_use.astype(np.int64)
                ]).ravel()
                mesh = pv.PolyData(self.nodes, faces_pv)
            elif n_per in (4, 10):
                n_elem = self.elements.shape[0]
                vtk_tet_type = 24 if n_per == 10 else 10  # VTK_QUADRATIC_TETRA vs VTK_TETRA
                cells = np.hstack([
                    np.full((n_elem, 1), n_per, dtype=np.int64),
                    self.elements.astype(np.int64)
                ]).ravel()
                celltypes = np.full(n_elem, vtk_tet_type, dtype=np.uint8)
                mesh = pv.UnstructuredGrid(cells, celltypes, self.nodes)
            elif n_per == 3:
                n_elem = self.elements.shape[0]
                faces_pv = np.hstack([
                    np.full((n_elem, 1), 3, dtype=np.int64),
                    self.elements.astype(np.int64)
                ]).ravel()
                mesh = pv.PolyData(self.nodes, faces_pv)
            else:
                p.add_points(self.nodes, color=self._COL_SURFACE, point_size=3, pickable=False)
                return

            self._mesh_actor = p.add_mesh(
                mesh,
                color=self._COL_SURFACE,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=ec,
                line_width=1.0,
                ambient=0.25,
                diffuse=0.8,
                specular=0.25,
                specular_power=20,
                smooth_shading=use_smooth,
                pickable=True,
                label='Surface')
        except Exception as e:
            print(f'  [MESH] Surface display error: {e}')
            p.add_points(self.nodes, color=self._COL_SURFACE, point_size=3, pickable=False)

    def _check_tet_quality(self, nodes, elements):
        """Compute basic tetra quality stats (Tet4/Tet10 corners) used before optimization."""
        q = {
            'total': 0,
            'invalid_count': 0,
            'inverted_count': 0,
            'degenerate_count': 0,
            'poor_aspect_count': 0,
            'min_abs_volume': 0.0,
            'median_abs_volume': 0.0,
        }

        try:
            n = np.asarray(nodes, dtype=np.float64)
            t = np.asarray(elements, dtype=np.int32)
            if n.ndim != 2 or n.shape[1] != 3 or t.ndim != 2 or t.shape[1] < 4 or len(t) == 0:
                return q

            q['total'] = int(t.shape[0])

            corners = t[:, :4]
            dup = (
                (corners[:, 0] == corners[:, 1]) | (corners[:, 0] == corners[:, 2]) | (corners[:, 0] == corners[:, 3]) |
                (corners[:, 1] == corners[:, 2]) | (corners[:, 1] == corners[:, 3]) | (corners[:, 2] == corners[:, 3])
            )

            a = n[corners[:, 0]]
            b = n[corners[:, 1]]
            c = n[corners[:, 2]]
            d = n[corners[:, 3]]
            vol6 = np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)

            span = np.maximum(np.max(n, axis=0) - np.min(n, axis=0), 1e-12)
            vol_tol = float(max(np.prod(span) * 1e-14, 1e-18))
            deg = (np.abs(vol6) <= 6.0 * vol_tol) | dup
            inv = (vol6 < -6.0 * vol_tol) & (~deg)

            q['degenerate_count'] = int(np.sum(deg))
            q['inverted_count'] = int(np.sum(inv))
            q['invalid_count'] = int(q['degenerate_count'] + q['inverted_count'])

            abs_vol = np.abs(vol6) / 6.0
            if len(abs_vol) > 0:
                q['min_abs_volume'] = float(np.min(abs_vol))
                q['median_abs_volume'] = float(np.median(abs_vol))

            # Simple shape metric: max edge / min edge ratio
            l2_min = None
            l2_max = None
            l2_sum = np.zeros(q['total'], dtype=np.float64)
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                e = n[corners[:, i]] - n[corners[:, j]]
                l2 = np.einsum('ij,ij->i', e, e)
                l2_sum += l2
                if l2_min is None:
                    l2_min = l2.copy()
                    l2_max = l2.copy()
                else:
                    l2_min = np.minimum(l2_min, l2)
                    l2_max = np.maximum(l2_max, l2)

            with np.errstate(divide='ignore', invalid='ignore'):
                aspect = np.sqrt(l2_max / np.maximum(l2_min, 1e-30))
                # Element Quality Q = C * V / (sum(L_i^2))^1.5
                # For regular tet, C = 72 * sqrt(3) ~= 124.707658
                element_quality = 124.707658 * abs_vol / np.maximum((l2_sum)**1.5, 1e-30)

            poor = aspect > 20.0
            skewness = np.clip(1.0 - element_quality, 0.0, 1.0)
            
            q['poor_aspect_count'] = int(np.sum(poor))
            q['max_aspect'] = float(np.max(aspect)) if len(aspect) > 0 else 0.0
            q['mean_aspect'] = float(np.mean(aspect)) if len(aspect) > 0 else 0.0
            q['min_quality'] = float(np.min(element_quality)) if len(element_quality) > 0 else 0.0
            q['mean_quality'] = float(np.mean(element_quality)) if len(element_quality) > 0 else 0.0
            q['max_skewness'] = float(np.max(skewness)) if len(skewness) > 0 else 0.0
            q['mean_skewness'] = float(np.mean(skewness)) if len(skewness) > 0 else 0.0
            return q
        except Exception:
            return q

    def _auto_repair_tet_mesh(self):
        """Repair inverted/degenerate tetra corner tets in-place when possible."""
        try:
            n = np.asarray(self._original_nodes, dtype=np.float64)
            t = np.asarray(self._original_elements, dtype=np.int32)
            if n.ndim != 2 or n.shape[1] != 3 or t.ndim != 2 or t.shape[1] < 4 or len(t) == 0:
                return None

            corners = t[:, :4]
            dup = (
                (corners[:, 0] == corners[:, 1]) | (corners[:, 0] == corners[:, 2]) | (corners[:, 0] == corners[:, 3]) |
                (corners[:, 1] == corners[:, 2]) | (corners[:, 1] == corners[:, 3]) | (corners[:, 2] == corners[:, 3])
            )

            a = n[corners[:, 0]]
            b = n[corners[:, 1]]
            c = n[corners[:, 2]]
            d = n[corners[:, 3]]
            vol6 = np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)

            span = np.maximum(np.max(n, axis=0) - np.min(n, axis=0), 1e-12)
            vol_tol = float(max(np.prod(span) * 1e-14, 1e-18))
            deg = (np.abs(vol6) <= 6.0 * vol_tol) | dup
            inv = (vol6 < -6.0 * vol_tol) & (~deg)

            t_fix = t.copy()
            if np.any(inv):
                tmp = t_fix[inv, 0].copy()
                t_fix[inv, 0] = t_fix[inv, 1]
                t_fix[inv, 1] = tmp

            keep = ~deg
            t_new = t_fix[keep]
            if len(t_new) == 0:
                return None

            self._original_elements = np.asarray(t_new, dtype=np.int32)
            self.elements = self._original_elements.copy()

            if self._original_nodes.shape[1] == 2:
                z = np.zeros((self._original_nodes.shape[0], 1))
                self.nodes = np.hstack([self._original_nodes, z])
            else:
                self.nodes = np.asarray(self._original_nodes, dtype=np.float64)

            self._boundary_faces_cache = None
            self._surface_edges_cache = None
            self.surf_faces = self._extract_boundary_faces_from_tets(self.elements)

            return {
                'removed': int(np.sum(deg)),
                'reoriented': int(np.sum(inv)),
                'remaining': int(self.elements.shape[0]),
            }
        except Exception:
            return None
    def _build_bc_outputs(self):
        ndim = 2 if self.mode == '2D' else 3
        n_nodes = self._original_nodes.shape[0]

        fixed_dofs = sorted(n * ndim + i for n in self.fixed_nodes for i in range(ndim))
        fixed_dofs = np.array(fixed_dofs, dtype=int) if fixed_dofs else np.array([], dtype=int)

        forces = np.zeros(n_nodes * ndim)
        
        # Area-weighted consistent nodal force distribution for face loads
        if self.mode == '3D' and len(self._load_face_indices) > 0:
            from fea_solver_3d import distribute_surface_traction
            # Use the full topological boundary faces (which may have 6 nodes for Tet10)
            if self._boundary_faces_cache is None:
                self._boundary_faces_cache = self._extract_boundary_faces_from_tets(self.elements)
            topological_faces = self._boundary_faces_cache
            
            if topological_faces is not None:
                load_triangles = topological_faces[list(self._load_face_indices)]
                
                # Determine total force vector from the user's input
                # The sum of magnitudes in self.loads gives the total magnitude intended by the user
                total_f = np.zeros(3)
                for node, load in self.loads.items():
                    total_f[load['dir']] += load['mag']
                
                if np.linalg.norm(total_f) > 1e-12:
                    dist_forces = distribute_surface_traction(self._original_nodes, load_triangles, total_f)
                    forces += dist_forces
        else:
            for node, load in self.loads.items():
                dof = node * ndim + load['dir']
                if dof < len(forces):
                    forces[dof] = load['mag']

        return fixed_dofs, forces
    def _build_passive_solid_mask(self):
        """Create a non-design solid mask around support and load selections."""
        elems = np.asarray(self._original_elements, dtype=np.int64)
        if elems.ndim != 2 or elems.shape[0] == 0:
            return None

        nodes = np.asarray(self._original_nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            return np.zeros(elems.shape[0], dtype=bool)

        n_elem = elems.shape[0]
        corners = elems[:, :min(4, elems.shape[1])]
        centroids = np.mean(nodes[corners], axis=1)
        elem_radius = np.max(np.linalg.norm(nodes[corners] - centroids[:, None, :], axis=2), axis=1)

        def _build_region_mask(region_nodes, region_face_indices, distance_mm, node_layers):
            protected_nodes = set(int(n) for n in region_nodes)
            faces = self._get_display_faces()
            face_indices = set()
            if faces is not None and len(faces) > 0 and region_face_indices:
                face_indices.update(int(i) for i in region_face_indices)

            face_indices = [i for i in face_indices if 0 <= int(i) < (faces.shape[0] if faces is not None else 0)]
            selected_tris = None
            if faces is not None and len(face_indices) > 0:
                selected_tris = np.asarray(faces[np.asarray(face_indices, dtype=np.int64)], dtype=np.int64)
                protected_nodes.update(int(n) for n in selected_tris.reshape(-1).tolist())

            protected_nodes = sorted(int(n) for n in protected_nodes if 0 <= int(n) < nodes.shape[0])
            touch_mask = np.any(np.isin(corners, np.asarray(protected_nodes, dtype=np.int64)), axis=1) if protected_nodes else np.zeros(n_elem, dtype=bool)

            if selected_tris is not None and selected_tris.size > 0:
                face_nodes = np.unique(selected_tris.reshape(-1))
                touch_mask |= np.any(np.isin(corners, face_nodes), axis=1)

            effective_distance = float(distance_mm)

            if (effective_distance <= 0.0) and (selected_tris is not None and selected_tris.size > 0):
                tri_pts = nodes[np.asarray(selected_tris, dtype=np.int64)]
                e01 = np.linalg.norm(tri_pts[:, 1, :] - tri_pts[:, 0, :], axis=1)
                e12 = np.linalg.norm(tri_pts[:, 2, :] - tri_pts[:, 1, :], axis=1)
                e20 = np.linalg.norm(tri_pts[:, 0, :] - tri_pts[:, 2, :], axis=1)
                edge_ref = float(np.median(np.hstack([e01, e12, e20]))) if tri_pts.shape[0] > 0 else 0.0
                if np.isfinite(edge_ref) and edge_ref > 0.0:
                    effective_distance = max(0.35 * edge_ref, 1e-9)

            if selected_tris is not None and selected_tris.size > 0 and effective_distance > 0.0:
                tri_pts = nodes[np.asarray(selected_tris, dtype=np.int64)]
                tri_centers = np.mean(tri_pts, axis=1)
                tri_edge_max = np.max(np.linalg.norm(tri_pts[:, 1, :] - tri_pts[:, 0, :], axis=1)) if tri_pts.shape[0] > 0 else 0.0
                search_pad = float(effective_distance + tri_edge_max)

                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(tri_centers)
                except Exception:
                    tree = None

                def _pt_tri_dist2(p, a, b, c):
                    ab = b - a
                    ac = c - a
                    ap = p - a
                    d1 = float(np.dot(ab, ap))
                    d2 = float(np.dot(ac, ap))
                    if d1 <= 0.0 and d2 <= 0.0:
                        return float(np.dot(ap, ap))
                    bp = p - b
                    d3 = float(np.dot(ab, bp))
                    d4 = float(np.dot(ac, bp))
                    if d3 >= 0.0 and d4 <= d3:
                        return float(np.dot(bp, bp))
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        v = d1 / (d1 - d3)
                        proj = a + v * ab
                        dp = p - proj
                        return float(np.dot(dp, dp))
                    cp = p - c
                    d5 = float(np.dot(ab, cp))
                    d6 = float(np.dot(ac, cp))
                    if d6 >= 0.0 and d5 <= d6:
                        return float(np.dot(cp, cp))
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        proj = a + w * ac
                        dp = p - proj
                        return float(np.dot(dp, dp))
                    va = d3 * d6 - d5 * d4
                    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                        proj = b + w * (c - b)
                        dp = p - proj
                        return float(np.dot(dp, dp))
                    denom = 1.0 / max(va + vb + vc, 1e-12)
                    v = vb * denom
                    w = vc * denom
                    proj = a + ab * v + ac * w
                    dp = p - proj
                    return float(np.dot(dp, dp))

                mask = np.zeros(n_elem, dtype=bool)
                for i in range(n_elem):
                    nodes_i = nodes[corners[i]]
                    hit = False
                    for p in nodes_i:
                        if tree is not None:
                            cand = tree.query_ball_point(p, effective_distance + search_pad)
                        else:
                            diff = tri_centers - p.reshape(1, 3)
                            cand = np.where(np.sum(diff * diff, axis=1) <= (effective_distance + search_pad) ** 2)[0].tolist()
                        for j in cand:
                            a, b, c = tri_pts[int(j)]
                            if _pt_tri_dist2(p, a, b, c) <= (effective_distance ** 2):
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        mask[i] = True

                try:
                    edge_ref = None
                    if tri_pts.shape[0] > 0:
                        e01 = np.linalg.norm(tri_pts[:, 1, :] - tri_pts[:, 0, :], axis=1)
                        e12 = np.linalg.norm(tri_pts[:, 2, :] - tri_pts[:, 1, :], axis=1)
                        e20 = np.linalg.norm(tri_pts[:, 0, :] - tri_pts[:, 2, :], axis=1)
                        edge_ref = float(np.median(np.hstack([e01, e12, e20])))
                    if edge_ref is not None and edge_ref > 1e-12:
                        layers = int(max(1, np.ceil(effective_distance / edge_ref)))
                        n_nodes_total = int(nodes.shape[0])
                        node_to_elems = [[] for _ in range(n_nodes_total)]
                        for eidx in range(n_elem):
                            for nid in elems[eidx]:
                                ni = int(nid)
                                if 0 <= ni < n_nodes_total:
                                    node_to_elems[ni].append(eidx)
                        seed_elems = set()
                        seed_nodes = set(int(n) for n in selected_tris.reshape(-1).tolist())
                        for nid in seed_nodes:
                            if 0 <= nid < n_nodes_total:
                                seed_elems.update(node_to_elems[nid])
                        expanded = set(seed_elems)
                        frontier = set(seed_elems)
                        for _ in range(layers):
                            next_frontier = set()
                            for eidx in frontier:
                                for nid in elems[int(eidx)]:
                                    for j in node_to_elems[int(nid)]:
                                        if j not in expanded:
                                            next_frontier.add(j)
                            expanded.update(next_frontier)
                            frontier = next_frontier
                        if expanded:
                            mask[np.fromiter(expanded, dtype=np.int64)] = True
                except Exception:
                    pass

                mask |= touch_mask
                return mask

            if effective_distance > 0.0 and protected_nodes:
                protected_pts = nodes[np.asarray(protected_nodes, dtype=np.int64)]
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(protected_pts)
                    try:
                        d_cent, _ = tree.query(centroids, k=1, workers=-1)
                    except TypeError:
                        d_cent, _ = tree.query(centroids, k=1)
                    d_cent = np.asarray(d_cent, dtype=np.float64)
                    mask = d_cent <= (effective_distance + elem_radius)
                except Exception:
                    diff_c = centroids[:, None, :] - protected_pts[None, :, :]
                    d2 = np.sum(diff_c * diff_c, axis=2)
                    d_cent = np.sqrt(np.min(d2, axis=1))
                    mask = d_cent <= (effective_distance + elem_radius)
                mask |= touch_mask
                return mask

            n_nodes_total = int(nodes.shape[0])
            node_to_elems = [[] for _ in range(n_nodes_total)]
            for eidx in range(n_elem):
                for nid in elems[eidx]:
                    ni = int(nid)
                    if 0 <= ni < n_nodes_total:
                        node_to_elems[ni].append(eidx)

            protected_nodes_curr = set(int(n) for n in protected_nodes)
            protected_elems = set()
            for _ in range(node_layers):
                frontier_elems = set()
                for nid in protected_nodes_curr:
                    frontier_elems.update(node_to_elems[nid])
                protected_elems.update(frontier_elems)
                next_nodes = set(protected_nodes_curr)
                for eidx in frontier_elems:
                    for nid in elems[int(eidx)]:
                        ni = int(nid)
                        if 0 <= ni < n_nodes_total:
                            next_nodes.add(ni)
                protected_nodes_curr = next_nodes

            mask = np.zeros(n_elem, dtype=bool)
            if protected_elems:
                mask[np.fromiter(protected_elems, dtype=np.int64)] = True
            return mask

        support_dist = float(self.config.get('support_non_design_distance_mm', self.config.get('non_design_distance_mm', 0.0)))
        support_layers = int(max(1, self.config.get('support_non_design_node_layers', self.config.get('non_design_node_layers', 1))))
        support_mask = _build_region_mask(self.fixed_nodes, self._support_face_indices, support_dist, support_layers)
        
        force_dist = float(self.config.get('force_non_design_distance_mm', self.config.get('non_design_distance_mm', 0.0)))
        force_layers = int(max(1, self.config.get('force_non_design_node_layers', self.config.get('non_design_node_layers', 1))))
        force_mask = _build_region_mask(self.loads.keys(), self._load_face_indices, force_dist, force_layers)
        
        return support_mask | force_mask

    def _enforce_manufacturable_result(self, rho):
        """Keep only load-path-connected solid material and remove floating islands."""
        rr = np.asarray(rho, dtype=np.float64).copy()
        elems = np.asarray(self._original_elements, dtype=np.int64)
        if elems.ndim != 2 or elems.shape[0] == 0:
            return rr

        solid = rr >= 0.5
        if not np.any(solid):
            return rr

        n_elem = elems.shape[0]
        n_nodes = int(np.asarray(self._original_nodes).shape[0])
        node_to_elems = [[] for _ in range(n_nodes)]
        for eidx in range(n_elem):
            if not solid[eidx]:
                continue
            for nid in elems[eidx]:
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    node_to_elems[ni].append(eidx)

        seeds = set()
        if self._passive_solid_mask is not None and len(self._passive_solid_mask) == n_elem:
            idx = np.where(solid & self._passive_solid_mask)[0]
            seeds.update(int(i) for i in idx.tolist())

        if not seeds:
            for nid in self.fixed_nodes:
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    seeds.update(node_to_elems[ni])
            for nid in self.loads.keys():
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    seeds.update(node_to_elems[ni])

        if not seeds:
            # Fallback: keep largest connected component.
            visited = np.zeros(n_elem, dtype=bool)
            best_comp = []
            for i in np.where(solid)[0].tolist():
                if visited[i]:
                    continue
                comp = []
                stack = [int(i)]
                visited[i] = True
                while stack:
                    e = stack.pop()
                    comp.append(e)
                    for nid in elems[e]:
                        for j in node_to_elems[int(nid)]:
                            if not visited[j]:
                                visited[j] = True
                                stack.append(j)
                if len(comp) > len(best_comp):
                    best_comp = comp
            keep = np.zeros(n_elem, dtype=bool)
            if best_comp:
                keep[np.asarray(best_comp, dtype=np.int64)] = True
        else:
            keep = np.zeros(n_elem, dtype=bool)
            stack = list(seeds)
            for s in seeds:
                keep[int(s)] = True
            while stack:
                e = int(stack.pop())
                for nid in elems[e]:
                    for j in node_to_elems[int(nid)]:
                        if solid[j] and not keep[j]:
                            keep[j] = True
                            stack.append(int(j))

        removed = int(np.sum(solid & (~keep)))
        if removed > 0:
            rr[solid & (~keep)] = 0.0
            print(f'  [POST] Removed disconnected solid islands: {removed} elements')

        rr = self._apply_overhang_filter(rr)

        if self._passive_solid_mask is not None and len(self._passive_solid_mask) == n_elem:
            rr[self._passive_solid_mask] = 1.0

        return rr



    def _apply_overhang_filter(self, rho):
        rr = np.asarray(rho, dtype=np.float64).copy()
        if not bool(self.config.get('enable_overhang_filter', False)):
            return rr

        elems = np.asarray(self._original_elements, dtype=np.int64)
        if elems.ndim != 2 or elems.shape[0] != rr.size:
            return rr

        axis_name = str(self.config.get('build_axis', 'z')).lower()
        axis = {'x': 0, 'y': 1, 'z': 2}.get(axis_name, 2)
        angle_deg = float(self.config.get('overhang_angle_deg', 45.0))
        tan_crit = np.tan(np.deg2rad(max(1.0, min(angle_deg, 89.0))))

        c = np.mean(np.asarray(self._original_nodes)[elems], axis=1)
        # Estimate filter radius from config or mesh bounding box (self.filter_radius doesn't exist on GUI)
        try:
            from density_filter_3d import auto_filter_radius_from_mesh
            _fr, _ = auto_filter_radius_from_mesh(self._original_nodes, self._original_elements,
                                                   factor=float(self.config.get('filter_radius_factor', 3.5)))
        except Exception:
            bbox = np.max(self._original_nodes, axis=0) - np.min(self._original_nodes, axis=0)
            _fr = float(np.mean(bbox)) * 0.05
        dz = np.maximum(1e-9, _fr)
        order = np.argsort(c[:, axis])

        rr_out = rr.copy()
        for idx in order.tolist():
            if rr[idx] < 0.5:
                continue
            ci = c[idx]
            d = ci - c
            vert = d[:, axis]
            horiz = np.linalg.norm(np.delete(d, axis, axis=1), axis=1)
            support = (vert > 0.0) & (vert <= dz) & (horiz <= tan_crit * np.maximum(vert, 1e-12)) & (rr_out >= 0.5)
            if not np.any(support):
                rr_out[idx] = 0.75 * rr_out[idx]

        if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(rr_out):
            rr_out[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0
        return rr_out

    def _check_enclosed_voids(self, rho):
        rr = np.asarray(rho, dtype=np.float64)
        elems = np.asarray(self._original_elements, dtype=np.int64)
        if elems.ndim != 2 or elems.shape[0] != rr.size:
            return None

        void = rr < 0.5
        if not np.any(void):
            return {'count': 0, 'void_elements': 0}

        n_nodes = int(np.asarray(self._original_nodes).shape[0])
        node_to_void = [[] for _ in range(n_nodes)]
        for eidx in np.where(void)[0].tolist():
            for nid in elems[eidx]:
                ni = int(nid)
                if 0 <= ni < n_nodes:
                    node_to_void[ni].append(int(eidx))

        all_nodes = np.asarray(self._original_nodes, dtype=np.float64)
        mins = np.min(all_nodes, axis=0)
        maxs = np.max(all_nodes, axis=0)
        spans = np.maximum(maxs - mins, 1e-12)
        node_on_boundary = np.any((np.abs(all_nodes - mins) <= 1e-6 * spans) | (np.abs(all_nodes - maxs) <= 1e-6 * spans), axis=1)

        visited = np.zeros(rr.size, dtype=bool)
        enclosed_count = 0
        enclosed_void_elems = 0

        for start in np.where(void)[0].tolist():
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            comp = []
            touches_boundary = False
            while stack:
                e = stack.pop()
                comp.append(e)
                if np.any(node_on_boundary[elems[e]]):
                    touches_boundary = True
                for nid in elems[e]:
                    for j in node_to_void[int(nid)]:
                        if not visited[j]:
                            visited[j] = True
                            stack.append(int(j))
            if not touches_boundary:
                enclosed_count += 1
                enclosed_void_elems += len(comp)

        return {'count': int(enclosed_count), 'void_elements': int(enclosed_void_elems)}

    def _regularize_final_density(self, rho):
        """Smooth final element-density field to reduce jagged cavity artifacts before binarization."""
        rr = np.asarray(rho, dtype=np.float64).copy()
        elems = np.asarray(self._original_elements, dtype=np.int64)
        if rr.ndim != 1 or elems.ndim != 2 or elems.shape[0] != rr.size:
            return rr

        passes = int(max(0, self.config.get('final_density_smoothing_passes', 2)))
        blend = float(np.clip(self.config.get('final_density_smoothing_blend', 0.60), 0.0, 1.0))
        if passes <= 0:
            return rr

        n_nodes = int(np.asarray(self._original_nodes).shape[0])
        if n_nodes <= 0:
            return rr

        n_per = int(elems.shape[1])
        rho_floor = float(np.clip(self.config.get('rho_min', 1e-6), 1e-9, 0.2))

        for _ in range(passes):
            node_sum = np.zeros(n_nodes, dtype=np.float64)
            node_cnt = np.zeros(n_nodes, dtype=np.float64)

            flat_nodes = elems.reshape(-1)
            flat_vals = np.repeat(rr, n_per)
            np.add.at(node_sum, flat_nodes, flat_vals)
            np.add.at(node_cnt, flat_nodes, 1.0)

            node_avg = node_sum / np.maximum(node_cnt, 1.0)
            elem_avg = np.mean(node_avg[elems], axis=1)
            rr = blend * rr + (1.0 - blend) * elem_avg
            rr = np.clip(rr, rho_floor, 1.0)

            if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(rr):
                rr[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0

        return rr

    def _finalize_topology_by_target_volume(self, rho):
        """Create a usable binary layout at target volume while preserving protected regions."""
        rr = np.asarray(rho, dtype=np.float64).copy()
        n = rr.size
        if n == 0:
            return rr

        target_vf = float(self.config.get('volume_fraction', 0.3))
        target_vf = float(np.clip(target_vf, 0.05, 1.0))

        passive = np.zeros(n, dtype=bool)
        if self._passive_solid_mask is not None and len(self._passive_solid_mask) == n:
            passive = np.asarray(self._passive_solid_mask, dtype=bool)

        min_keep = int(np.sum(passive))
        n_keep = int(np.clip(round(target_vf * n), 1, n))
        n_keep = max(n_keep, min_keep)

        order = np.argsort(rr)[::-1]
        out = np.zeros(n, dtype=np.float64)
        out[order[:n_keep]] = 1.0
        if np.any(passive):
            out[passive] = 1.0
        return out

    def _verify_non_design_enforcement(self, rho):
        if self._passive_solid_mask is None:
            return None
        mask = np.asarray(self._passive_solid_mask, dtype=bool)
        if mask.size == 0 or not np.any(mask) or len(mask) != len(rho):
            return None
        vals = np.asarray(rho, dtype=np.float64)[mask]
        return {
            'count': int(vals.size),
            'kept_099': int(np.sum(vals > 0.99)),
            'kept_095': int(np.sum(vals > 0.95)),
            'min': float(np.min(vals)) if vals.size else 0.0,
            'mean': float(np.mean(vals)) if vals.size else 0.0,
        }

    def _run_binary_verification_fea(self):
        """Run one verification FEA on final binary topology and report stress margin."""
        try:
            from fea_solver_3d import FEASolver3D
        except Exception:
            return None

        if self.rho_result is None:
            return None

        try:
            fixed_dofs, forces = self._build_bc_outputs()
            thr = self._compute_display_cutoff(self.rho_result)
            rho_min = float(self.config.get('rho_min', 1e-6))
            rho_bin = np.where(np.asarray(self.rho_result, dtype=np.float64) >= thr, 1.0, rho_min)
            if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(rho_bin):
                rho_bin[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0

            pen = float(self.config.get('penalization', self.config.get('penalty', 3.0)))
            fea = FEASolver3D(self._original_nodes, self._original_elements, self.material)
            u = fea.solve(rho_bin, forces, fixed_dofs, pen)

            uu = np.asarray(u, dtype=np.float64).reshape(-1, 3)
            disp = np.linalg.norm(uu, axis=1)
            max_disp = float(np.max(disp)) if disp.size else 0.0

            vm_all = fea.calculate_stress_all_gauss(u, rho_bin, pen)
            max_vm = float(np.max(vm_all)) if vm_all.size else 0.0

            ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
            sf = float(self.config.get('safety_factor', 1.5))
            allowable = self._get_allowable_stress(sf)
            sf_margin = allowable / max(max_vm, 1e-12)

            fatigue_limit = getattr(self.material, 'fatigue_limit', self.config.get('fatigue_limit', None))
            fatigue_usage = None
            fatigue_ok = None
            if fatigue_limit is not None:
                fl = float(fatigue_limit)
                if fl > 0.0:
                    alt_vm = 0.5 * max_vm
                    fatigue_usage = alt_vm / fl
                    fatigue_ok = bool(fatigue_usage <= 1.0)

            fracture_toughness = getattr(self.material, 'fracture_toughness', self.config.get('fracture_toughness', None))
            a_min_m = None
            if fracture_toughness is not None:
                kic = float(fracture_toughness)
                sigma_mpa = max_vm
                if kic > 0.0 and sigma_mpa > 1e-12:
                    a_min_m = ((kic / sigma_mpa) ** 2) / np.pi

            return {
                'max_disp': max_disp,
                'max_vm': max_vm,
                'allowable': allowable,
                'sf_margin': sf_margin,
                'fatigue_usage': fatigue_usage,
                'fatigue_ok': fatigue_ok,
                'a_min_m': a_min_m,
            }
        except Exception:
            return None



    def _get_allowable_stress(self, safety_factor=None):
        sf = float(safety_factor if safety_factor is not None else self.config.get('safety_factor', 1.5))
        if self.material is not None and hasattr(self.material, 'get_allowable_stress'):
            try:
                return float(self.material.get_allowable_stress(sf))
            except Exception:
                pass
        ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
        return ys / max(sf, 1e-12)

    def _compute_mass_report(self, rho):
        """Compute mass report assuming mm native units.
        
        Density is expected in kg/mm³.
        Mass = Volume(mm³) * Density(kg/mm³) = kg
        """
        try:
            rr = np.asarray(rho, dtype=np.float64)
            elems = np.asarray(self._original_elements, dtype=np.int64)
            nodes = np.asarray(self._original_nodes, dtype=np.float64)
            if rr.ndim != 1 or elems.ndim != 2 or elems.shape[0] != rr.size or elems.shape[1] < 4:
                return None

            corners = elems[:, :4]
            a = nodes[corners[:, 0]]
            b = nodes[corners[:, 1]]
            c = nodes[corners[:, 2]]
            d = nodes[corners[:, 3]]
            vol = np.abs(np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)) / 6.0

            mat_rho = float(getattr(self.material, 'density', self.config.get('material_density', 7.8e-6)))
            mass_baseline = float(np.sum(vol) * mat_rho)
            mass_current = float(np.sum(vol * np.clip(rr, 0.0, 1.0)) * mat_rho)
            saved_pct = 100.0 * max(0.0, mass_baseline - mass_current) / max(mass_baseline, 1e-12)

            return {
                'mass_current': mass_current,
                'mass_baseline': mass_baseline,
                'saved_pct': saved_pct,
            }
        except Exception:
            return None

    def _save_compliance_history_plot(self):
        hist = [float(v) for v in (self.compliance_history or []) if np.isfinite(v)]
        if len(hist) < 2:
            return None

        try:
            import matplotlib.pyplot as plt
            out = Path('compliance_history.png').resolve()
            fig = plt.figure(figsize=(8, 4.2))
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(hist) + 1), hist, color='#1f77b4', linewidth=2.0)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Compliance')
            ax.set_title('Compliance History')
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            return str(out)
        except Exception:
            return None

    # --- Restored stage/remesh/render handlers ---
    def _get_display_faces(self):
        if self.surf_faces is not None:
            sf = np.asarray(self.surf_faces, dtype=np.int32)
            if sf.ndim == 2 and sf.shape[1] == 3 and len(sf) > 0 and sf.max() < self.nodes.shape[0]:
                return sf

        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
        if n_per == 3:
            return np.asarray(self.elements, dtype=np.int32)
        if n_per in (4, 10):
            if self._boundary_faces_cache is None:
                self._boundary_faces_cache = self._extract_boundary_faces_from_tets(self.elements)
            return self._boundary_faces_cache[:, :3]
        return None

    def _get_surface_edges(self):
        if self.edge_segments is not None:
            es = np.asarray(self.edge_segments, dtype=np.int32)
            if es.ndim == 2 and es.shape[1] == 2 and len(es) > 0 and es.max() < self.nodes.shape[0]:
                return es
        if self._surface_edges_cache is None:
            faces = self._get_display_faces()
            if faces is None or len(faces) == 0:
                return None
            edges = np.sort(np.vstack([
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]), axis=1)
            self._surface_edges_cache = np.unique(edges, axis=0).astype(np.int32)
        return self._surface_edges_cache

    @staticmethod
    def _extract_boundary_faces_from_tets(tets):
        t = np.asarray(tets, dtype=np.int32)
        if t.ndim != 2 or t.shape[1] < 4 or len(t) == 0:
            return None
        corners = t[:, :4]
        all_corners = np.vstack([
            corners[:, [0, 1, 2]],
            corners[:, [0, 1, 3]],
            corners[:, [0, 2, 3]],
            corners[:, [1, 2, 3]],
        ])
        
        if t.shape[1] == 10:
            # Full 6-node topological faces
            all_faces = np.vstack([
                t[:, [0, 1, 2, 4, 5, 6]],
                t[:, [0, 1, 3, 4, 8, 7]],
                t[:, [0, 2, 3, 6, 9, 7]],
                t[:, [1, 2, 3, 5, 9, 8]],
            ])
        else:
            all_faces = all_corners
            
        sorted_corners = np.sort(all_corners, axis=1)
        uniq, indices, counts = np.unique(sorted_corners, axis=0, return_index=True, return_counts=True)
        return all_faces[indices[counts == 1]].astype(np.int32)

    @staticmethod
    def _sample_points_uniform(points, n_samples):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return pts
        k = int(max(1, n_samples))
        n = pts.shape[0]
        if n <= k:
            rep = int(np.ceil(k / float(max(n, 1))))
            return np.tile(pts, (rep, 1))[:k]
        sel = [int(np.argmax(np.sum((pts - np.mean(pts, axis=0)) ** 2, axis=1)))]
        d2 = np.sum((pts - pts[sel[0]]) ** 2, axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d2))
            sel.append(nxt)
            d2 = np.minimum(d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
        return pts[np.asarray(sel, dtype=np.int64)]

    def _sync_face_region_sets(self):
        faces = self._get_display_faces()
        if faces is None or len(faces) == 0:
            self._support_face_indices = set()
            self._load_face_indices = set()
            return
        max_i = int(faces.shape[0])
        self._support_face_indices = set(i for i in self._support_face_indices if 0 <= int(i) < max_i)
        self._load_face_indices = set(i for i in self._load_face_indices if 0 <= int(i) < max_i)

    def _capture_face_region_centers(self, faces, face_indices, nodes):
        if faces is None or len(faces) == 0 or face_indices is None or len(face_indices) == 0:
            return None
        idx = np.array(sorted(int(i) for i in face_indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < faces.shape[0])]
        if idx.size == 0:
            return None
        tri = np.asarray(faces[idx], dtype=np.int64)
        return np.mean(np.asarray(nodes, dtype=np.float64)[tri], axis=1)

    def _map_face_region_centers_to_indices(self, faces, nodes, region_centers):
        if faces is None or len(faces) == 0 or region_centers is None:
            return set()
        rc = np.asarray(region_centers, dtype=np.float64)
        if rc.ndim != 2 or rc.shape[0] == 0:
            return set()
        tri = np.asarray(faces, dtype=np.int64)
        ctr = np.mean(np.asarray(nodes, dtype=np.float64)[tri], axis=1)
        if ctr.shape[0] == 0:
            return set()
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(ctr)
            try:
                _, nn = tree.query(rc, k=1, workers=-1)
            except TypeError:
                _, nn = tree.query(rc, k=1)
            return set(int(i) for i in np.asarray(nn, dtype=np.int64).reshape(-1).tolist())
        except Exception:
            out = set()
            for p in rc:
                j = int(np.argmin(np.sum((ctr - p.reshape(1, 3)) ** 2, axis=1)))
                out.add(j)
            return out
    def _redraw_bc_indicators(self):
        p = self._plotter
        if p is None:
            return
        for a in self._indicator_actors:
            try:
                p.remove_actor(a)
            except Exception:
                pass
        self._indicator_actors = []
        faces = self._get_display_faces()
        have_faces = faces is not None and len(faces) > 0
        if have_faces and len(self._support_face_indices) > 0:
            try:
                sidx = np.array(sorted(int(i) for i in self._support_face_indices), dtype=np.int64)
                sidx = sidx[(sidx >= 0) & (sidx < faces.shape[0])]
                if sidx.size > 0:
                    tri = np.asarray(faces[sidx], dtype=np.int64)
                    faces_pv = np.hstack([
                        np.full((tri.shape[0], 1), 3, dtype=np.int64),
                        tri
                    ]).ravel()
                    pd = pv.PolyData(self.nodes, faces_pv)
                    a = p.add_mesh(pd, color=self._COL_FIXED, opacity=0.32, show_edges=False, pickable=False)
                    self._indicator_actors.append(a)
            except Exception:
                pass
        if have_faces and len(self._load_face_indices) > 0:
            try:
                lidx = np.array(sorted(int(i) for i in self._load_face_indices), dtype=np.int64)
                lidx = lidx[(lidx >= 0) & (lidx < faces.shape[0])]
                if lidx.size > 0:
                    tri = np.asarray(faces[lidx], dtype=np.int64)
                    faces_pv = np.hstack([
                        np.full((tri.shape[0], 1), 3, dtype=np.int64),
                        tri
                    ]).ravel()
                    pd = pv.PolyData(self.nodes, faces_pv)
                    a = p.add_mesh(pd, color=self._COL_LOAD, opacity=0.24, show_edges=False, pickable=False)
                    self._indicator_actors.append(a)
            except Exception:
                pass
        if len(self.fixed_nodes) > 0:
            idx = np.asarray(sorted(int(i) for i in self.fixed_nodes if 0 <= int(i) < self.nodes.shape[0]), dtype=np.int64)
            if idx.size > 0:
                pd = pv.PolyData(self.nodes[idx])
                a = p.add_mesh(pd, color=self._COL_FIXED, point_size=10, render_points_as_spheres=True, pickable=False)
                self._indicator_actors.append(a)
        # --- Arrow indicators (reduced count for performance) ---
        _MAX_ARROWS = 10
        model_c = np.mean(self.nodes, axis=0)

        def _orient_outward(centers, normals):
            c = np.asarray(centers, dtype=np.float64)
            n = np.asarray(normals, dtype=np.float64)
            if c.ndim != 2 or n.ndim != 2 or c.shape[0] == 0:
                return n
            ref = c - model_c.reshape(1, 3)
            flip = np.sum(ref * n, axis=1) < 0.0
            if np.any(flip):
                n[flip] *= -1.0
            ln = np.linalg.norm(n, axis=1)
            good = ln > 1e-12
            if np.any(good):
                n[good] = n[good] / ln[good][:, None]
            return n

        def _sample_centers(centers, max_n):
            if centers.shape[0] <= max_n:
                return centers
            return self._sample_points_uniform(centers, max_n)

        # Pre-compute arrow length (used by both support triads and load arrows)
        arrow_len = self._arrow_scale * 2.5

        # ---- SUPPORT TRIADS (3 cones at each point for X/Y/Z block) ----
        sup_pts = None
        sup_nrm = None
        if have_faces and len(self._support_face_indices) > 0:
            try:
                sidx = np.array(sorted(int(i) for i in self._support_face_indices), dtype=np.int64)
                sidx = sidx[(sidx >= 0) & (sidx < faces.shape[0])]
                if sidx.size > 0:
                    tri = faces[sidx]
                    pts = self.nodes[tri]
                    centers = np.mean(pts, axis=1)
                    nrm = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
                    nrm = _orient_outward(centers, nrm)
                    sup_pts = _sample_centers(centers, _MAX_ARROWS)
                    # match normals to sampled points
                    if sup_pts.shape[0] < centers.shape[0]:
                        sel = [int(np.argmin(np.sum((centers - sp) ** 2, axis=1))) for sp in sup_pts]
                        sup_nrm = nrm[np.asarray(sel)]
                    else:
                        sup_nrm = nrm
            except Exception:
                pass
        if sup_pts is None and len(self.fixed_nodes) > 0:
            idx = np.asarray(sorted(int(i) for i in self.fixed_nodes if 0 <= int(i) < self.nodes.shape[0]), dtype=np.int64)
            if idx.size > 0:
                pts = self.nodes[idx]
                sup_pts = _sample_centers(pts, _MAX_ARROWS)
                ref = sup_pts - model_c.reshape(1, 3)
                ln = np.linalg.norm(ref, axis=1, keepdims=True)
                sup_nrm = np.where(ln > 1e-12, ref / ln, np.array([[1, 0, 0]], dtype=np.float64))

        if sup_pts is not None and len(sup_pts) > 0:
            cone_h = self._ind_size * 4.0
            cone_r = self._ind_size * 1.5
            offset = self._ind_size * 1.8 + arrow_len * 0.2
            axis_dirs = [
                np.array([1, 0, 0], dtype=np.float64),
                np.array([0, 1, 0], dtype=np.float64),
                np.array([0, 0, 1], dtype=np.float64),
            ]
            for k in range(len(sup_pts)):
                pt = sup_pts[k]
                nk = sup_nrm[k] if sup_nrm is not None and k < len(sup_nrm) else np.array([0, 0, 1.0])
                base = pt + nk * offset
                for ad in axis_dirs:
                    try:
                        tip = base + ad * cone_h
                        cc = base + ad * cone_h * 0.5
                        cone = pv.Cone(center=cc, direction=ad, height=cone_h, radius=cone_r, resolution=12)
                        a = p.add_mesh(cone, color=self._COL_FIXED, opacity=0.9, pickable=False, lighting=False)
                        self._indicator_actors.append(a)
                    except Exception:
                        continue

        # ---- LOAD ARROWS (single arrow per point in correct direction) ----
        load_dir = int(np.clip(self._load_direction, 0, 2))
        load_sign = np.sign(self._load_magnitude) if abs(self._load_magnitude) > 1e-12 else -1.0
        if len(self.loads) > 0:
            dirs_list = [int(v.get('dir', load_dir)) for v in self.loads.values()]
            mags_list = [float(v.get('mag', self._load_magnitude)) for v in self.loads.values()]
            load_dir = int(np.bincount(np.clip(dirs_list, 0, 2), minlength=3).argmax())
            total_mag = sum(mags_list)
            load_sign = np.sign(total_mag) if abs(total_mag) > 1e-12 else -1.0

        ld_pts = None
        ld_nrm = None
        if have_faces and len(self._load_face_indices) > 0:
            try:
                lidx = np.array(sorted(int(i) for i in self._load_face_indices), dtype=np.int64)
                lidx = lidx[(lidx >= 0) & (lidx < faces.shape[0])]
                if lidx.size > 0:
                    tri = faces[lidx]
                    pts = self.nodes[tri]
                    centers = np.mean(pts, axis=1)
                    nrm = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
                    nrm = _orient_outward(centers, nrm)
                    ld_pts = _sample_centers(centers, _MAX_ARROWS)
                    if ld_pts.shape[0] < centers.shape[0]:
                        sel = [int(np.argmin(np.sum((centers - sp) ** 2, axis=1))) for sp in ld_pts]
                        ld_nrm = nrm[np.asarray(sel)]
                    else:
                        ld_nrm = nrm
            except Exception:
                pass
        if ld_pts is None and len(self.loads) > 0:
            s = []
            for nid in self.loads:
                ni = int(nid)
                if 0 <= ni < self.nodes.shape[0]:
                    s.append(self.nodes[ni])
            if s:
                pts = np.array(s, dtype=np.float64)
                ld_pts = _sample_centers(pts, _MAX_ARROWS)
                ref = ld_pts - model_c.reshape(1, 3)
                ln = np.linalg.norm(ref, axis=1, keepdims=True)
                ld_nrm = np.where(ln > 1e-12, ref / ln, np.array([[1, 0, 0]], dtype=np.float64))

        if ld_pts is not None and len(ld_pts) > 0:
            arrow_len = self._arrow_scale * 2.5
            tip_len = arrow_len * 0.3
            shaft_len = arrow_len - tip_len
            d_vec = np.zeros(3, dtype=np.float64)
            d_vec[load_dir] = load_sign
            padding = self._ind_size * 0.5 + arrow_len * 0.6
            for k in range(len(ld_pts)):
                pt = ld_pts[k]
                nk = ld_nrm[k] if ld_nrm is not None and k < len(ld_nrm) else np.array([0, 0, 1.0])
                center = pt + nk * padding
                base = center - d_vec * (arrow_len * 0.5)
                try:
                    shaft_end = base + d_vec * shaft_len
                    line = pv.Line(base, shaft_end, resolution=1)
                    tube = line.tube(radius=max(self._ind_size * 0.22, 1e-6), n_sides=10)
                    a1 = p.add_mesh(tube, color=self._COL_LOAD, opacity=1.0, pickable=False, lighting=False)
                    self._indicator_actors.append(a1)
                    cone_c = shaft_end + d_vec * tip_len * 0.5
                    cone = pv.Cone(center=cone_c, direction=d_vec, height=tip_len, radius=max(self._ind_size * 0.55, 1e-6), resolution=12)
                    a2 = p.add_mesh(cone, color=self._COL_LOAD, opacity=1.0, pickable=False, lighting=False)
                    self._indicator_actors.append(a2)
                except Exception:
                    continue

        if self._stage == self.STAGE_BC:
            self._request_non_design_mask_update()

    def _request_non_design_mask_update(self):
        def worker():
            mask = self._build_passive_solid_mask()
            self._input_queue.put(('new_mask', mask))
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _draw_non_design_mask(self, mask=None):
        if self._plotter is None:
            return
        try:
            if self._mask_actor is not None:
                self._plotter.remove_actor(self._mask_actor)
        except Exception:
            pass
        self._mask_actor = None

        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
        if n_per not in (4, 10):
            return

        if mask is None:
            mask = self._build_passive_solid_mask()
        self._passive_solid_mask = mask.copy() if mask is not None else None
        if mask is None or not np.any(mask):
            return

        n_elem = self.elements.shape[0]
        cells = np.hstack([
            np.full((n_elem, 1), n_per, dtype=np.int64),
            self.elements.astype(np.int64)
        ]).ravel()
        vtk_tet_type = 24 if n_per == 10 else 10
        celltypes = np.full(n_elem, vtk_tet_type, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
        grid['mask'] = np.asarray(mask, dtype=np.float64)
        try:
            visible = grid.threshold(value=0.5, scalars='mask')
        except Exception:
            visible = grid

        self._mask_actor = self._plotter.add_mesh(
            visible,
            color='#ff884d',
            opacity=0.35,
            show_edges=False,
            pickable=False,
            lighting=False,
        )

    def _setup_mesh_stage(self):
        try:
            self._plotter.disable_picking()
        except Exception:
            pass
        self._add_surface_mesh(opacity=1.0, show_edges=True, edge_color=self._COL_MESH_EDGE)
        self._draw_non_design_mask()

        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
        quality = 'GOOD (volumetric)' if n_per in (4, 10) else 'LIMITED (surface)' if n_per == 3 else 'Unknown'
        tet_type = 'Tet10' if n_per == 10 else 'Tet4' if n_per == 4 else 'Tri3' if n_per == 3 else str(n_per)
        info = (
            f'MESH REVIEW\n'
            f'Nodes: {self.nodes.shape[0]:,}  Elements: {self.elements.shape[0]:,}\n'
            f'Type: {tet_type}  Quality: {quality}\n'
            f'Domain size: {float(np.max(np.ptp(self.nodes, axis=0))):.2f}'
        )
        self._info_actor = self._plotter.add_text(info, position='upper_right', font_size=12, color='white')

        if n_per in (4, 10):
            q = self._check_tet_quality(self._original_nodes, self._original_elements)
            q_status = 'PASS' if q['invalid_count'] == 0 else 'FAIL'
            q_invalid = q['invalid_count']
            q_inverted = q['inverted_count']
            q_poor = q['poor_aspect_count']
            q_poor_pct = 100.0 * q_poor / max(q['total'], 1)
            q_mean_aspect = q.get('mean_aspect', 0.0)
            q_max_aspect = q.get('max_aspect', 0.0)
            q_mean_qual = q.get('mean_quality', 0.0)
            q_min_qual = q.get('min_quality', 0.0)
            q_mean_skew = q.get('mean_skewness', 0.0)
            q_max_skew = q.get('max_skewness', 0.0)
            qtxt = (
                'MESH QUALITY\n'
                f'Status: {q_status}\n'
                f'Invalid: {q_invalid}  Inverted: {q_inverted}\n'
                f'Poor aspect (>20): {q_poor} ({q_poor_pct:.1f}%)\n'
                f'Aspect Ratio: mean {q_mean_aspect:.2f}, max {q_max_aspect:.2f}\n'
                f'Element Quality: mean {q_mean_qual:.3f}, min {q_min_qual:.3f}\n'
                f'Skewness: mean {q_mean_skew:.3f}, max {q_max_skew:.3f}'
            )
            self._quality_actor = self._plotter.add_text(qtxt, position='upper_left', font_size=14, color='#00d26a')
        else:
            self._quality_actor = self._plotter.add_text('MESH QUALITY\nStatus: N/A\nRequires Tet4 or Tet10 mesh', position='upper_left', font_size=14, color='#d0d0d0')

    def _setup_opt_stage(self):
        try:
            self._plotter.disable_picking()
        except Exception:
            pass
        self._add_surface_mesh(opacity=0.2, show_edges=False)
        self._info_actor = self._plotter.add_text('OPTIMIZING...\nPlease wait', position='upper_left', font_size=12, color='white')
        self._opt_done.clear()
        t = threading.Thread(target=self._run_optimization_thread, daemon=True)
        t.start()

    def _run_optimization_thread(self):
        try:
            fixed_dofs, forces = self._build_bc_outputs()
            if forces is None or np.sum(np.abs(forces)) < 1e-10:
                print('  [OPT] No loads defined; aborting optimization.')
                self._opt_done.set()
                return
            cfg_opt = dict(self.config)
            passive_mask = self._build_passive_solid_mask()
            # Force elements touching load nodes to be passive.
            try:
                if passive_mask is None:
                    passive_mask = np.zeros(int(self._original_elements.shape[0]), dtype=bool)
                load_nodes = np.array(sorted(int(n) for n in self.loads.keys()), dtype=np.int64) if len(self.loads) > 0 else np.array([], dtype=np.int64)
                if load_nodes.size > 0:
                    corners = self._original_elements[:, :min(4, self._original_elements.shape[1])]
                    touch_mask2 = np.any(np.isin(corners, load_nodes), axis=1)
                    passive_mask = np.asarray(passive_mask, dtype=bool) | touch_mask2
            except Exception:
                pass

            self._passive_solid_mask = passive_mask.copy() if passive_mask is not None else None
            if passive_mask is not None and np.any(passive_mask):
                cfg_opt['passive_solid_elements'] = passive_mask
                n_passive = int(np.sum(passive_mask))
                n_total = int(len(passive_mask))
                passive_frac = float(n_passive) / float(max(n_total, 1))
                target_vf = float(np.clip(cfg_opt.get('volume_fraction', cfg_opt.get('volfrac', 0.5)), 0.05, 1.0))
                max_reduction = 100.0 * max(0.0, 1.0 - passive_frac)
                print(f'  [OPT] Non-design solid region enforced: {n_passive}/{n_total} elements ({passive_frac:.4f})')
                s_dist = float(self.config.get('support_non_design_distance_mm', self.config.get('non_design_distance_mm', 0.0)))
                f_dist = float(self.config.get('force_non_design_distance_mm', self.config.get('non_design_distance_mm', 0.0)))
                print(f'  [OPT] BC summary: supports={len(self.fixed_nodes)}, loads={len(self.loads)}, support_faces={len(self._support_face_indices)}, load_faces={len(self._load_face_indices)}, supp_dist_mm={s_dist:.3f}, force_dist_mm={f_dist:.3f}')
                if passive_frac > target_vf + 1e-9:
                    print(f'  [OPT][WARN] Target volume fraction {target_vf:.4f} is below protected fraction {passive_frac:.4f}.')
                    print(f'  [OPT][WARN] Current setup can remove at most {max_reduction:.2f}% material unless protected region is reduced.')

            optimizer = self.optimizer_factory(self._original_nodes, self._original_elements, self.material, cfg_opt)
            vis = _QueueVisualizer(self._opt_queue, stop_cb=lambda: self._abort or self._closing)
            ret = optimizer.optimize(forces, fixed_dofs, visualizer=vis)

            rho = None
            comp = None
            if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                rho, comp = ret[0], ret[1]
            else:
                rho = ret
                comp = vis.compliance_history[-1] if vis.compliance_history else 0.0

            if rho is None:
                self._opt_done.set()
                return

            self.compliance_history = list(vis.compliance_history)
            self.volume_history = list(vis.volume_history)
            self.rho_result = np.asarray(rho, dtype=np.float64)
            self.rho_result = self._enforce_manufacturable_result(self.rho_result)
            if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(self.rho_result):
                self.rho_result[self._passive_solid_mask] = 1.0
            self.compliance_result = float(self.compliance_history[-1]) if self.compliance_history else float(comp)
        except Exception as e:
            print(f'  [OPT] Optimization error: {e}')
        finally:
            self._opt_done.set()

    def _process_opt_queue(self):
        updated = False
        latest = None
        try:
            while True:
                latest = self._opt_queue.get_nowait()
                updated = True
        except Exception:
            pass

        if updated and latest is not None:
            iteration, rho, compliance, vol_frac = latest
            rho_view = np.asarray(rho, dtype=np.float64).copy()
            if self._passive_solid_mask is not None and len(self._passive_solid_mask) == len(rho_view):
                rho_view[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0
            self._draw_density_mesh(rho_view)
            n_solid = int(np.sum(rho > 0.5))
            n_total = len(rho)
            stats = (
                f'OPTIMIZING...\n'
                f'Iteration: {iteration + 1}\n'
                f'Compliance: {compliance:.4e}\n'
                f'Volume: {vol_frac:.4f}\n'
                f'Solid: {n_solid}/{n_total}'
            )
            if self._info_actor is not None:
                try:
                    self._plotter.remove_actor(self._info_actor)
                except Exception:
                    pass
            self._info_actor = self._plotter.add_text(stats, position='upper_left', font_size=12, color='white')

    def _compute_display_cutoff(self, rho):
        """Return auto-derived cutoff from target volume (fallback to configured value)."""
        arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(self.config.get('threshold', 0.5))

        vf = self.config.get('volume_fraction', self.config.get('volfrac', 0.3))
        obj = str(self.config.get('objective_function', self.config.get('objective', 'stiffness'))).lower()
        if obj == 'weight':
            wr = self.config.get('weight_reduction_percent', self.config.get('target_weight_reduction_percent', None))
            if wr is not None:
                try:
                    vf = 1.0 - float(wr) / 100.0
                except Exception:
                    pass
        vf = float(np.clip(vf, 0.05, 0.99))

        n = int(arr.size)
        n_solid = int(np.clip(round(vf * n), 1, n))
        k = max(0, n - n_solid)
        thr = float(np.partition(arr, k)[k])
        return float(np.clip(thr, 0.30, 0.80))

    def _draw_density_mesh(self, rho):
        p = self._plotter
        if p is None:
            return

        if self._density_actor is not None:
            try:
                p.remove_actor(self._density_actor)
            except Exception:
                pass
            self._density_actor = None

        n_elem = self.elements.shape[0]
        n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0

        try:
            if n_per in (4, 10):
                self.surf_faces = self._extract_boundary_faces_from_tets(self.elements)
                cells = np.hstack([
                    np.full((n_elem, 1), n_per, dtype=np.int64),
                    self.elements.astype(np.int64)
                ]).ravel()
                vtk_tet_type = 24 if n_per == 10 else 10
                celltypes = np.full(n_elem, vtk_tet_type, dtype=np.uint8)
                grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
            elif n_per == 3:
                cells = np.hstack([np.full((n_elem, 1), 3, dtype=np.int64), self.elements.astype(np.int64)]).ravel()
                grid = pv.PolyData(self.nodes, cells)
            else:
                return

            rho_clipped = np.clip(np.asarray(rho, dtype=np.float64)[:n_elem], 0.0, 1.0)
            if self._passive_solid_mask is not None and len(self._passive_solid_mask) == n_elem:
                rho_clipped[np.asarray(self._passive_solid_mask, dtype=bool)] = 1.0
            grid['density'] = rho_clipped
            draw_cutoff = self._compute_display_cutoff(rho_clipped)
            try:
                visible = grid.threshold(value=draw_cutoff, scalars='density')
            except Exception:
                visible = grid

            self._clear_scalar_bars()
            self._density_actor = p.add_mesh(
                visible,
                scalars='density' if 'density' in visible.array_names else None,
                cmap=['#5b1a98', '#1e56d8', '#1a8cff', '#38d8c9', '#f4e321'],
                clim=[0.0, 1.0],
                show_edges=False,
                opacity=0.92,
                pickable=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Density',
                    'color': 'white',
                    'fmt': '%.2f',
                    'n_labels': 5,
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'width': 0.08,
                    'height': 0.6,
                },
            )
        except Exception as e:
            print(f'  [VIEWER] Draw error: {e}')

    def _setup_results_stage(self):
        if self.rho_result is not None:
            self._draw_density_mesh(self.rho_result)
            thr = self._compute_display_cutoff(self.rho_result)
            n_solid = int(np.sum(np.asarray(self.rho_result, dtype=np.float64) >= thr))
            n_total = len(self.rho_result)
            vol = float(np.mean(self.rho_result))
            comp = float(self.compliance_result) if self.compliance_result is not None else 0.0
            n_iter = len(self.compliance_history) if self.compliance_history else 0

            # Post-opt stress summary
            stress_summary = ''
            vm_post = self._post_stress_vm_all
            if vm_post is not None and len(vm_post) > 0:
                ys = float(getattr(self.material, 'yield_strength', self.config.get('yield_strength', 250.0)))
                sf = float(self.config.get('safety_factor', 1.5))
                allowable = self._get_allowable_stress(sf)
                max_vm = float(np.max(vm_post))
                util = max_vm / max(allowable, 1e-12) * 100.0
                verdict = 'PASS' if util <= 100.0 else 'FAIL'
                stress_summary = (
                    f'\nStress check: {verdict}\n'
                    f'Max VM: {max_vm:.3e} MPa\n'
                    f'Utilization: {util:.1f}%'
                )

            # Mass report
            mass_summary = ''
            mass_report = self._compute_mass_report(self.rho_result)
            if mass_report is not None:
                mass_summary = f'\nMass saved: {mass_report["saved_pct"]:.1f}%'

            reopt_label = f'\nRe-optimizations: {self._reopt_count}' if self._reopt_count > 0 else ''

            info = (
                f'FINAL RESULT\n'
                f'Compliance: {comp:.4e}\n'
                f'Volume fraction: {vol:.4f}\n'
                f'Display cutoff: {thr:.3f}\n'
                f'Visible elements: {n_solid}/{n_total}\n'
                f'Iterations: {n_iter}'
                f'{stress_summary}'
                f'{mass_summary}'
                f'{reopt_label}\n\n'
                f'Q=Quit & EXPORT  |  R=Re-optimize'
            )
            self._info_actor = self._plotter.add_text(info, position='upper_left', font_size=11, color='white')
        else:
            self._add_surface_mesh(opacity=1.0, show_edges=True, edge_color=self._COL_MESH_EDGE)
            self._info_actor = self._plotter.add_text('OPTIMIZATION COMPLETED\n(No density result available)\n\nR=Re-optimize  Q=Exit', position='upper_left', font_size=12, color='white')

    def _key_S(self):
        if self._stage == self.STAGE_BC:
            self._bc_mode = 'SUPPORTS'
            print('  [MODE] -> SUPPORTS')
            self._update_overlays()

    def _key_L(self):
        if self._stage == self.STAGE_BC:
            self._bc_mode = 'LOADS'
            print('  [MODE] -> LOADS')
            self._update_overlays()

    def _key_U(self):
        if self._stage != self.STAGE_BC or not self._undo_stack:
            return
        action, data = self._undo_stack.pop()
        if action == 'restore_supports':
            node_prev = data.get('node_prev', data) if isinstance(data, dict) else data
            for nid, was_fixed in node_prev.items():
                if was_fixed:
                    self.fixed_nodes.add(int(nid))
                else:
                    self.fixed_nodes.discard(int(nid))
            if isinstance(data, dict):
                if 'support_faces_prev' in data:
                    self._support_face_indices = set(int(i) for i in data['support_faces_prev'])
                if 'load_faces_prev' in data:
                    self._load_face_indices = set(int(i) for i in data['load_faces_prev'])
        elif action == 'restore_loads':
            load_prev = data.get('load_prev', data) if isinstance(data, dict) else data
            for nid, prev_load in load_prev.items():
                if prev_load is None:
                    self.loads.pop(int(nid), None)
                else:
                    self.loads[int(nid)] = prev_load
            if isinstance(data, dict):
                if 'support_faces_prev' in data:
                    self._support_face_indices = set(int(i) for i in data['support_faces_prev'])
                if 'load_faces_prev' in data:
                    self._load_face_indices = set(int(i) for i in data['load_faces_prev'])
        self._sync_face_region_sets()
        self._redraw_bc_indicators()
        self._update_overlays()

    def _key_N(self):
        if self._stage == self.STAGE_MESH:
            n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
            if self.mode == '3D' and n_per not in (4, 10):
                print(f'  [WARNING] Cannot proceed: mesh has {n_per}-node elements (need Tet4 or Tet10).')
                print('  [WARNING] Press R to remesh first.')
                return
            self._enter_stage(self.STAGE_BC)
        elif self._stage == self.STAGE_BC:
            if len(self.fixed_nodes) == 0:
                print('  [WARNING] No supports defined. Add supports before proceeding.')
                return
            if len(self.loads) == 0:
                print('  [WARNING] No loads defined. Add loads before proceeding.')
                return
            # Go to stress analysis stage instead of directly to optimization
            self._enter_stage(self.STAGE_STRESS)
        elif self._stage == self.STAGE_STRESS:
            if not self._vf_entered:
                print('  [WARNING] Set volume fraction / mass reduction first (press V).')
                return
            self._enter_stage(self.STAGE_OPT)
        elif self._stage == self.STAGE_POST_STRESS:
            # Accept the optimized design and go to final results
            self._enter_stage(self.STAGE_RESULTS)

    def _key_B(self):
        if self._stage == self.STAGE_BC:
            self._enter_stage(self.STAGE_MESH)
        elif self._stage == self.STAGE_STRESS:
            # Go back to BC definition to adjust supports/loads
            self._enter_stage(self.STAGE_BC)
        elif self._stage == self.STAGE_POST_STRESS:
            # Go back to stress analysis for VF adjustment and re-optimization
            self._reopt_count += 1
            self._vf_entered = False  # force user to re-enter VF
            print(f'  [RE-OPT] Starting re-optimization #{self._reopt_count}...')
            print('  [RE-OPT] Returning to stress analysis stage. Set new VF and press N.')
            self._enter_stage(self.STAGE_STRESS)

    def _key_T(self):
        if self._stage != self.STAGE_MESH or self.mode != '3D':
            return
        if self._remesh_in_progress:
            print('  [MESH] Cannot toggle element type while remeshing is in progress.')
            return
        order = int(self.config.get('gmsh_element_order', 2))
        new_order = 1 if order >= 2 else 2
        self.config['gmsh_element_order'] = int(new_order)
        self.config['tet_element_type'] = 'tet10' if new_order >= 2 else 'tet4'
        tet_label = 'Tet10' if new_order >= 2 else 'Tet4'
        self._remesh_message = f'Ready ({tet_label} selected)'
        print(f'  [MESH] Next remesh element type set to {tet_label}.')
        self._update_overlays()

    def _key_R(self):
        if self._stage == self.STAGE_POST_STRESS or self._stage == self.STAGE_RESULTS:
            # Re-optimize: go back to stress analysis to change VF
            self._reopt_count += 1
            self._vf_entered = False  # force user to re-enter VF
            print(f'  [RE-OPT] Starting re-optimization #{self._reopt_count}...')
            print('  [RE-OPT] Returning to stress analysis stage. Set new VF and press N.')
            self._enter_stage(self.STAGE_STRESS)
            return
        if self._stage != self.STAGE_MESH:
            return
        if self._remesh_callback is None:
            print('  [REMESH] No remesh callback registered')
            return
        if self._remesh_in_progress:
            print('  [REMESH] Already running. Please wait...')
            return
        order = int(self.config.get('gmsh_element_order', 2))
        tet_label = 'Tet10' if order >= 2 else 'Tet4'
        print(f'  [REMESH] Requesting re-mesh ({tet_label})...')
        self._remesh_in_progress = True
        self._remesh_progress = 0.0
        self._remesh_message = 'Starting remesh...'
        self._update_overlays()

        # gmsh requires the main thread (signal handling), so run synchronously for 3D
        if self.mode == '3D':
            self._run_remesh_thread()
            self._process_remesh_queue()
            return

        self._remesh_thread = threading.Thread(target=self._run_remesh_thread, daemon=True)
        self._remesh_thread.start()

    def _run_remesh_thread(self):
        try:
            try:
                new_nodes, new_elements = self._remesh_callback(self._original_nodes, self._original_elements, self._report_remesh_progress)
            except TypeError:
                new_nodes, new_elements = self._remesh_callback(self._original_nodes, self._original_elements)
            self._remesh_queue.put(('ok', new_nodes, new_elements))
        except Exception as e:
            self._remesh_queue.put(('err', str(e), None))

    def _report_remesh_progress(self, percent, message=''):
        if self._abort or self._closing:
            return
        try:
            p = float(np.clip(percent, 0.0, 100.0))
        except Exception:
            p = 0.0
        msg = str(message) if message else 'Meshing...'
        self._remesh_queue.put(('progress', p, msg))

    def _process_remesh_queue(self):
        while not self._remesh_queue.empty():
            try:
                status, a, b = self._remesh_queue.get_nowait()
            except Exception:
                break

            if status == 'progress':
                self._remesh_progress = float(a)
                self._remesh_message = str(b)
                self._update_overlays()
                continue

            self._remesh_in_progress = False

            if status == 'err':
                self._remesh_message = f'Failed: {a}'
                print(f'  [REMESH] Error: {a}')
                self._update_overlays()
                continue

            new_nodes, new_elements = a, b
            if new_nodes is None or new_elements is None:
                self._remesh_message = 'No mesh returned'
                self._update_overlays()
                continue

            old_nodes = np.asarray(self.nodes, dtype=np.float64)
            old_fixed_ids = [int(n) for n in self.fixed_nodes if 0 <= int(n) < old_nodes.shape[0]]
            old_load_items = [(int(n), self.loads[int(n)]) for n in self.loads.keys() if 0 <= int(n) < old_nodes.shape[0]]

            old_faces = self._get_display_faces()
            old_support_centers = self._capture_face_region_centers(old_faces, self._support_face_indices, old_nodes)
            old_load_centers = self._capture_face_region_centers(old_faces, self._load_face_indices, old_nodes)

            self._original_nodes = np.asarray(new_nodes, dtype=np.float64)
            self._original_elements = np.asarray(new_elements, dtype=np.int32)
            self.nodes = self._original_nodes.copy() if self._original_nodes.shape[1] == 3 else np.hstack([self._original_nodes, np.zeros((self._original_nodes.shape[0], 1))])
            self.elements = self._original_elements.copy()

            self._boundary_faces_cache = None
            self._surface_edges_cache = None
            n_per = self.elements.shape[1] if self.elements.ndim == 2 else 0
            if n_per in (4, 10):
                self.surf_faces = self._extract_boundary_faces_from_tets(self.elements)
            elif n_per == 3:
                self.surf_faces = self.elements.copy()
            else:
                self.surf_faces = None

            self.face_ids = None
            self.edge_segments = None

            self.fixed_nodes = set()
            self.loads = {}
            for nid in old_fixed_ids:
                cc = old_nodes[nid]
                j = int(np.argmin(np.sum((self.nodes - cc.reshape(1, 3)) ** 2, axis=1)))
                self.fixed_nodes.add(j)
            for nid, lv in old_load_items:
                cc = old_nodes[nid]
                j = int(np.argmin(np.sum((self.nodes - cc.reshape(1, 3)) ** 2, axis=1)))
                self.loads[j] = {'dir': int(lv.get('dir', self._load_direction)), 'mag': float(lv.get('mag', self._load_magnitude))}

            new_faces = self._get_display_faces()
            self._support_face_indices = self._map_face_region_centers_to_indices(new_faces, self.nodes, old_support_centers)
            self._load_face_indices = self._map_face_region_centers_to_indices(new_faces, self.nodes, old_load_centers)
            self._sync_face_region_sets()
            self._remesh_progress = 100.0
            self._remesh_message = 'Remesh complete'
            self._redraw_bc_indicators()
            self._enter_stage(self.STAGE_MESH)
    def _print_welcome(self):
        n_v = self.nodes.shape[0]
        n_e = self.elements.shape[0]
        print('\n' + '=' * 70)
        print('  UNIFIED TOPOLOGY OPTIMIZATION GUI')
        print('=' * 70)
        print(f'  Mode: {self.mode}  |  Nodes: {n_v:,}  |  Elements: {n_e:,}')
        print('\n  Stage 1: Review & remesh (R key to generate tet mesh)')
        print('  Stage 2: Define boundary conditions (supports + loads)')
        print('  Stage 3: Static stress analysis (view stress heatmap)')
        print('  Stage 4: Set volume fraction & run optimization live')
        print('  Stage 5: Post-optimization stress analysis (verify design)')
        print('  Stage 6: Accept result OR re-optimize with different VF')
        print('  Stage 7: View and export final result')
        print('\n  The GUI will stay open through ALL stages.')
        print('  You can re-optimize as many times as needed (R key).')
        print('=' * 70 + '\n')


class _QueueVisualizer:
    """Adapter that matches the LiveViewer3D interface but pushes updates to a queue."""

    def __init__(self, q, stop_cb=None):
        self._q = q
        self._stop_cb = stop_cb
        self.compliance_history = []
        self.volume_history = []

    def start(self):
        pass

    def update(self, iteration, rho, compliance, volume_fraction):
        if self._stop_cb is not None:
            try:
                if self._stop_cb():
                    return
            except Exception:
                pass

        self.compliance_history.append(float(compliance))
        self.volume_history.append(float(volume_fraction))

        thr = 0.5
        rho_arr = np.asarray(rho, dtype=np.float64)
        n_total = len(rho)
        n_occ = np.sum(rho_arr >= thr)
        occ_pct = n_occ / n_total * 100 if n_total else 0.0
        mass_pct = float(np.mean(rho_arr) * 100.0) if n_total else 0.0
        rho_min = float(np.min(rho_arr)) if n_total else 0.0
        rho_mean = float(np.mean(rho_arr)) if n_total else 0.0
        rho_max = float(np.max(rho_arr)) if n_total else 0.0
        bar_len = 30
        filled = int(bar_len * mass_pct / 100)
        bar = '|' * filled + '.' * (bar_len - filled)
        print(f'  [{bar}] mass={mass_pct:.1f}%  occ(rho>=0.50)={occ_pct:.0f}%  |  C={compliance:.4e}  V={volume_fraction:.4f}  rho[min/mean/max]={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f}')

        self._q.put((iteration, rho.copy(), compliance, volume_fraction))

    def show_final(self, rho):
        pass

    def close(self):
        pass







































































