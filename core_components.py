import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from enum import Enum

class OptimizationMethod(Enum):
    """Enumeration of available topology optimization methods"""
    SIMP = "SIMP"
    BESO = "BESO"
    LSM = "LSM"
    MORI_TANAKA = "MORI_TANAKA"
    HOMOGENIZATION = "MORI_TANAKA"  # backward-compatible alias

class MaterialModel:
    """Material properties and SIMP penalization."""
    def __init__(self, E0=1.0, nu=0.3, rho_min=1e-6,
                 density=1.0, yield_strength=1.0e9,
                 thermal_expansion=None, fatigue_limit=None, fracture_toughness=None,
                 anisotropy_factors=None, yield_anisotropy_factors=None, build_axis='z', process_name=None):
        self.E0 = E0  # Young's modulus (reference stiffness, Pa)
        self.nu = nu  # Poisson's ratio (dimensionless)
        self.rho_min = rho_min  # Minimum relative density (SIMP floor)
        self.density = density  # Mass density (kg/m^3)
        self.yield_strength = yield_strength  # Yield stress (Pa)
        self.thermal_expansion = thermal_expansion  # Linear CTE (1/K)
        self.fatigue_limit = fatigue_limit  # Approximate endurance limit (Pa)
        self.fracture_toughness = fracture_toughness  # K_IC (MPa*sqrt(m))
        self.anisotropy_factors = anisotropy_factors if anisotropy_factors else None
        self.yield_anisotropy_factors = yield_anisotropy_factors if yield_anisotropy_factors else None
        self.build_axis = build_axis if build_axis in ('x', 'y', 'z') else 'z'
        self.process_name = process_name

    def get_density_scale(self, rho, penalty=3.0):
        """Density-based stiffness scaling for SIMP."""
        rho_f = float(rho)
        return self.rho_min + (1.0 - self.rho_min) * (rho_f ** penalty)

    def get_modulus(self, rho, penalty=3.0):
        """Calculate Young's modulus based on density (SIMP model)"""
        return self.E0 * self.get_density_scale(rho, penalty)

    def get_yield_factor(self):
        factors = self.yield_anisotropy_factors or self.anisotropy_factors
        if not factors:
            return 1.0
        fx = float(factors.get('x', 1.0))
        fy = float(factors.get('y', 1.0))
        fz = float(factors.get('z', 1.0))
        return float(min(fx, fy, fz))

    def get_allowable_stress(self, safety_factor=1.0):
        sf = max(float(safety_factor), 1e-12)
        ys = float(self.yield_strength)
        return ys * self.get_yield_factor() / sf

    def _isotropic_stiffness_matrix(self, E, nu):
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return np.array([
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ], dtype=np.float64)

    def _orthotropic_stiffness_matrix(self, ex, ey, ez, nu, gxy, gyz, gxz):
        """Approximate orthotropic stiffness from directional moduli.

        Uses a symmetric compliance matrix with reciprocal Poisson ratios.
        """
        ex = max(float(ex), 1e-12)
        ey = max(float(ey), 1e-12)
        ez = max(float(ez), 1e-12)
        nu_xy = float(nu)
        nu_yz = float(nu)
        nu_xz = float(nu)
        nu_yx = nu_xy * ey / ex
        nu_zy = nu_yz * ez / ey
        nu_zx = nu_xz * ez / ex

        s = np.zeros((6, 6), dtype=np.float64)
        s[0, 0] = 1.0 / ex
        s[1, 1] = 1.0 / ey
        s[2, 2] = 1.0 / ez
        s[0, 1] = -nu_yx / ey
        s[0, 2] = -nu_zx / ez
        s[1, 0] = -nu_xy / ex
        s[1, 2] = -nu_zy / ez
        s[2, 0] = -nu_xz / ex
        s[2, 1] = -nu_yz / ey
        s[3, 3] = 1.0 / max(float(gyz), 1e-12)
        s[4, 4] = 1.0 / max(float(gxz), 1e-12)
        s[5, 5] = 1.0 / max(float(gxy), 1e-12)

        try:
            return np.linalg.inv(s)
        except Exception:
            return self._isotropic_stiffness_matrix(self.E0, self.nu)

    def get_constitutive_matrix(self):
        """Return the 3D constitutive matrix at full density."""
        if not self.anisotropy_factors:
            return self._isotropic_stiffness_matrix(self.E0, self.nu)

        fx = float(self.anisotropy_factors.get('x', 1.0))
        fy = float(self.anisotropy_factors.get('y', 1.0))
        fz = float(self.anisotropy_factors.get('z', 1.0))
        if abs(fx - 1.0) < 1e-8 and abs(fy - 1.0) < 1e-8 and abs(fz - 1.0) < 1e-8:
            return self._isotropic_stiffness_matrix(self.E0, self.nu)

        ex = self.E0 * fx
        ey = self.E0 * fy
        ez = self.E0 * fz
        gxy = (np.sqrt(ex * ey)) / (2.0 * (1.0 + self.nu))
        gyz = (np.sqrt(ey * ez)) / (2.0 * (1.0 + self.nu))
        gxz = (np.sqrt(ex * ez)) / (2.0 * (1.0 + self.nu))
        return self._orthotropic_stiffness_matrix(ex, ey, ez, self.nu, gxy, gyz, gxz)

    def get_stiffness_matrix_2d(self, E, nu):
        """2D plane stress constitutive matrix"""
        factor = E / (1 - nu**2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])

class MeshGenerator:
    """Generate 2D rectangular meshes for FEA"""
    def __init__(self, length=10.0, height=10.0, elem_x=20, elem_y=20):
        self.length = length
        self.height = height
        self.elem_x = elem_x
        self.elem_y = elem_y
        self.n_elements = elem_x * elem_y
        
    def generate(self):
        """Generate structured mesh"""
        # Node coordinates
        x = np.linspace(0, self.length, self.elem_x + 1)
        y = np.linspace(0, self.height, self.elem_y + 1)
        xx, yy = np.meshgrid(x, y)
        
        nodes = np.column_stack([xx.flatten(), yy.flatten()])
        n_nodes = nodes.shape[0]
        
        # Element connectivity
        elements = np.zeros((self.n_elements, 4), dtype=int)
        elem_idx = 0
        for j in range(self.elem_y):
            for i in range(self.elem_x):
                n1 = j * (self.elem_x + 1) + i
                n2 = n1 + 1
                n3 = n1 + self.elem_x + 2
                n4 = n1 + self.elem_x + 1
                elements[elem_idx] = [n1, n2, n3, n4]
                elem_idx += 1
        
        return nodes, elements
    
    def element_area(self):
        """Calculate element area"""
        return (self.length / self.elem_x) * (self.height / self.elem_y)

class FEASolver:
    """Simplified 2D FEA Solver for plane stress problems"""
    def __init__(self, nodes, elements, material, thickness=1.0):
        self.nodes = nodes
        self.elements = elements
        self.material = material
        self.thickness = thickness
        self.n_nodes = nodes.shape[0]
        self.n_dofs = self.n_nodes * 2
        self.n_elements = elements.shape[0]
        
    def get_element_stiffness(self, elem_idx, rho, penalty=3.0):
        """Calculate element stiffness matrix for Q4 element"""
        elem = self.elements[elem_idx]
        coords = self.nodes[elem]
        
        # Element stiffness matrix (simplified 8x8 for Q4)
        # Using standard bilinear quadrilateral element
        E = self.material.get_modulus(rho[elem_idx], penalty)
        D = self.material.get_stiffness_matrix_2d(E, self.material.nu)
        
        # Simplified: use averages and integration points
        Ke = self._compute_Ke_Q4(coords, D)
        return Ke * self.thickness
    
    def _compute_Ke_Q4(self, coords, D):
        """Compute Q4 element stiffness using numerical integration"""
        # Integration points (2x2 Gauss)
        gp_coords = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), 1/np.sqrt(3)],
                              [-1/np.sqrt(3), 1/np.sqrt(3)]])
        
        Ke = np.zeros((8, 8))
        
        for gp in gp_coords:
            xi, eta = gp
            # Shape functions derivatives
            dNdxi = 0.25 * np.array([
                [-(1-eta), (1-eta), (1+eta), -(1+eta)],
                [-(1-xi), -(1+xi), (1+xi), (1-xi)]
            ])
            
            # Jacobian
            J = dNdxi @ coords
            detJ = np.linalg.det(J)
            dNdxy = np.linalg.solve(J, dNdxi)
            
            # B matrix (strain-displacement)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i] = dNdxy[0, i]
                B[1, 2*i+1] = dNdxy[1, i]
                B[2, 2*i] = dNdxy[1, i]
                B[2, 2*i+1] = dNdxy[0, i]
            
            Ke += B.T @ D @ B * detJ
        
        return Ke
    
    def assemble_global_stiffness(self, rho, penalty=3.0):
        """Assemble global stiffness matrix (sparse format)"""
        from scipy.sparse import csr_matrix, coo_matrix
        
        row_ind, col_ind, data = [], [], []
        
        for elem_idx in range(self.n_elements):
            Ke = self.get_element_stiffness(elem_idx, rho, penalty)
            elem = self.elements[elem_idx]
            
            # Global DOF indices
            dofs = np.concatenate([[2*n, 2*n+1] for n in elem])
            
            # Add to triplet format
            for i in range(8):
                for j in range(8):
                    row_ind.append(dofs[i])
                    col_ind.append(dofs[j])
                    data.append(Ke[i, j])
        
        K = coo_matrix((data, (row_ind, col_ind)), shape=(self.n_dofs, self.n_dofs))
        return K.tocsr()
    
    def solve(self, rho, forces, fixed_dofs, penalty=3.0):
        """Solve linear system Ku = F"""
        K = self.assemble_global_stiffness(rho, penalty)
        
        # Apply boundary conditions (simple approach)
        free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
        
        # Reduce system
        K_ff = K[free_dofs][:, free_dofs].tocsr()
        f_f = forces[free_dofs]
        
        # Solve reduced system
        from scipy.sparse.linalg import spsolve
        u_f = spsolve(K_ff, f_f)
        
        # Full displacement vector
        u = np.zeros(self.n_dofs)
        u[free_dofs] = u_f
        
        return u
    
    def calculate_compliance(self, u, rho, penalty=3.0):
        """Calculate structural compliance (objective function)"""
        compliance = 0.0
        
        for elem_idx in range(self.n_elements):
            elem = self.elements[elem_idx]
            dofs = np.concatenate([[2*n, 2*n+1] for n in elem])
            u_elem = u[dofs]
            
            Ke = self.get_element_stiffness(elem_idx, rho, penalty)
            compliance += u_elem @ Ke @ u_elem
        
        return compliance
    
    def calculate_stress(self, u, elem_idx, rho, penalty=3.0):
        """Calculate stress in element"""
        elem = self.elements[elem_idx]
        coords = self.nodes[elem]
        dofs = np.concatenate([[2*n, 2*n+1] for n in elem])
        u_elem = u[dofs]
        
        E = self.material.get_modulus(rho[elem_idx], penalty)
        D = self.material.get_stiffness_matrix_2d(E, self.material.nu)
        
        # Simplified stress calculation at element center
        xi, eta = 0.0, 0.0
        dNdxi = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        
        J = dNdxi @ coords
        dNdxy = np.linalg.solve(J, dNdxi)
        
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dNdxy[0, i]
            B[1, 2*i+1] = dNdxy[1, i]
            B[2, 2*i] = dNdxy[1, i]
            B[2, 2*i+1] = dNdxy[0, i]
        
        strain = B @ u_elem
        stress = D @ strain
        
        return stress  # [sigma_xx, sigma_yy, sigma_xy]

class Visualizer:
    """Real-time visualization of topology optimization"""
    def __init__(self, mesh_generator, update_interval=100):
        self.mesh_gen = mesh_generator
        self.update_interval = update_interval
        self.fig = None
        self.ax = None
        self.iteration = 0
        self.history = {'compliance': [], 'volume_fraction': []}
        
    def plot_density(self, rho, title="Density Distribution", show=True):
        """Plot current density distribution"""
        elem_x = self.mesh_gen.elem_x
        elem_y = self.mesh_gen.elem_y
        
        density_grid = np.reshape(rho, (elem_y, elem_x))
        
        if self.fig is None:
            self.fig, (self.ax, self.ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
        
        self.ax.clear()
        im = self.ax.imshow(density_grid, cmap='gray', origin='lower', 
                           extent=[0, self.mesh_gen.length, 0, self.mesh_gen.height])
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.colorbar(im, ax=self.ax, label='Density')
        
        # Update history plots
        self.ax_hist.clear()
        if len(self.history['compliance']) > 0:
            ax1 = self.ax_hist
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(self.history['compliance'], 'b-', label='Compliance')
            line2 = ax2.plot(self.history['volume_fraction'], 'r-', label='Volume Fraction')
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Compliance', color='b')
            ax2.set_ylabel('Volume Fraction', color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        if show:
            plt.pause(0.01)
        
        return self.fig, self.ax
    
    def update_history(self, compliance, volume_fraction):
        """Update optimization history"""
        self.history['compliance'].append(compliance)
        self.history['volume_fraction'].append(volume_fraction)

# =============================================================================
# CONFIG.PY CONTENT
# =============================================================================
DEFAULT_CONFIG = {
    'method': 'SIMP',  # TO method
    'domain_length': 10.0,
    'domain_height': 10.0,
    'mesh_x': 20,
    'mesh_y': 20,
    'volume_fraction': 0.5,  # Target volume fraction
    'iterations': 300,
    'lsm_reinit_interval': 12,
    'lsm_reinit_band': 0.04,
    'lsm_reinit_blend': 0.70,
    'lsm_reinit_method': 'graph_fmm',
    'lsm_use_interface_shape_derivative': True,
    'lsm_interface_band': 0.08,
    'penalization': 3.0,  # SIMP penalty exponent
    'auto_filter_radius': True,  # Compute filter radius from mesh element size
    'filter_radius_factor': 3.5,  # Radius = factor x median tet edge length
    'filter_radius': 1.5,  # Optional manual override when auto_filter_radius is False
    'thickness': 1.0,  # Element thickness
    'young_modulus': 1.0,
    'poisson_ratio': 0.3,
    'material_density': 7800.0,
    'yield_strength': 250e6,
    'thermal_expansion': None,
    'fatigue_limit': None,
    'fracture_toughness': None,
    'am_process_key': None,
    'am_process_name': None,
    'am_build_axis': 'z',
    'am_anisotropy_factors': None,
    'am_yield_anisotropy_factors': None,
    'am_stiffness_scale': 1.0,
    'am_yield_scale': 1.0,
    'am_density_scale': 1.0,
    'use_stress_constraint': True,
    'safety_factor': 1.5,
    'stress_penalty_weight': 2.0,
    'use_stress_augmented_lagrangian': True,
    'stress_al_mu0': 8.0,
    'stress_al_mu_growth': 1.10,
    'stress_al_mu_max': 1e4,
    'threshold': 0.5,  # Density threshold for visualization
    'final_density_smoothing_passes': 2,
    'final_density_smoothing_blend': 0.60,
    'smooth_result_surface': True,
    'result_surface_smooth_iterations': 25,
    'result_surface_relaxation': 0.08,
    'step_export_target_faces': 5000,
    'run_verification_fea': True,
    'linear_solver': 'auto',
    'iterative_solver_dof_threshold': 50000,
    'iterative_solver_tol': 1e-8,
    'iterative_solver_maxiter': 2000,
    'use_pyamg_preconditioner': True,
    'min_3d_elements': 40,
    'max_3d_elements': 100000,
    'tet_element_type': 'tet10',
    'gmsh_element_order': 2,
    'tet10_warning_elements': 12000,
    'stress_eval_interval': 5,
    'log_iteration_timing': True,
    'enable_overhang_filter': False,
    'overhang_angle_deg': 45.0,
    'build_axis': 'z',
    'check_enclosed_voids': True,
    'min_member_size_mm': None,
    'beso_addback_alpha': 0.95,
    'fair_method_comparison': True,
    'use_min_member_projection_beso': False,
    'use_min_member_projection_lsm': False,
    'fair_raw_method_output': True,
}

print("Core components created")


