import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, cg

try:
    import pyamg
    _HAS_PYAMG = True
except Exception:
    pyamg = None
    _HAS_PYAMG = False


class FEASolver3D:
    """3D linear-elastic FEA solver for Tet4 and Tet10 elements."""

    def __init__(self, nodes, elements, material, thickness=1.0):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.material = material
        self.thickness = thickness

        if self.elements.ndim != 2 or self.elements.shape[1] not in (4, 10):
            raise ValueError('FEA solver expects 4-node (Tet4) or 10-node (Tet10) tetrahedral connectivity')

        self.n_nodes = self.nodes.shape[0]
        self.n_dofs = self.n_nodes * 3
        self.n_elements = self.elements.shape[0]
        self.npe = int(self.elements.shape[1])
        self.dof_per_elem = 3 * self.npe

        self.nu = float(self.material.nu)
        self._d_base = self.material.get_constitutive_matrix()

        self.elem_dofs = (3 * self.elements[:, :, None] + np.arange(3, dtype=np.int64)).reshape(self.n_elements, self.dof_per_elem)

        local = np.arange(self.dof_per_elem, dtype=np.int64)
        ii = np.repeat(local, self.dof_per_elem)
        jj = np.tile(local, self.dof_per_elem)
        self._row_ind = self.elem_dofs[:, ii].reshape(-1)
        self._col_ind = self.elem_dofs[:, jj].reshape(-1)

        self._ke_unit = self._precompute_unit_stiffness()
        self._bm_centroid, self._det_centroid = self._precompute_centroid_b()

        self.linear_solver_mode = 'auto'
        self.iterative_solver_dof_threshold = 50000
        self.iterative_solver_tol = 1e-8
        self.iterative_solver_maxiter = 2000
        self.use_pyamg_preconditioner = True
        self._solver_notice_printed = False

    def configure_linear_solver(self, mode='auto', iterative_threshold_dofs=50000,
                                cg_tol=1e-8, cg_maxiter=2000, use_pyamg=True):
        self.linear_solver_mode = str(mode).lower()
        self.iterative_solver_dof_threshold = int(max(1, iterative_threshold_dofs))
        self.iterative_solver_tol = float(max(cg_tol, 1e-14))
        self.iterative_solver_maxiter = int(max(10, cg_maxiter))
        self.use_pyamg_preconditioner = bool(use_pyamg)

    @staticmethod
    def _constitutive_matrix(E, nu):
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

    @staticmethod
    def _shape_derivatives_nat_tet10(r, s, t):
        l1 = 1.0 - r - s - t
        l2 = r
        l3 = s
        l4 = t

        dNdL = np.zeros((10, 4), dtype=np.float64)
        dNdL[0, 0] = 4.0 * l1 - 1.0
        dNdL[1, 1] = 4.0 * l2 - 1.0
        dNdL[2, 2] = 4.0 * l3 - 1.0
        dNdL[3, 3] = 4.0 * l4 - 1.0

        dNdL[4, 0] = 4.0 * l2
        dNdL[4, 1] = 4.0 * l1
        dNdL[5, 1] = 4.0 * l3
        dNdL[5, 2] = 4.0 * l2
        dNdL[6, 0] = 4.0 * l3
        dNdL[6, 2] = 4.0 * l1
        dNdL[7, 0] = 4.0 * l4
        dNdL[7, 3] = 4.0 * l1
        dNdL[8, 1] = 4.0 * l4
        dNdL[8, 3] = 4.0 * l2
        dNdL[9, 2] = 4.0 * l4
        dNdL[9, 3] = 4.0 * l3

        dL_dr = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        dL_ds = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        dL_dt = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float64)

        dNdr = dNdL @ dL_dr
        dNds = dNdL @ dL_ds
        dNdt = dNdL @ dL_dt
        return np.vstack([dNdr, dNds, dNdt])

    @staticmethod
    def _shape_derivatives_nat_tet4():
        return np.array([
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @staticmethod
    def _build_b_matrix(dN_xyz):
        nshape = int(dN_xyz.shape[1])
        b = np.zeros((6, 3 * nshape), dtype=np.float64)
        for i in range(nshape):
            dx, dy, dz = dN_xyz[0, i], dN_xyz[1, i], dN_xyz[2, i]
            ii = 3 * i
            b[0, ii] = dx
            b[1, ii + 1] = dy
            b[2, ii + 2] = dz
            b[3, ii] = dy
            b[3, ii + 1] = dx
            b[4, ii + 1] = dz
            b[4, ii + 2] = dy
            b[5, ii] = dz
            b[5, ii + 2] = dx
        return b

    @classmethod
    def _b_matrix_tet10(cls, coords, r, s, t):
        dN_nat = cls._shape_derivatives_nat_tet10(r, s, t)
        j = dN_nat @ coords
        det_j = float(np.linalg.det(j))
        if abs(det_j) < 1e-14:
            return None, 0.0
        inv_j = np.linalg.inv(j)
        dN_xyz = inv_j @ dN_nat
        return cls._build_b_matrix(dN_xyz), det_j

    @classmethod
    def _b_matrix_tet4(cls, coords4):
        dN_nat = cls._shape_derivatives_nat_tet4()
        j = dN_nat @ coords4
        det_j = float(np.linalg.det(j))
        if abs(det_j) < 1e-14:
            return None, 0.0
        inv_j = np.linalg.inv(j)
        dN_xyz = inv_j @ dN_nat
        return cls._build_b_matrix(dN_xyz), det_j

    def _compute_unit_element_stiffness(self, elem_idx):
        elem = self.elements[int(elem_idx)]

        if self.npe == 4:
            coords4 = self.nodes[elem[:4]]
            bm, det_j = self._b_matrix_tet4(coords4)
            if bm is None:
                return np.eye(self.dof_per_elem, dtype=np.float64) * 1e-12
            ke = bm.T @ self._d_base @ bm * abs(det_j) / 6.0
            if not np.any(np.isfinite(ke)) or float(np.max(np.abs(ke))) < 1e-20:
                return np.eye(self.dof_per_elem, dtype=np.float64) * 1e-12
            return ke

        coords = self.nodes[elem]
        # 5-point Keast rule -- integrates up to degree 3 exactly.
        # The negative central weight is inherent to the Keast formulation
        # and does not affect accuracy for well-shaped elements.
        gps = [
            (0.25, 0.25, 0.25),
            (1.0/6, 1.0/6, 1.0/6),
            (1.0/6, 1.0/6, 0.5),
            (1.0/6, 0.5, 1.0/6),
            (0.5, 1.0/6, 1.0/6),
        ]
        gw = [-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0]

        ke = np.zeros((self.dof_per_elem, self.dof_per_elem), dtype=np.float64)
        for (r, s, t), w in zip(gps, gw):
            bm, det_j = self._b_matrix_tet10(coords, r, s, t)
            if bm is None:
                continue
            ke += bm.T @ self._d_base @ bm * abs(det_j) * w

        if not np.any(np.isfinite(ke)) or float(np.max(np.abs(ke))) < 1e-20:
            return np.eye(self.dof_per_elem, dtype=np.float64) * 1e-12
        return ke

    def _precompute_unit_stiffness(self):
        """Precompute unit stiffness matrices for all elements.

        For Tet4: fully vectorized using batched NumPy operations (10-50x faster).
        For Tet10: per-element loop with 5-point Gauss quadrature.
        """
        ke_unit = np.zeros((self.n_elements, self.dof_per_elem, self.dof_per_elem), dtype=np.float64)

        if self.npe == 4:
            # Vectorized Tet4: compute all Jacobians, B-matrices, and ke simultaneously
            dN_nat = self._shape_derivatives_nat_tet4()  # (3, 4) — constant for all Tet4
            elem_coords = self.nodes[self.elements[:, :4]]  # (n_elem, 4, 3)

            # Jacobians: J[e] = dN_nat @ coords[e] -> (n_elem, 3, 3)
            # dN_nat is (3, 4), elem_coords is (n_elem, 4, 3)
            J_all = np.einsum('ij,ejk->eik', dN_nat, elem_coords)
            det_J = np.linalg.det(J_all)  # (n_elem,)

            # Identify valid (non-degenerate) elements
            valid = np.abs(det_J) >= 1e-14

            # Inverse Jacobians for valid elements
            inv_J = np.zeros_like(J_all)
            inv_J[valid] = np.linalg.inv(J_all[valid])

            # dN/dxyz = inv_J @ dN_nat -> (n_elem, 3, 4)
            dN_xyz = np.einsum('eij,jk->eik', inv_J, dN_nat)

            # Build B-matrices in batch: (n_elem, 6, 12)
            B_all = np.zeros((self.n_elements, 6, 12), dtype=np.float64)
            for i in range(4):
                ii = 3 * i
                B_all[:, 0, ii]     = dN_xyz[:, 0, i]  # dN/dx
                B_all[:, 1, ii + 1] = dN_xyz[:, 1, i]  # dN/dy
                B_all[:, 2, ii + 2] = dN_xyz[:, 2, i]  # dN/dz
                B_all[:, 3, ii]     = dN_xyz[:, 1, i]   # dN/dy
                B_all[:, 3, ii + 1] = dN_xyz[:, 0, i]   # dN/dx
                B_all[:, 4, ii + 1] = dN_xyz[:, 2, i]   # dN/dz
                B_all[:, 4, ii + 2] = dN_xyz[:, 1, i]   # dN/dy
                B_all[:, 5, ii]     = dN_xyz[:, 2, i]   # dN/dz
                B_all[:, 5, ii + 2] = dN_xyz[:, 0, i]   # dN/dx

            # ke = B^T @ D @ B * |det_J| / 6 — batched via einsum
            DB = np.einsum('ij,ejk->eik', self._d_base, B_all)           # (n_elem, 6, 12)
            ke_batch = np.einsum('eji,ejk->eik', B_all, DB)              # (n_elem, 12, 12)
            ke_batch *= (np.abs(det_J) / 6.0)[:, None, None]

            # Assign valid elements; invalid get tiny diagonal
            ke_unit[valid] = ke_batch[valid]
            invalid = ~valid
            if np.any(invalid):
                ke_unit[invalid] = np.eye(self.dof_per_elem, dtype=np.float64)[None, :, :] * 1e-12

            # Sanity check: zero out non-finite entries
            bad = ~np.all(np.isfinite(ke_unit), axis=(1, 2))
            if np.any(bad):
                ke_unit[bad] = np.eye(self.dof_per_elem, dtype=np.float64)[None, :, :] * 1e-12

        else:
            # Tet10: per-element loop (Gauss quadrature required)
            for elem_idx in range(self.n_elements):
                ke_unit[elem_idx] = self._compute_unit_element_stiffness(elem_idx)

        return ke_unit

    def _precompute_centroid_b(self):
        bm_all = np.zeros((self.n_elements, 6, self.dof_per_elem), dtype=np.float64)
        det_all = np.zeros(self.n_elements, dtype=np.float64)

        for elem_idx in range(self.n_elements):
            elem = self.elements[elem_idx]
            if self.npe == 4:
                coords4 = self.nodes[elem[:4]]
                bm, det_j = self._b_matrix_tet4(coords4)
            else:
                coords = self.nodes[elem]
                bm, det_j = self._b_matrix_tet10(coords, 0.25, 0.25, 0.25)

            if bm is not None and abs(det_j) >= 1e-14:
                bm_all[elem_idx] = bm
                det_all[elem_idx] = det_j

        return bm_all, det_all

    def _element_modulus(self, rho, penalty):
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        return np.array([self.material.get_density_scale(float(r), penalty) for r in rho_arr], dtype=np.float64)

    def get_element_stiffness(self, elem_idx, rho, penalty=3.0):
        scale = float(self.material.get_density_scale(float(np.asarray(rho, dtype=np.float64)[int(elem_idx)]), penalty))
        return self._ke_unit[int(elem_idx)] * scale

    def assemble_global_stiffness(self, rho, penalty=3.0):
        scale = self._element_modulus(rho, penalty)
        data = (self._ke_unit.reshape(self.n_elements, -1) * scale[:, None]).reshape(-1)
        return coo_matrix((data, (self._row_ind, self._col_ind)), shape=(self.n_dofs, self.n_dofs)).tocsr()

    def solve(self, rho, forces, fixed_dofs, penalty=3.0):
        k = self.assemble_global_stiffness(rho, penalty)
        free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
        k_ff = k[free_dofs][:, free_dofs].tocsr()
        f_f = np.asarray(forces, dtype=np.float64)[free_dofs]

        mode = str(getattr(self, 'linear_solver_mode', 'auto')).lower()
        use_iterative = False
        if mode in ('iterative', 'cg', 'cg-amg', 'cg_amg'):
            use_iterative = True
        elif mode == 'auto':
            use_iterative = (k_ff.shape[0] >= int(getattr(self, 'iterative_solver_dof_threshold', 50000)))

        if use_iterative:
            M = None
            if bool(getattr(self, 'use_pyamg_preconditioner', True)) and _HAS_PYAMG:
                try:
                    ml = pyamg.smoothed_aggregation_solver(k_ff)
                    M = ml.aspreconditioner()
                except Exception:
                    M = None

            tol = float(getattr(self, 'iterative_solver_tol', 1e-8))
            maxiter = int(getattr(self, 'iterative_solver_maxiter', 2000))
            try:
                u_f, info = cg(k_ff, f_f, tol=tol, maxiter=maxiter, M=M)
            except TypeError:
                u_f, info = cg(k_ff, f_f, rtol=tol, atol=0.0, maxiter=maxiter, M=M)

            if info == 0 and np.all(np.isfinite(u_f)):
                if not self._solver_notice_printed:
                    msg = 'CG+AMG' if (M is not None) else 'CG'
                    print(f'  [SOLVER] Using iterative {msg} linear solve path.')
                    self._solver_notice_printed = True
            else:
                if not self._solver_notice_printed:
                    print('  [SOLVER] Iterative solve did not converge, falling back to direct spsolve.')
                    self._solver_notice_printed = True
                u_f = spsolve(k_ff, f_f)
        else:
            u_f = spsolve(k_ff, f_f)

        u = np.zeros(self.n_dofs, dtype=np.float64)
        u[free_dofs] = u_f
        return u

    def calculate_compliance(self, u, rho, penalty=3.0):
        scale = self._element_modulus(rho, penalty)
        uu = np.asarray(u, dtype=np.float64)
        ue = uu[self.elem_dofs]
        ce_unit = np.einsum('ni,nij,nj->n', ue, self._ke_unit, ue)
        return float(np.sum(scale * ce_unit))

    def calculate_stress(self, u, elem_idx, rho, penalty=3.0):
        e = int(elem_idx)
        dofs = self.elem_dofs[e]
        ue = np.asarray(u, dtype=np.float64)[dofs]

        if abs(self._det_centroid[e]) < 1e-14:
            return np.zeros(6, dtype=np.float64), 0.0

        strain = self._bm_centroid[e] @ ue
        scale = float(self.material.get_density_scale(float(np.asarray(rho, dtype=np.float64)[e]), penalty))
        stress = (self._d_base * scale) @ strain

        sx, sy, sz = stress[0], stress[1], stress[2]
        txy, tyz, tzx = stress[3], stress[4], stress[5]
        vm = float(np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))))
        return stress, vm

    def calculate_stress_gauss(self, u, elem_idx, rho, penalty=3.0):
        """Calculate stress at Gauss integration points and return max von Mises.

        For Tet4 this is equivalent to centroid evaluation (constant strain).
        For Tet10 this evaluates at all 4 Gauss points and returns the maximum,
        capturing peak stress in high-gradient zones more accurately.
        """
        e = int(elem_idx)
        dofs = self.elem_dofs[e]
        ue = np.asarray(u, dtype=np.float64)[dofs]
        elem = self.elements[e]
        scale = float(self.material.get_density_scale(float(np.asarray(rho, dtype=np.float64)[e]), penalty))

        if self.npe == 4:
            # Tet4: constant strain — single evaluation is exact
            return self.calculate_stress(u, elem_idx, rho, penalty)

        # Tet10: evaluate at all 5 Keast Gauss points (degree 3)
        coords = self.nodes[elem]
        gps = [
            (0.25, 0.25, 0.25),
            (1.0/6, 1.0/6, 1.0/6),
            (1.0/6, 1.0/6, 0.5),
            (1.0/6, 0.5, 1.0/6),
            (0.5, 1.0/6, 1.0/6),
        ]

        max_vm = 0.0
        max_stress = np.zeros(6, dtype=np.float64)

        for r, s, t in gps:
            bm, det_j = self._b_matrix_tet10(coords, r, s, t)
            if bm is None or abs(det_j) < 1e-14:
                continue

            strain = bm @ ue
            stress = (self._d_base * scale) @ strain

            sx, sy, sz = stress[0], stress[1], stress[2]
            txy, tyz, tzx = stress[3], stress[4], stress[5]
            vm = float(np.sqrt(0.5 * (
                (sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2
                + 6.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
            )))

            if vm > max_vm:
                max_vm = vm
                max_stress = stress.copy()

        return max_stress, max_vm

    def calculate_stress_all_gauss(self, u, rho, penalty=3.0):
        """Vectorized Gauss-point stress evaluation for ALL elements.

        Returns array of max von Mises stress per element.
        Much faster than calling calculate_stress_gauss in a loop.
        """
        uu = np.asarray(u, dtype=np.float64)
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        scale_all = self._element_modulus(rho_arr, penalty)
        vm_all = np.zeros(self.n_elements, dtype=np.float64)

        if self.npe == 4:
            # Tet4: fully vectorized constant-strain stress (10-50x faster)
            valid = np.abs(self._det_centroid) >= 1e-14
            ue_all = uu[self.elem_dofs]                                        # (n_elem, 12)
            strain_all = np.einsum('eij,ej->ei', self._bm_centroid, ue_all)    # (n_elem, 6)
            stress_all = np.einsum('ij,ej->ei', self._d_base, strain_all) * scale_all[:, None]  # (n_elem, 6)
            sx, sy, sz = stress_all[:, 0], stress_all[:, 1], stress_all[:, 2]
            txy, tyz, tzx = stress_all[:, 3], stress_all[:, 4], stress_all[:, 5]
            vm_all = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                                    + 6.0 * (txy**2 + tyz**2 + tzx**2)))
            vm_all[~valid] = 0.0
        else:
            # Tet10: evaluate at 5 Keast Gauss points (degree 3), keep max per element
            gps = [
                (0.25, 0.25, 0.25),
                (1.0/6, 1.0/6, 1.0/6),
                (1.0/6, 1.0/6, 0.5),
                (1.0/6, 0.5, 1.0/6),
                (0.5, 1.0/6, 1.0/6),
            ]

            for e in range(self.n_elements):
                ue = uu[self.elem_dofs[e]]
                coords = self.nodes[self.elements[e]]
                emod = scale_all[e]

                for r, s, t in gps:
                    bm, det_j = self._b_matrix_tet10(coords, r, s, t)
                    if bm is None or abs(det_j) < 1e-14:
                        continue
                    strain = bm @ ue
                    stress = (self._d_base * emod) @ strain
                    sx, sy, sz = stress[0], stress[1], stress[2]
                    txy, tyz, tzx = stress[3], stress[4], stress[5]
                    vm = float(np.sqrt(0.5 * (
                        (sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2
                        + 6.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
                    )))
                    if vm > vm_all[e]:
                        vm_all[e] = vm

        return vm_all

    def solve_multiload(self, rho, force_cases, fixed_dofs, penalty=3.0):
        """Solve for multiple load cases and return list of displacement vectors.

        Parameters
        ----------
        force_cases : list of np.ndarray
            Each element is a full-DOF force vector.
        fixed_dofs : np.ndarray
            DOFs with zero displacement (shared across all load cases).

        Returns
        -------
        list of np.ndarray
            Displacement vector for each load case.
        """
        k = self.assemble_global_stiffness(rho, penalty)
        free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
        k_ff = k[free_dofs][:, free_dofs].tocsr()

        mode = str(getattr(self, 'linear_solver_mode', 'auto')).lower()
        use_iterative = False
        if mode in ('iterative', 'cg', 'cg-amg', 'cg_amg'):
            use_iterative = True
        elif mode == 'auto':
            use_iterative = (k_ff.shape[0] >= int(getattr(self, 'iterative_solver_dof_threshold', 50000)))

        M = None
        if use_iterative and bool(getattr(self, 'use_pyamg_preconditioner', True)) and _HAS_PYAMG:
            try:
                ml = pyamg.smoothed_aggregation_solver(k_ff)
                M = ml.aspreconditioner()
            except Exception:
                M = None

        displacements = []
        for forces in force_cases:
            f_f = np.asarray(forces, dtype=np.float64)[free_dofs]

            if use_iterative:
                tol = float(getattr(self, 'iterative_solver_tol', 1e-8))
                maxiter = int(getattr(self, 'iterative_solver_maxiter', 2000))
                try:
                    u_f, info = cg(k_ff, f_f, tol=tol, maxiter=maxiter, M=M)
                except TypeError:
                    u_f, info = cg(k_ff, f_f, rtol=tol, atol=0.0, maxiter=maxiter, M=M)

                if info != 0 or not np.all(np.isfinite(u_f)):
                    u_f = spsolve(k_ff, f_f)
            else:
                u_f = spsolve(k_ff, f_f)

            u = np.zeros(self.n_dofs, dtype=np.float64)
            u[free_dofs] = u_f
            displacements.append(u)

        return displacements

    def calculate_compliance_multiload(self, displacements, rho, penalty=3.0, weights=None):
        """Calculate weighted compliance across multiple load cases.

        Parameters
        ----------
        displacements : list of np.ndarray
            Displacement vectors from solve_multiload.
        weights : list of float, optional
            Weight for each load case (defaults to equal weights).

        Returns
        -------
        float
            Total weighted compliance.
        """
        n_cases = len(displacements)
        if weights is None:
            weights = [1.0 / n_cases] * n_cases

        total = 0.0
        for u, w in zip(displacements, weights):
            total += w * self.calculate_compliance(u, rho, penalty)
        return total


print('3D FEA Solver module created (Tet4/Tet10, Gauss-pt stress, multi-load)')
