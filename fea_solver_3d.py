import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, cg

try:
    import pyamg
    _HAS_PYAMG = True
except Exception:
    pyamg = None
    _HAS_PYAMG = False

try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
    _HAS_CHOLMOD = True
except Exception:
    cholmod_cholesky = None
    _HAS_CHOLMOD = False


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
        self._precompute_assembly_map()

        self.linear_solver_mode = 'auto'
        self.iterative_solver_dof_threshold = 20000
        self.iterative_solver_tol = 1e-8
        self.iterative_solver_maxiter = 2000
        self.use_pyamg_preconditioner = True
        self._solver_notice_printed = False
        self._last_u = None  # Warm-start for iterative solver
        self._cholmod_factor = None  # Cached cholmod symbolic factorization
        self._cholmod_factor_shape = None
        self._cached_free_dofs = None  # Cached free DOF indices
        self._cached_free_dofs_key = None

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

    def _batch_b_tet10(self, elem_coords, r, s, t):
        """Vectorized B-matrix computation for all Tet10 elements at natural coords (r,s,t).

        Returns (B_all, det_J, valid) where B_all is (n_elem, 6, 30).
        """
        n_elem = elem_coords.shape[0]
        dN_nat = self._shape_derivatives_nat_tet10(r, s, t)  # (3, 10)
        J_all = np.einsum('ij,ejk->eik', dN_nat, elem_coords)  # (n_elem, 3, 3)
        det_J = np.linalg.det(J_all)
        valid = np.abs(det_J) >= 1e-14

        inv_J = np.zeros_like(J_all)
        if np.any(valid):
            inv_J[valid] = np.linalg.inv(J_all[valid])

        dN_xyz = np.einsum('eij,jk->eik', inv_J, dN_nat)  # (n_elem, 3, 10)

        B_all = np.zeros((n_elem, 6, 30), dtype=np.float64)
        for i in range(10):
            c = 3 * i
            B_all[:, 0, c]     = dN_xyz[:, 0, i]
            B_all[:, 1, c + 1] = dN_xyz[:, 1, i]
            B_all[:, 2, c + 2] = dN_xyz[:, 2, i]
            B_all[:, 3, c]     = dN_xyz[:, 1, i]
            B_all[:, 3, c + 1] = dN_xyz[:, 0, i]
            B_all[:, 4, c + 1] = dN_xyz[:, 2, i]
            B_all[:, 4, c + 2] = dN_xyz[:, 1, i]
            B_all[:, 5, c]     = dN_xyz[:, 2, i]
            B_all[:, 5, c + 2] = dN_xyz[:, 0, i]
        return B_all, det_J, valid

    def _batch_b_tet4(self, elem_coords):
        """Vectorized B-matrix computation for all Tet4 elements (centroid)."""
        n_elem = elem_coords.shape[0]
        dN_nat = self._shape_derivatives_nat_tet4()  # (3, 4)
        J_all = np.einsum('ij,ejk->eik', dN_nat, elem_coords)
        det_J = np.linalg.det(J_all)
        valid = np.abs(det_J) >= 1e-14

        inv_J = np.zeros_like(J_all)
        if np.any(valid):
            inv_J[valid] = np.linalg.inv(J_all[valid])

        dN_xyz = np.einsum('eij,jk->eik', inv_J, dN_nat)  # (n_elem, 3, 4)

        B_all = np.zeros((n_elem, 6, 12), dtype=np.float64)
        for i in range(4):
            c = 3 * i
            B_all[:, 0, c]     = dN_xyz[:, 0, i]
            B_all[:, 1, c + 1] = dN_xyz[:, 1, i]
            B_all[:, 2, c + 2] = dN_xyz[:, 2, i]
            B_all[:, 3, c]     = dN_xyz[:, 1, i]
            B_all[:, 3, c + 1] = dN_xyz[:, 0, i]
            B_all[:, 4, c + 1] = dN_xyz[:, 2, i]
            B_all[:, 4, c + 2] = dN_xyz[:, 1, i]
            B_all[:, 5, c]     = dN_xyz[:, 2, i]
            B_all[:, 5, c + 2] = dN_xyz[:, 0, i]
        return B_all, det_J, valid

    def _compute_unit_element_stiffness(self, elem_idx):
        """Per-element stiffness (kept as fallback for single-element queries)."""
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
        gps = [
            (0.25, 0.25, 0.25), (1.0/6, 1.0/6, 1.0/6),
            (1.0/6, 1.0/6, 0.5), (1.0/6, 0.5, 1.0/6), (0.5, 1.0/6, 1.0/6),
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

        Both Tet4 and Tet10 are fully vectorized using batched NumPy operations.
        """
        ke_unit = np.zeros((self.n_elements, self.dof_per_elem, self.dof_per_elem), dtype=np.float64)
        eye_fallback = np.eye(self.dof_per_elem, dtype=np.float64)[None, :, :] * 1e-12

        if self.npe == 4:
            B_all, det_J, valid = self._batch_b_tet4(self.nodes[self.elements[:, :4]])
            DB = np.einsum('ij,ejk->eik', self._d_base, B_all)
            ke_batch = np.einsum('eji,ejk->eik', B_all, DB)
            ke_batch *= (np.abs(det_J) / 6.0)[:, None, None]
            ke_unit[valid] = ke_batch[valid]
            if np.any(~valid):
                ke_unit[~valid] = eye_fallback
        else:
            # Vectorized Tet10: batch all elements per Gauss point
            elem_coords = self.nodes[self.elements]  # (n_elem, 10, 3)
            gps = [
                (0.25, 0.25, 0.25), (1.0/6, 1.0/6, 1.0/6),
                (1.0/6, 1.0/6, 0.5), (1.0/6, 0.5, 1.0/6), (0.5, 1.0/6, 1.0/6),
            ]
            gw = [-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0]
            any_valid = np.zeros(self.n_elements, dtype=bool)

            for (r, s, t), w in zip(gps, gw):
                B_gp, det_J, valid = self._batch_b_tet10(elem_coords, r, s, t)
                any_valid |= valid
                DB = np.einsum('ij,ejk->eik', self._d_base, B_gp)
                ke_gp = np.einsum('eji,ejk->eik', B_gp, DB)
                ke_gp *= (np.abs(det_J) * w)[:, None, None]
                ke_unit += ke_gp

            if np.any(~any_valid):
                ke_unit[~any_valid] = eye_fallback

        bad = ~np.all(np.isfinite(ke_unit), axis=(1, 2))
        if np.any(bad):
            ke_unit[bad] = eye_fallback
        return ke_unit

    def _precompute_centroid_b(self):
        """Vectorized centroid B-matrix precomputation for all elements."""
        bm_all = np.zeros((self.n_elements, 6, self.dof_per_elem), dtype=np.float64)
        det_all = np.zeros(self.n_elements, dtype=np.float64)

        if self.npe == 4:
            B_all, det_J, valid = self._batch_b_tet4(self.nodes[self.elements[:, :4]])
        else:
            B_all, det_J, valid = self._batch_b_tet10(self.nodes[self.elements], 0.25, 0.25, 0.25)

        bm_all[valid] = B_all[valid]
        det_all[valid] = det_J[valid]
        return bm_all, det_all

    def _element_modulus(self, rho, penalty):
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        return np.array([self.material.get_density_scale(float(r), penalty) for r in rho_arr], dtype=np.float64)

    def get_element_stiffness(self, elem_idx, rho, penalty=3.0):
        scale = float(self.material.get_density_scale(float(np.asarray(rho, dtype=np.float64)[int(elem_idx)]), penalty))
        return self._ke_unit[int(elem_idx)] * scale

    def _precompute_assembly_map(self):
        """Precompute CSC sparsity structure and scatter map for O(n) assembly.

        Called once at init. Eliminates expensive COO->CSR->CSC conversion
        that was taking ~80s per iteration.
        """
        rows = self._row_ind
        cols = self._col_ind
        n_coo = len(rows)

        # Sort by (col, row) for CSC column-major ordering
        sort_order = np.lexsort((rows, cols))
        sorted_rows = rows[sort_order]
        sorted_cols = cols[sort_order]

        # Find unique (col, row) pairs
        pair_keys = sorted_cols.astype(np.int64) * np.int64(self.n_dofs) + sorted_rows.astype(np.int64)
        _, first_occ, inverse_sorted = np.unique(pair_keys, return_index=True, return_inverse=True)

        nnz = len(first_occ)
        u_rows = sorted_rows[first_occ].astype(np.int32)
        u_cols = sorted_cols[first_occ]

        # Build CSC indptr from column counts
        col_counts = np.bincount(u_cols, minlength=self.n_dofs)
        indptr = np.zeros(self.n_dofs + 1, dtype=np.int64)
        np.cumsum(col_counts, out=indptr[1:])

        # Scatter map: original COO index -> CSC data slot
        self._asm_scatter = np.empty(n_coo, dtype=np.int64)
        self._asm_scatter[sort_order] = inverse_sorted
        self._asm_csc_indptr = indptr
        self._asm_csc_indices = u_rows
        self._asm_nnz = nnz

    def assemble_global_stiffness(self, rho, penalty=3.0):
        scale = self._element_modulus(rho, penalty)
        coo_data = (self._ke_unit.reshape(self.n_elements, -1) * scale[:, None]).reshape(-1)
        csc_data = np.bincount(self._asm_scatter, weights=coo_data, minlength=self._asm_nnz)
        return csc_matrix(
            (csc_data, self._asm_csc_indices, self._asm_csc_indptr),
            shape=(self.n_dofs, self.n_dofs), copy=False
        )

    def _direct_solve(self, k_ff, f_f):
        """Direct solve using cholmod (preferred) or spsolve (fallback).

        Caches the cholmod symbolic factorization so subsequent solves with
        the same sparsity pattern only redo the numeric factorization.
        """
        if _HAS_CHOLMOD:
            try:
                k_csc = k_ff if hasattr(k_ff, 'format') and k_ff.format == 'csc' else k_ff.tocsc()
                # Reuse symbolic factorization if sparsity pattern hasn't changed
                if (self._cholmod_factor is not None and
                        self._cholmod_factor_shape == k_csc.shape):
                    try:
                        self._cholmod_factor.cholesky_inplace(k_csc)
                    except Exception:
                        self._cholmod_factor = cholmod_cholesky(k_csc)
                        self._cholmod_factor_shape = k_csc.shape
                else:
                    self._cholmod_factor = cholmod_cholesky(k_csc)
                    self._cholmod_factor_shape = k_csc.shape
                u_f = self._cholmod_factor(f_f)
                if np.all(np.isfinite(u_f)):
                    if not self._solver_notice_printed:
                        print('  [SOLVER] Using CHOLMOD direct solver (cached symbolic factorization).')
                    return u_f
            except Exception:
                self._cholmod_factor = None
                self._cholmod_factor_shape = None
        return spsolve(k_ff, f_f)

    def _apply_bc_penalty(self, k_csc, forces, fixed_dofs):
        """Apply boundary conditions via penalty method on the full matrix.

        Much faster than submatrix extraction k[free][:, free] which is O(nnz).
        Adds large diagonal values at fixed DOFs to enforce u=0.
        """
        f = np.asarray(forces, dtype=np.float64).copy()
        fixed = np.asarray(fixed_dofs, dtype=np.int64)
        if fixed.size > 0:
            diag_vals = np.abs(k_csc.diagonal())
            big = float(np.max(diag_vals[diag_vals > 0])) * 1e8 if np.any(diag_vals > 0) else 1e20
            f[fixed] = 0.0
            # Add penalty to diagonal of fixed DOFs
            penalty_data = np.full(fixed.size, big, dtype=np.float64)
            penalty_mat = coo_matrix(
                (penalty_data, (fixed, fixed)),
                shape=(self.n_dofs, self.n_dofs)
            ).tocsc()
            k_bc = k_csc + penalty_mat
        else:
            k_bc = k_csc
        return k_bc, f

    def solve(self, rho, forces, fixed_dofs, penalty=3.0):
        k = self.assemble_global_stiffness(rho, penalty)

        # Cache fixed_dofs key for cholmod invalidation
        fd_key = (fixed_dofs.tobytes() if hasattr(fixed_dofs, 'tobytes') else bytes(fixed_dofs))
        if self._cached_free_dofs_key != fd_key:
            self._cached_free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
            self._cached_free_dofs_key = fd_key
            self._cholmod_factor = None
            self._cholmod_factor_shape = None

        mode = str(getattr(self, 'linear_solver_mode', 'auto')).lower()
        use_iterative = False
        n_free = len(self._cached_free_dofs)
        if mode in ('iterative', 'cg', 'cg-amg', 'cg_amg'):
            use_iterative = True
        elif mode == 'auto':
            use_iterative = (n_free >= int(getattr(self, 'iterative_solver_dof_threshold', 50000)))

        if use_iterative:
            # Apply BCs via penalty method (avoids slow submatrix extraction)
            k_bc, f_bc = self._apply_bc_penalty(k, forces, fixed_dofs)
            k_bc_csr = k_bc.tocsr()

            M = None
            if bool(getattr(self, 'use_pyamg_preconditioner', True)) and _HAS_PYAMG:
                try:
                    ml = pyamg.smoothed_aggregation_solver(k_bc_csr)
                    M = ml.aspreconditioner()
                except Exception:
                    M = None

            x0 = self._last_u if self._last_u is not None else None
            if x0 is not None and (x0.shape[0] != f_bc.shape[0] or not np.all(np.isfinite(x0))):
                x0 = None

            tol = float(getattr(self, 'iterative_solver_tol', 1e-8))
            maxiter = int(getattr(self, 'iterative_solver_maxiter', 2000))
            try:
                u, info = cg(k_bc_csr, f_bc, x0=x0, tol=tol, maxiter=maxiter, M=M)
            except TypeError:
                u, info = cg(k_bc_csr, f_bc, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter, M=M)

            if info == 0 and np.all(np.isfinite(u)):
                if not self._solver_notice_printed:
                    msg = 'CG+AMG' if (M is not None) else 'CG'
                    ws = ' (warm-started)' if x0 is not None else ''
                    print(f'  [SOLVER] Using iterative {msg} with penalty BCs{ws}.')
                    self._solver_notice_printed = True
            else:
                if not self._solver_notice_printed:
                    print('  [SOLVER] Iterative solve did not converge, falling back to direct.')
                    self._solver_notice_printed = True
                u = self._direct_solve(k_bc, f_bc)
        else:
            # Direct path: penalty method avoids expensive submatrix extraction
            k_bc, f_bc = self._apply_bc_penalty(k.tocsc(), forces, fixed_dofs)
            u = self._direct_solve(k_bc, f_bc)
            if not self._solver_notice_printed:
                print('  [SOLVER] Using penalty-method BC application (no submatrix extraction).')

        self._last_u = u.copy()
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
            # Vectorized Tet10: batch all elements per Gauss point, keep max VM
            elem_coords = self.nodes[self.elements]  # (n_elem, 10, 3)
            ue_all = uu[self.elem_dofs]  # (n_elem, 30)
            gps = [
                (0.25, 0.25, 0.25), (1.0/6, 1.0/6, 1.0/6),
                (1.0/6, 1.0/6, 0.5), (1.0/6, 0.5, 1.0/6), (0.5, 1.0/6, 1.0/6),
            ]
            for r, s, t in gps:
                B_gp, det_J, valid = self._batch_b_tet10(elem_coords, r, s, t)
                strain = np.einsum('eij,ej->ei', B_gp, ue_all)  # (n_elem, 6)
                stress = np.einsum('ij,ej->ei', self._d_base, strain) * scale_all[:, None]
                sx, sy, sz = stress[:, 0], stress[:, 1], stress[:, 2]
                txy, tyz, tzx = stress[:, 3], stress[:, 4], stress[:, 5]
                vm_gp = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                                       + 6.0 * (txy**2 + tyz**2 + tzx**2)))
                vm_gp[~valid] = 0.0
                vm_all = np.maximum(vm_all, vm_gp)

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
