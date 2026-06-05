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
        self._bc_locked = False  # Lock BC after first solve (BCs don't change during optimization)

        # §4.1: Precomputed free-DOF CSC scatter map for O(nnz) submatrix extraction
        self._free_dof_map_ready = False
        self._free_src_indices = None    # Indices into full CSC data for free-free entries
        self._free_csc_indices = None    # Row indices of the free-DOF CSC
        self._free_csc_indptr = None     # Column pointers of the free-DOF CSC
        self._free_n = 0                 # Number of free DOFs
        self._free_dofs_arr = None       # Sorted array of free DOF indices

        # §4.2: Cached AMG preconditioner (hierarchy reuse across iterations)
        self._amg_M = None               # Cached preconditioner LinearOperator
        self._amg_base_cg_iters = None   # CG iterations on first use (baseline)
        self._amg_rebuild_counter = 0    # Iterations since last AMG rebuild
        self._amg_rebuild_interval = 25  # Rebuild every N iterations as safety net
        self._amg_nullspace = None       # Near-nullspace vectors for 3D elasticity

        # Precompute Gauss-point B-matrices for Tet10 (§3.4)
        self._gp_b_cache = None
        if self.npe == 10:
            self._precompute_gauss_point_b()

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
        # 4-point all-positive Gauss rule (degree 2, weight sum = 1/6).
        # Replaces the 5-point Keast rule that had a negative centroid weight
        # (-2/15) which could cause non-positive-definite ke on distorted elements.
        a1, a2 = 0.1381966011250105, 0.5854101966249685
        gps = [(a2, a1, a1), (a1, a2, a1), (a1, a1, a2), (a1, a1, a1)]
        gw = [1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0]

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
            # 4-point all-positive Gauss rule (degree 2, weight sum = 1/6)
            a1, a2 = 0.1381966011250105, 0.5854101966249685
            gps = [(a2, a1, a1), (a1, a2, a1), (a1, a1, a2), (a1, a1, a1)]
            gw = [1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0]
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

    def _precompute_gauss_point_b(self):
        """Cache B-matrices at all 4 Gauss points for Tet10 elements (§3.4).

        Eliminates repeated Jacobian inverse computation during
        calculate_stress_all_gauss, which is called every stress evaluation
        iteration. Memory cost: ~5.6 MB per 1000 elements.
        """
        if self.npe != 10:
            self._gp_b_cache = None
            return

        elem_coords = self.nodes[self.elements]  # (n_elem, 10, 3)
        a1, a2 = 0.1381966011250105, 0.5854101966249685
        gps = [(a2, a1, a1), (a1, a2, a1), (a1, a1, a2), (a1, a1, a1)]

        self._gp_b_cache = []
        for r, s, t in gps:
            B_gp, det_J, valid = self._batch_b_tet10(elem_coords, r, s, t)
            self._gp_b_cache.append((B_gp.copy(), valid.copy()))

        mem_mb = 4 * self.n_elements * 6 * self.dof_per_elem * 8 / (1024**2)
        print(f'  [FEA] Cached Tet10 Gauss-point B-matrices: {mem_mb:.0f} MB for {self.n_elements} elements')

    def _element_modulus(self, rho, penalty):
        """Vectorized SIMP density-stiffness scaling for all elements.

        Equivalent to calling material.get_density_scale(r, p) per element,
        but ~100x faster via batched NumPy: rho_min + (1 - rho_min) * rho^p.
        """
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        rho_min = float(self.material.rho_min)
        return rho_min + (1.0 - rho_min) * (rho_arr ** float(penalty))

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

    def assemble_global_stiffness_from_scale(self, scale):
        """Assemble global stiffness matrix directly from element scale factors."""
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        coo_data = (self._ke_unit.reshape(self.n_elements, -1) * scale[:, None]).reshape(-1)
        csc_data = np.bincount(self._asm_scatter, weights=coo_data, minlength=self._asm_nnz)
        return csc_matrix(
            (csc_data, self._asm_csc_indices, self._asm_csc_indptr),
            shape=(self.n_dofs, self.n_dofs), copy=False
        )

    def assemble_global_stiffness(self, rho, penalty=3.0):
        scale = self._element_modulus(rho, penalty)
        return self.assemble_global_stiffness_from_scale(scale)

    # ------------------------------------------------------------------ #
    # §4.1  Free-DOF CSC scatter map (precomputed, O(nnz) extraction)    #
    # ------------------------------------------------------------------ #

    def _build_free_dof_csc_map(self, fixed_dofs):
        """Precompute a scatter map from full CSC data → free-DOF CSC data.

        Called once when BCs are first established. Subsequent calls to
        _extract_free_dof_matrix() become a single indexed-copy, avoiding
        the expensive k.tocsr()[free][:, free].tocsr() path which does
        3–4 sparse format conversions per iteration.
        """
        import time as _time
        t0 = _time.perf_counter()

        fixed = np.asarray(fixed_dofs, dtype=np.int64)
        is_free = np.ones(self.n_dofs, dtype=bool)
        if fixed.size > 0:
            is_free[fixed] = False
        free = np.where(is_free)[0].astype(np.int64)
        n_free = free.size

        # Global DOF → free-DOF index (-1 if fixed)
        g2f = np.full(self.n_dofs, -1, dtype=np.int64)
        g2f[free] = np.arange(n_free, dtype=np.int64)

        # Expand CSC column indices for every nonzero entry (vectorized)
        indptr = self._asm_csc_indptr
        row_indices = self._asm_csc_indices.astype(np.int64)
        nnz = row_indices.size
        col_for_entry = np.repeat(np.arange(self.n_dofs, dtype=np.int64), np.diff(indptr))

        # Keep entries where both row and col are free
        mask = is_free[row_indices] & is_free[col_for_entry]
        src_indices = np.where(mask)[0]

        # Map to free-DOF row and column indices
        free_rows = g2f[row_indices[src_indices]]
        free_cols = g2f[col_for_entry[src_indices]]

        # Build free-DOF CSC indptr (entries are already in column-major order)
        col_counts = np.bincount(free_cols, minlength=n_free)
        dst_indptr = np.zeros(n_free + 1, dtype=np.int64)
        np.cumsum(col_counts, out=dst_indptr[1:])

        self._free_src_indices = src_indices.astype(np.int64)
        self._free_csc_indices = free_rows.astype(np.int32)
        self._free_csc_indptr = dst_indptr
        self._free_n = n_free
        self._free_dofs_arr = free
        self._free_dof_map_ready = True

        t1 = _time.perf_counter()
        kept = src_indices.size
        print(f'  [SOLVER] Free-DOF scatter map built: {n_free} free DOFs, '
              f'{kept}/{nnz} entries kept ({100.0*kept/max(nnz,1):.0f}%), '
              f'{t1-t0:.2f}s')

    def _extract_free_dof_matrix(self, k_csc):
        """Extract free-DOF submatrix in O(nnz_free) using precomputed scatter map.

        Returns a CSC matrix of shape (n_free, n_free).
        """
        free_data = k_csc.data[self._free_src_indices]
        return csc_matrix(
            (free_data, self._free_csc_indices, self._free_csc_indptr),
            shape=(self._free_n, self._free_n), copy=False
        )

    # ------------------------------------------------------------------ #
    # §4.3  Near-nullspace for 3D elasticity AMG                         #
    # ------------------------------------------------------------------ #

    def _build_elasticity_nullspace(self, free_dofs):
        """Build 6-vector near-nullspace (rigid body modes) for AMG.

        For 3D elasticity, the near-nullspace consists of 3 translations
        and 3 infinitesimal rotations. Providing these to AMG enables
        proper coarsening of the vector-valued displacement field,
        dramatically reducing CG iteration count (often 3–5× fewer).
        """
        free = np.asarray(free_dofs, dtype=np.int64)
        n_free = free.size
        node_idx = free // 3
        local_dof = free % 3  # 0=x, 1=y, 2=z

        coords = self.nodes[node_idx]  # (n_free, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        B = np.zeros((n_free, 6), dtype=np.float64)
        # Translation modes
        B[local_dof == 0, 0] = 1.0  # Tx
        B[local_dof == 1, 1] = 1.0  # Ty
        B[local_dof == 2, 2] = 1.0  # Tz
        # Rotation about x-axis: (0, -z, y)
        B[local_dof == 1, 3] = -z[local_dof == 1]
        B[local_dof == 2, 3] =  y[local_dof == 2]
        # Rotation about y-axis: (z, 0, -x)
        B[local_dof == 0, 4] =  z[local_dof == 0]
        B[local_dof == 2, 4] = -x[local_dof == 2]
        # Rotation about z-axis: (-y, x, 0)
        B[local_dof == 0, 5] = -y[local_dof == 0]
        B[local_dof == 1, 5] =  x[local_dof == 1]
        return B

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
        import time as _time
        t_start = _time.perf_counter()

        k = self.assemble_global_stiffness(rho, penalty)
        t_asm = _time.perf_counter()

        # Cache fixed_dofs key for cholmod invalidation
        # §3.5: Once BCs are established (first solve), lock them — during
        # topology optimization fixed_dofs never change, only K values do.
        # This avoids re-dropping the CHOLMOD symbolic factorization.
        if not self._bc_locked:
            fd_key = (fixed_dofs.tobytes() if hasattr(fixed_dofs, 'tobytes') else bytes(fixed_dofs))
            if self._cached_free_dofs_key != fd_key:
                self._cached_free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
                self._cached_free_dofs_key = fd_key
                self._cholmod_factor = None
                self._cholmod_factor_shape = None
                # Build free-DOF scatter map and nullspace for iterative path
                self._build_free_dof_csc_map(fixed_dofs)
                self._amg_nullspace = self._build_elasticity_nullspace(self._free_dofs_arr)
                # Invalidate cached AMG
                self._amg_M = None
                self._amg_base_cg_iters = None
                self._amg_rebuild_counter = 0
            self._bc_locked = True  # Lock after first successful setup

        mode = str(getattr(self, 'linear_solver_mode', 'auto')).lower()
        use_iterative = False
        n_free = len(self._cached_free_dofs)
        if mode in ('iterative', 'cg', 'cg-amg', 'cg_amg'):
            use_iterative = True
        elif mode == 'auto':
            use_iterative = (n_free >= int(getattr(self, 'iterative_solver_dof_threshold', 50000)))

        if use_iterative:
            # §4.1: Extract free-DOF submatrix using precomputed scatter map.
            # O(nnz_free) indexed copy instead of k.tocsr()[free][:, free] which
            # does 3–4 sparse format conversions.
            free = self._free_dofs_arr if self._free_dof_map_ready else self._cached_free_dofs
            if self._free_dof_map_ready:
                k_ff = self._extract_free_dof_matrix(k)
            else:
                k_ff = k.tocsr()[free][:, free].tocsr()
            f_f = np.asarray(forces, dtype=np.float64)[free]
            t_bc = _time.perf_counter()

            # §4.2: AMG preconditioner with hierarchy caching.
            # AMG setup (coarsening + interpolation) is expensive (~30-50% of solve).
            # Since sparsity pattern is constant during optimization, the hierarchy
            # is reused across iterations. Rebuild only when CG convergence degrades
            # or after a fixed interval.
            M = None
            need_amg_rebuild = False
            if bool(getattr(self, 'use_pyamg_preconditioner', True)) and _HAS_PYAMG:
                self._amg_rebuild_counter += 1
                if self._amg_M is None:
                    need_amg_rebuild = True
                elif self._amg_rebuild_counter >= self._amg_rebuild_interval:
                    need_amg_rebuild = True

                if need_amg_rebuild:
                    try:
                        k_ff_csr = k_ff.tocsr() if k_ff.format != 'csr' else k_ff
                        # §4.3: Near-nullspace (rigid body modes) gives AMG proper
                        # vector-valued coarsening for 3D elasticity — typically
                        # reduces CG iterations by 3–5×.
                        B_ns = self._amg_nullspace
                        ml = pyamg.smoothed_aggregation_solver(k_ff_csr, B=B_ns)
                        self._amg_M = ml.aspreconditioner()
                        self._amg_rebuild_counter = 0
                    except Exception:
                        self._amg_M = None
                M = self._amg_M

            x0 = None
            if self._last_u is not None:
                x0_full = self._last_u
                if x0_full.shape[0] == self.n_dofs and np.all(np.isfinite(x0_full)):
                    x0 = x0_full[free]

            tol = float(getattr(self, 'iterative_solver_tol', 1e-8))
            maxiter = int(getattr(self, 'iterative_solver_maxiter', 2000))

            # Track CG iteration count to detect preconditioner staleness
            cg_iters = [0]
            def _cg_counter(x):
                cg_iters[0] += 1

            try:
                u_f, info = cg(k_ff, f_f, x0=x0, tol=tol, maxiter=maxiter, M=M,
                               callback=_cg_counter)
            except TypeError:
                u_f, info = cg(k_ff, f_f, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter,
                               M=M, callback=_cg_counter)

            # §4.2b: Adaptive AMG rebuild — if CG iterations grow > 3× baseline,
            # the stale preconditioner is hurting. Rebuild on next iteration.
            if info == 0 and cg_iters[0] > 0:
                if self._amg_base_cg_iters is None:
                    self._amg_base_cg_iters = cg_iters[0]
                elif cg_iters[0] > 3 * max(self._amg_base_cg_iters, 10):
                    self._amg_M = None  # Force rebuild on next iteration
                    self._amg_base_cg_iters = None

            if info == 0 and np.all(np.isfinite(u_f)):
                u = np.zeros(self.n_dofs, dtype=np.float64)
                u[free] = u_f
                if not self._solver_notice_printed:
                    msg = 'CG+AMG' if (M is not None) else 'CG'
                    ws = ' (warm-started)' if x0 is not None else ''
                    scatter_msg = ' (scatter-map)' if self._free_dof_map_ready else ''
                    print(f'  [SOLVER] Using iterative {msg}{scatter_msg}{ws}, '
                          f'{cg_iters[0]} CG iters.')
                    self._solver_notice_printed = True
            else:
                if not self._solver_notice_printed:
                    print(f'  [SOLVER] Iterative solve did not converge '
                          f'({cg_iters[0]} iters), falling back to direct.')
                    self._solver_notice_printed = True
                # Direct fallback: use penalty method (fast for CHOLMOD/spsolve)
                k_bc, f_bc = self._apply_bc_penalty(k.tocsc(), forces, fixed_dofs)
                u = self._direct_solve(k_bc, f_bc)
        else:
            # Direct path: penalty method avoids expensive submatrix extraction
            k_bc, f_bc = self._apply_bc_penalty(k.tocsc(), forces, fixed_dofs)
            t_bc = _time.perf_counter()
            u = self._direct_solve(k_bc, f_bc)
            if not self._solver_notice_printed:
                print('  [SOLVER] Using penalty-method BC application (no submatrix extraction).')

        t_solve = _time.perf_counter()
        if not self._solver_notice_printed:
            print(f'  [SOLVER] Timing: assembly={t_asm-t_start:.2f}s, BC+precond={t_bc-t_asm:.2f}s, solve={t_solve-t_bc:.2f}s, total={t_solve-t_start:.2f}s')
            self._solver_notice_printed = True

        self._last_u = u.copy()
        return u

    def solve_with_modulus(self, scale, forces, fixed_dofs):
        """Solve FEA system using direct element stiffness scales instead of SIMP."""
        import time as _time
        t_start = _time.perf_counter()

        k = self.assemble_global_stiffness_from_scale(scale)
        t_asm = _time.perf_counter()

        if not self._bc_locked:
            fd_key = (fixed_dofs.tobytes() if hasattr(fixed_dofs, 'tobytes') else bytes(fixed_dofs))
            if self._cached_free_dofs_key != fd_key:
                self._cached_free_dofs = np.setdiff1d(np.arange(self.n_dofs), fixed_dofs)
                self._cached_free_dofs_key = fd_key
                self._cholmod_factor = None
                self._cholmod_factor_shape = None
                self._build_free_dof_csc_map(fixed_dofs)
                self._amg_nullspace = self._build_elasticity_nullspace(self._free_dofs_arr)
                self._amg_M = None
                self._amg_base_cg_iters = None
                self._amg_rebuild_counter = 0

        self._bc_locked = True
        k_ff = self._extract_free_dof_matrix(k)
        t_bc = _time.perf_counter()

        f_f = np.asarray(forces, dtype=np.float64)[self._free_dofs_arr]
        u_f = self._direct_solve(k_ff, f_f)
        t_solve = _time.perf_counter()

        u = np.zeros(self.n_dofs, dtype=np.float64)
        u[self._free_dofs_arr] = u_f

        if not self._solver_notice_printed:
            print(f'  [SOLVER] Timing: assembly={t_asm-t_start:.2f}s, BC+precond={t_bc-t_asm:.2f}s, solve={t_solve-t_bc:.2f}s, total={t_solve-t_start:.2f}s')
            self._solver_notice_printed = True

        return u

    def calculate_compliance(self, u, rho, penalty=3.0):
        scale = self._element_modulus(rho, penalty)
        uu = np.asarray(u, dtype=np.float64)
        ue = uu[self.elem_dofs]
        ce_unit = np.einsum('ni,nij,nj->n', ue, self._ke_unit, ue)
        return float(np.sum(scale * ce_unit))

    def calculate_compliance_with_modulus(self, u, scale):
        """Calculate compliance using direct element stiffness scales."""
        uu = np.asarray(u, dtype=np.float64)
        ue = uu[self.elem_dofs]
        ce_unit = np.einsum('ni,nij,nj->n', ue, self._ke_unit, ue)
        return float(np.sum(np.asarray(scale, dtype=np.float64).reshape(-1) * ce_unit))

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

        # Tet10: evaluate at all 4 Gauss points (degree 2, all-positive weights)
        coords = self.nodes[elem]
        a1, a2 = 0.1381966011250105, 0.5854101966249685
        gps = [(a2, a1, a1), (a1, a2, a1), (a1, a1, a2), (a1, a1, a1)]

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
            # Vectorized Tet10: use precomputed Gauss-point B-matrices if available
            ue_all = uu[self.elem_dofs]  # (n_elem, 30)
            if self._gp_b_cache is not None:
                for B_gp, valid in self._gp_b_cache:
                    strain = np.einsum('eij,ej->ei', B_gp, ue_all)
                    stress = np.einsum('ij,ej->ei', self._d_base, strain) * scale_all[:, None]
                    sx, sy, sz = stress[:, 0], stress[:, 1], stress[:, 2]
                    txy, tyz, tzx = stress[:, 3], stress[:, 4], stress[:, 5]
                    vm_gp = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                                           + 6.0 * (txy**2 + tyz**2 + tzx**2)))
                    vm_gp[~valid] = 0.0
                    vm_all = np.maximum(vm_all, vm_gp)
            else:
                elem_coords = self.nodes[self.elements]  # (n_elem, 10, 3)
                a1, a2 = 0.1381966011250105, 0.5854101966249685
                gps = [(a2, a1, a1), (a1, a2, a1), (a1, a1, a2), (a1, a1, a1)]
                for r, s, t in gps:
                    B_gp, det_J, valid = self._batch_b_tet10(elem_coords, r, s, t)
                    strain = np.einsum('eij,ej->ei', B_gp, ue_all)
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


def distribute_surface_traction(nodes, face_nodes, total_force_vec):
    """Distribute a total force as consistent nodal forces via area-weighted traction.

    Instead of splitting force equally among selected nodes (which creates
    stress singularities), this distributes force proportional to the
    tributary area of each node on the loaded faces.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 3)
        Node coordinates.
    face_nodes : list of array-like
        Each entry is a list/array of 3 node indices forming a triangular face.
    total_force_vec : ndarray (3,)
        Total force vector [Fx, Fy, Fz] to distribute.

    Returns
    -------
    forces : ndarray (n_nodes*3,)
        Global force vector with consistent nodal forces.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    f_total = np.asarray(total_force_vec, dtype=np.float64).ravel()[:3]
    n_dofs = pts.shape[0] * 3
    forces = np.zeros(n_dofs, dtype=np.float64)

    # Compute tributary area per node
    node_areas = np.zeros(pts.shape[0], dtype=np.float64)
    total_area = 0.0
    for face in face_nodes:
        fn = np.asarray(face, dtype=np.int64)[:3]
        v1 = pts[fn[1]] - pts[fn[0]]
        v2 = pts[fn[2]] - pts[fn[0]]
        area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        total_area += area
        # Each vertex of a triangle gets 1/3 of the face area
        for ni in fn:
            node_areas[ni] += area / 3.0

    if total_area < 1e-30:
        # Fallback: equal distribution
        unique_nodes = np.unique(np.concatenate([np.asarray(f, dtype=np.int64)[:3] for f in face_nodes]))
        if unique_nodes.size > 0:
            f_per_node = f_total / unique_nodes.size
            for ni in unique_nodes:
                forces[3*ni:3*ni+3] = f_per_node
        return forces

    # Distribute force proportional to tributary area
    loaded_nodes = np.where(node_areas > 1e-30)[0]
    for ni in loaded_nodes:
        frac = node_areas[ni] / total_area
        forces[3*ni:3*ni+3] = f_total * frac

    return forces
