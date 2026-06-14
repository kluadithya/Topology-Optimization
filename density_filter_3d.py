import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


_TET_EDGE_PAIRS = np.asarray([
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
], dtype=np.int32)


def estimate_characteristic_tet_edge_length(nodes, elements, max_sample_elements=20000):
    """Estimate characteristic tetra edge length using median sampled corner-edge lengths."""
    nn = np.asarray(nodes, dtype=np.float64)
    ee = np.asarray(elements, dtype=np.int64)

    if ee.ndim != 2 or ee.shape[0] == 0 or ee.shape[1] < 4 or nn.shape[0] == 0:
        return 1.0

    n_elem = int(ee.shape[0])
    if n_elem > int(max_sample_elements):
        rng = np.random.default_rng(12345)
        idx = rng.choice(n_elem, size=int(max_sample_elements), replace=False)
        tet = ee[idx, :4]
    else:
        tet = ee[:, :4]

    pts = nn[tet]
    diffs = pts[:, _TET_EDGE_PAIRS[:, 0], :] - pts[:, _TET_EDGE_PAIRS[:, 1], :]
    edge_lengths = np.linalg.norm(diffs, axis=2).reshape(-1)

    valid = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1e-12)]
    if valid.size > 0:
        return float(np.median(valid))

    bb_min = np.min(nn, axis=0)
    bb_max = np.max(nn, axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))
    if diag <= 1e-12:
        return 1.0
    return max(diag / max(n_elem ** (1.0 / 3.0), 1.0), 1e-9)


def auto_filter_radius_from_mesh(
    nodes,
    elements,
    factor=3.5,
    min_factor=2.0,
    max_factor=4.0,
    min_dim_ratio=0.06,
    max_dim_ratio=0.18,
):
    """Compute robust filter radius from element size with geometric floor/ceiling guards."""
    h = estimate_characteristic_tet_edge_length(nodes, elements)
    fac = float(np.clip(float(factor), float(min_factor), float(max_factor)))
    r = fac * h

    nn = np.asarray(nodes, dtype=np.float64)
    if nn.ndim == 2 and nn.shape[0] > 0:
        ext = np.max(nn, axis=0) - np.min(nn, axis=0)
        pos_ext = ext[ext > 1e-12]
        if pos_ext.size > 0:
            min_dim = float(np.min(pos_ext))
            r = max(r, float(min_dim_ratio) * min_dim)
            r = min(r, float(max_dim_ratio) * min_dim)

    return float(max(r, 1e-9)), h


class DensityFilter3D:
    """Distance-weighted neighborhood filter on element centroids for tetrahedral meshes.

    Uses a precomputed sparse CSR weight matrix for O(nnz) BLAS filter application
    instead of per-element Python loops — typically 20–100× faster.
    """

    def __init__(self, nodes, elements, radius):
        self.radius = float(max(radius, 1e-9))
        self.centroids = np.mean(nodes[elements], axis=1)
        n = len(self.centroids)

        # Build neighbor/weight lists (still needed by MinimumMemberSizeProjection3D)
        self.neighbors = []
        self.weights = []

        # Accumulate triplets for sparse matrix construction
        rows, cols, data = [], [], []

        tree = cKDTree(self.centroids)
        for i, c in enumerate(self.centroids):
            idx = tree.query_ball_point(c, self.radius)
            if not idx:
                idx = [i]
            idx_arr = np.asarray(idx, dtype=np.int32)
            d = np.linalg.norm(self.centroids[idx_arr] - c, axis=1)
            w = np.maximum(0.0, self.radius - d)
            s = float(np.sum(w))
            if s <= 1e-16:
                idx_arr = np.asarray([i], dtype=np.int32)
                w = np.asarray([1.0], dtype=np.float64)
                s = 1.0
            w_norm = (w / s).astype(np.float64)
            self.neighbors.append(idx_arr)
            self.weights.append(w_norm)

            # Add to sparse matrix triplets
            rows.extend([i] * len(idx_arr))
            cols.extend(idx_arr.tolist())
            data.extend(w_norm.tolist())

        # Build sparse weight matrix W (n_elem × n_elem)
        self._W = csr_matrix((data, (rows, cols)), shape=(n, n))

    def apply_density(self, values):
        """Filter densities using sparse matrix-vector product (exact)."""
        return np.asarray(self._W @ np.asarray(values, dtype=np.float64)).ravel()

    def apply_sensitivity(self, sensitivities, rho=None, rho_min=1e-9):
        """Filter sensitivities using exact adjoint of the density filter.

        The exact adjoint of ρ̂ = W·ρ is simply W^T @ dc (chain rule).
        The previous density-weighted form W^T@(dc·ρ)/ρ was the Sigmund (1997)
        heuristic which introduces bias on non-uniform tetrahedral meshes.
        """
        dc = np.asarray(sensitivities, dtype=np.float64)
        return np.asarray(self._W.T @ dc).ravel()


class MinimumMemberSizeProjection3D:
    """Heaviside-style projection over a larger neighborhood to suppress tiny members."""

    def __init__(self, nodes, elements, base_radius, size_factor=1.5, beta=8.0):
        self.size_factor = float(max(size_factor, 1.0))
        self.beta = float(max(beta, 1.0))
        self.proj_filter = DensityFilter3D(nodes, elements, float(base_radius) * self.size_factor)

    def set_beta(self, beta):
        self.beta = float(max(beta, 1.0))

    def apply(self, rho, target_volume=None, rho_min=1e-6, passive_solid=None):
        r = np.asarray(rho, dtype=np.float64)
        rmin = float(np.clip(rho_min, 0.0, 0.2))

        x = np.clip((r - rmin) / max(1.0 - rmin, 1e-12), 0.0, 1.0)
        x_bar = self.proj_filter.apply_density(x)

        if target_volume is None:
            eta = 0.5
        else:
            tv = float(np.clip(target_volume, 1e-6, 1.0 - 1e-6))
            lo, hi = 0.0, 1.0
            for _ in range(60):
                eta_mid = 0.5 * (lo + hi)
                proj_mid = 1.0 / (1.0 + np.exp(-self.beta * (x_bar - eta_mid)))
                
                if passive_solid is not None:
                    proj_mid[passive_solid] = 1.0
                    
                phys = rmin + (1.0 - rmin) * proj_mid
                if float(np.mean(phys)) > tv:
                    lo = eta_mid
                else:
                    hi = eta_mid
            eta = 0.5 * (lo + hi)

        proj = 1.0 / (1.0 + np.exp(-self.beta * (x_bar - eta)))
        if passive_solid is not None:
            proj[passive_solid] = 1.0
            
        return rmin + (1.0 - rmin) * proj
