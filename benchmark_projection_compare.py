import argparse
import csv
import json
from pathlib import Path
import numpy as np

from cad_importer import CADImporter, Mesh3DGenerator
from core_components import MaterialModel, DEFAULT_CONFIG
from simp_3d import SIMP3DOptimizer
from beso_3d import BESO3DOptimizer
from lsm_3d import LSM3DOptimizer
from homogenization_3d import MoriTanaka3DOptimizer


def _pick_node(nodes, xmode='min', ymode='mid', zmode='mid'):
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]

    x_ref = np.min(x) if xmode == 'min' else np.max(x)
    y_ref = np.min(y) if ymode == 'min' else (np.max(y) if ymode == 'max' else 0.5 * (np.min(y) + np.max(y)))
    z_ref = np.min(z) if zmode == 'min' else (np.max(z) if zmode == 'max' else 0.5 * (np.min(z) + np.max(z)))

    score = (x - x_ref) ** 2 + (y - y_ref) ** 2 + (z - z_ref) ** 2
    return int(np.argmin(score))


def _build_simple_bc(nodes):
    n = nodes.shape[0]
    f = np.zeros(n * 3, dtype=np.float64)

    xmin = np.min(nodes[:, 0])
    fixed_nodes = np.where(np.abs(nodes[:, 0] - xmin) <= max(1e-6, 1e-4 * (np.max(nodes[:, 0]) - xmin + 1e-9)))[0]
    fixed_dofs = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in fixed_nodes]).astype(np.int32)

    load_node = _pick_node(nodes, xmode='max', ymode='mid', zmode='mid')
    f[3 * load_node + 1] = -1000.0

    return fixed_dofs, f


def _build_four_support_bc(nodes):
    """Four-point support benchmark BC for stronger cross-method comparability."""
    n = nodes.shape[0]
    f = np.zeros(n * 3, dtype=np.float64)

    supports = [
        _pick_node(nodes, xmode='min', ymode='min', zmode='min'),
        _pick_node(nodes, xmode='min', ymode='max', zmode='min'),
        _pick_node(nodes, xmode='min', ymode='min', zmode='max'),
        _pick_node(nodes, xmode='min', ymode='max', zmode='max'),
    ]
    supports = np.unique(np.asarray(supports, dtype=np.int32))
    fixed_dofs = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in supports]).astype(np.int32)

    load_node = _pick_node(nodes, xmode='max', ymode='mid', zmode='max')
    f[3 * load_node + 1] = -1000.0

    return fixed_dofs, f


def _build_bc(nodes, mode):
    if str(mode).lower() == 'four_support':
        return _build_four_support_bc(nodes)
    return _build_simple_bc(nodes)


def _make_optimizer(method, nodes, elements, material, cfg):
    m = method.upper()
    if m == 'SIMP':
        return SIMP3DOptimizer(nodes, elements, material, cfg)
    if m == 'BESO':
        return BESO3DOptimizer(nodes, elements, material, cfg)
    if m == 'LSM':
        return LSM3DOptimizer(nodes, elements, material, cfg)
    return MoriTanaka3DOptimizer(nodes, elements, material, cfg)


def _run_case(method, nodes, elements, base_cfg, projection_enabled, bc_mode):
    cfg = dict(base_cfg)
    cfg['use_min_member_projection'] = bool(projection_enabled)

    mat = MaterialModel(
        E0=float(cfg.get('young_modulus', DEFAULT_CONFIG.get('young_modulus', 210e9))),
        nu=float(cfg.get('poisson_ratio', DEFAULT_CONFIG.get('poisson_ratio', 0.3))),
        rho_min=float(cfg.get('rho_min', DEFAULT_CONFIG.get('rho_min', 1e-6))) if 'rho_min' in cfg else 1e-6,
        density=float(cfg.get('material_density', DEFAULT_CONFIG.get('material_density', 7800.0))),
        yield_strength=float(cfg.get('yield_strength', DEFAULT_CONFIG.get('yield_strength', 250e6))),
    )

    fixed_dofs, forces = _build_bc(nodes, bc_mode)
    opt = _make_optimizer(method, nodes, elements, mat, cfg)
    rho, comp = opt.optimize(forces, fixed_dofs, visualizer=None)

    if isinstance(comp, (list, tuple, np.ndarray)):
        comp_hist = [float(x) for x in comp]
    else:
        comp_hist = [float(comp)]

    vol_hist = [float(np.mean(np.asarray(rho, dtype=np.float64))) for _ in range(len(comp_hist))]
    return rho, comp_hist, vol_hist


def _trend_metrics(comp_hist):
    c = np.asarray(comp_hist, dtype=np.float64)
    if c.size < 2:
        return {
            'n_iterations': int(c.size),
            'steady_drop_ratio': 1.0,
            'oscillation_ratio': 0.0,
            'final_change_ratio': 0.0,
            'is_stable': True,
        }
    d = np.diff(c)
    neg = float(np.mean(d <= 0.0))
    osc = float(np.mean(np.sign(d[1:]) != np.sign(d[:-1]))) if d.size > 1 else 0.0
    final_rel = float(abs(c[-1] - c[-2]) / max(abs(c[-1]), 1e-12))
    stable = bool((neg >= 0.60) and (osc <= 0.50) and (final_rel <= 5e-3))
    return {
        'n_iterations': int(c.size),
        'steady_drop_ratio': neg,
        'oscillation_ratio': osc,
        'final_change_ratio': final_rel,
        'is_stable': stable,
    }


def _binarity_metric(rho):
    r = np.asarray(rho, dtype=np.float64)
    # 0 for binary-like field, up to 1 for gray-only field around 0.5.
    return float(np.mean(4.0 * r * (1.0 - r)))


def _write_history_csv(path, comp_hist, vol_hist):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'compliance', 'volume_fraction'])
        for i, (c, v) in enumerate(zip(comp_hist, vol_hist), start=1):
            w.writerow([i, c, v])


def main():
    ap = argparse.ArgumentParser(description='3D topology benchmark: before/after min-member projection')
    ap.add_argument('--cad', required=True, help='Path to CAD file (STEP/STL/OBJ)')
    ap.add_argument('--method', default='SIMP', choices=['SIMP', 'BESO', 'LSM', 'MORI_TANAKA'])
    ap.add_argument('--target-tets', type=int, default=8000)
    ap.add_argument('--bc-mode', default='four_support', choices=['simple', 'four_support'])
    ap.add_argument('--out', default='benchmark_results')
    args = ap.parse_args()

    cad_path = Path(args.cad)
    if not cad_path.exists():
        raise FileNotFoundError(f'CAD file not found: {cad_path}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    importer = CADImporter()
    if not importer.import_file(str(cad_path), auto_heal=True):
        raise RuntimeError('CAD import failed')

    surf_nodes = importer.vertices
    surf_elems = importer.elements

    mesh_gen = Mesh3DGenerator(surf_nodes, surf_elems)
    max_volume = float(np.prod(np.maximum(np.max(surf_nodes, axis=0) - np.min(surf_nodes, axis=0), 1e-9))) / max(int(args.target_tets), 1)
    ok = mesh_gen._mesh_with_gmsh(max_volume)
    if not ok:
        raise RuntimeError('Tet meshing failed for benchmark case')

    nodes, elements = mesh_gen.get_mesh()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        'objective': 'compliance',
        'volume_fraction': 0.3,
        'auto_filter_radius': True,
        'filter_radius_factor': 3.5,
        'penalization': 3.0,
        'max_iterations_auto': 80,
        'min_iterations_auto': 25,
        'use_stress_constraint': True,
        'safety_factor': 1.5,
        'stress_penalty_weight': 2.0,
        'min_member_size_factor': 1.5,
        'min_member_projection_beta_start': 1.0,
        'min_member_projection_beta_end': 32.0,
        'solver_type': 'direct',
    })

    rho_a, comp_a, vol_a = _run_case(args.method, nodes, elements, cfg, projection_enabled=False, bc_mode=args.bc_mode)
    rho_b, comp_b, vol_b = _run_case(args.method, nodes, elements, cfg, projection_enabled=True, bc_mode=args.bc_mode)

    _write_history_csv(out_dir / 'history_before_projection.csv', comp_a, vol_a)
    _write_history_csv(out_dir / 'history_after_projection.csv', comp_b, vol_b)

    trend_a = _trend_metrics(comp_a)
    trend_b = _trend_metrics(comp_b)

    summary = {
        'cad_file': str(cad_path),
        'method': args.method,
        'bc_mode': args.bc_mode,
        'target_tets': int(args.target_tets),
        'n_nodes': int(nodes.shape[0]),
        'n_elements': int(elements.shape[0]),
        'before': {
            'final_compliance': float(comp_a[-1]),
            'final_volume_fraction': float(vol_a[-1]),
            'binarity_index': _binarity_metric(rho_a),
            'trend': trend_a,
        },
        'after': {
            'final_compliance': float(comp_b[-1]),
            'final_volume_fraction': float(vol_b[-1]),
            'binarity_index': _binarity_metric(rho_b),
            'trend': trend_b,
        },
    }

    with open(out_dir / 'benchmark_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / 'BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
        f.write('# Benchmark Report\n\n')
        f.write(f'- CAD: {cad_path}\n')
        f.write(f'- Method: {args.method}\n')
        f.write(f'- BC mode: {args.bc_mode}\n')
        f.write(f'- Nodes: {nodes.shape[0]}\n')
        f.write(f'- Elements: {elements.shape[0]}\n\n')
        f.write('## Final Metrics\n\n')
        f.write(f'- Before projection: compliance={comp_a[-1]:.6e}, volume={vol_a[-1]:.4f}, binarity={summary["before"]["binarity_index"]:.4f}\n')
        f.write(f'- After projection: compliance={comp_b[-1]:.6e}, volume={vol_b[-1]:.4f}, binarity={summary["after"]["binarity_index"]:.4f}\n\n')
        f.write('## Convergence Quality\n\n')
        f.write(f'- Before stable trend: {trend_a["is_stable"]} (steady_drop={trend_a["steady_drop_ratio"]:.2f}, oscillation={trend_a["oscillation_ratio"]:.2f}, final_change={trend_a["final_change_ratio"]:.3e})\n')
        f.write(f'- After stable trend: {trend_b["is_stable"]} (steady_drop={trend_b["steady_drop_ratio"]:.2f}, oscillation={trend_b["oscillation_ratio"]:.2f}, final_change={trend_b["final_change_ratio"]:.3e})\n')
        f.write('\n## Files\n\n')
        f.write('- history_before_projection.csv\n')
        f.write('- history_after_projection.csv\n')
        f.write('- benchmark_summary.json\n')

    print(f'Benchmark outputs written to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

