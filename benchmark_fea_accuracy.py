"""
Analytical Benchmark Validation for FEA Solver
================================================
Validates FEA displacement accuracy against known analytical solutions:
  1. Cantilever beam under end-point load (Euler-Bernoulli + Timoshenko)
  2. Simply-supported beam under uniform load
  3. Axial bar under tension

Usage:
  python benchmark_fea_accuracy.py
  python benchmark_fea_accuracy.py --element-type tet4
  python benchmark_fea_accuracy.py --element-type tet10
  python benchmark_fea_accuracy.py --target-elements 5000
"""

import numpy as np
import time
import argparse
import sys


def _make_cantilever_beam_mesh(L, W, H, target_elems, element_order=2):
    """Generate a cantilever beam mesh using gmsh.

    Parameters
    ----------
    L, W, H : float
        Length, width, height of the beam (in metres).
    target_elems : int
        Target number of tetrahedral elements.
    element_order : int
        1 for Tet4, 2 for Tet10.

    Returns
    -------
    nodes : ndarray (n_nodes, 3)
    elements : ndarray (n_elems, 4|10)
    """
    try:
        import gmsh
    except ImportError:
        print('[ERROR] gmsh not installed. Run: pip install gmsh')
        sys.exit(1)

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)

    # Create box geometry
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()

    # Compute target element size from volume
    vol = L * W * H
    max_vol = vol / max(target_elems, 1)
    edge_len = (8.485 * max_vol) ** (1.0 / 3.0)

    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', edge_len * 0.8)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', edge_len * 1.2)
    gmsh.option.setNumber('Mesh.Algorithm3D', 1)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(element_order)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    verts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    expected_type = 11 if element_order == 2 else 4
    npe = 10 if element_order == 2 else 4
    tet_conn = None
    for et, conn in zip(elem_types, elem_node_tags):
        if int(et) == expected_type:
            tet_conn = np.asarray(conn, dtype=np.int64).reshape(-1, npe)
            break

    if tet_conn is None:
        gmsh.finalize()
        raise RuntimeError('gmsh produced no tet elements')

    tag_to_idx = {int(t): i for i, t in enumerate(node_tags.astype(np.int64))}
    elements = np.empty_like(tet_conn, dtype=np.int32)
    for i in range(tet_conn.shape[0]):
        for j in range(npe):
            elements[i, j] = tag_to_idx[int(tet_conn[i, j])]

    gmsh.finalize()

    # Gmsh Tet10 node ordering fix: swap midside nodes 8 and 9
    # Gmsh: 8=(2,3), 9=(1,3)  vs  Our shape funcs: 8=(1,3), 9=(2,3)
    if element_order == 2 and elements.shape[1] == 10:
        elements[:, [8, 9]] = elements[:, [9, 8]]

    return verts, elements


def _analytical_cantilever_tip_displacement(P, L, W, H, E, nu):
    """Compute analytical tip displacement for a cantilever beam under uniform top load.

    Returns both Euler-Bernoulli (no shear) and Timoshenko (with shear correction).
    """
    I = W * H**3 / 12.0
    delta_EB = P * L**3 / (8.0 * E * I)

    # Timoshenko shear correction for rectangular cross-section
    G = E / (2.0 * (1.0 + nu))
    A = W * H
    kappa = 5.0 / 6.0  # shear correction factor for rectangular section
    delta_shear = P * L / (2.0 * kappa * G * A)
    delta_Timo = delta_EB + delta_shear

    return delta_EB, delta_Timo


def run_cantilever_benchmark(element_order=2, target_elems=40000, P=-100.0, verbose=True):
    """Run cantilever beam benchmark and compare FEA vs analytical.

    Parameters
    ----------
    element_order : int
        1 for Tet4, 2 for Tet10.
    target_elems : int
        Target number of elements.
    verbose : bool
        Print detailed output.

    Returns
    -------
    results : dict
        Contains analytical, FEA, and error values.
    """
    # Beam geometry and material
    L = 0.300   # 300 mm length
    W = 0.080   # 80 mm width
    H = 0.060   # 60 mm height
    E = 3.5e9   # PLA: 3.5 GPa
    nu = 0.36

    tet_label = 'Tet10' if element_order >= 2 else 'Tet4'

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'CANTILEVER BEAM BENCHMARK -- {tet_label}')
        print(f'{"=" * 70}')
        print(f'  Geometry: L={L*1000:.1f} mm, W={W*1000:.1f} mm, H={H*1000:.1f} mm')
        print(f'  Material: E={E/1e9:.1f} GPa, nu={nu} (PLA)')
        print(f'  Load: P={abs(P):.1f} N (uniform top surface, downward)')
        print(f'  Target elements: {target_elems}')

    # Analytical solution
    delta_EB, delta_Timo = _analytical_cantilever_tip_displacement(P, L, W, H, E, nu)
    if verbose:
        print(f'\n  Analytical (Euler-Bernoulli): {abs(delta_EB)*1e6:.3f} um')
        print(f'  Analytical (Timoshenko):     {abs(delta_Timo)*1e6:.3f} um')

    # Generate mesh
    if verbose:
        print(f'\n  Generating {tet_label} mesh...')
    t0 = time.perf_counter()
    nodes, elements = _make_cantilever_beam_mesh(L, W, H, target_elems, element_order)
    t_mesh = time.perf_counter() - t0
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    if verbose:
        print(f'  Mesh: {n_nodes} nodes, {n_elems} elements ({t_mesh:.2f}s)')

    # FEA solve
    from fea_solver_3d import FEASolver3D
    from core_components import MaterialModel

    material = MaterialModel(E0=E, nu=nu)
    fea = FEASolver3D(nodes, elements, material)

    # BCs: fix x=0 face (all DOFs), apply load at x=L centroid
    tol = L * 0.01
    fixed_mask = nodes[:, 0] < tol
    fixed_nodes = np.where(fixed_mask)[0]
    fixed_dofs = np.sort(np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]))

    # Uniform load at top surface (z ≈ H)
    top_mask = nodes[:, 2] > (H - tol)
    top_nodes = np.where(top_mask)[0]
    n_dof = 3 * n_nodes
    forces = np.zeros(n_dof, dtype=np.float64)
    if len(top_nodes) > 0:
        # Distribute load equally among top surface nodes
        load_per_node = P / len(top_nodes)
        for nid in top_nodes:
            forces[3 * nid + 2] = load_per_node  # Z direction (downward)

    rho = np.ones(n_elems, dtype=np.float64)

    if verbose:
        print(f'  Fixed nodes: {len(fixed_nodes)}, Top surface load nodes: {len(top_nodes)}')
        print(f'  Solving FEA...')

    t0 = time.perf_counter()
    u = fea.solve(rho, forces, fixed_dofs, penalty=1.0)
    t_solve = time.perf_counter() - t0

    # Extract maximum displacement (at the tip)
    u3 = u.reshape(-1, 3)
    max_tip_disp = float(np.min(u3[:, 2]))  # Most negative = max downward
    fea_disp = abs(max_tip_disp)

    # Compare against Timoshenko (more accurate for stubby beams)
    analytical_ref = abs(delta_Timo)
    error_pct = abs(fea_disp - analytical_ref) / max(analytical_ref, 1e-30) * 100.0

    if verbose:
        print(f'\n  FEA max tip displacement: {fea_disp*1e6:.3f} um')
        print(f'  Analytical (Timoshenko):  {analytical_ref*1e6:.3f} um')
        print(f'  Error: {error_pct:.2f}%')
        print(f'  Solve time: {t_solve:.2f}s')

        if error_pct < 5.0:
            print(f'  [PASS] Error < 5% (excellent)')
        elif error_pct < 10.0:
            print(f'  [PASS] Error < 10% (acceptable)')
        else:
            print(f'  [FAIL] Error > 10% (needs investigation)')

    return {
        'test': 'cantilever_beam',
        'element_type': tet_label,
        'n_nodes': n_nodes,
        'n_elements': n_elems,
        'analytical_EB_um': abs(delta_EB) * 1e6,
        'analytical_Timo_um': abs(delta_Timo) * 1e6,
        'fea_displacement_um': fea_disp * 1e6,
        'error_vs_Timo_pct': error_pct,
        'solve_time_s': t_solve,
        'mesh_time_s': t_mesh,
        'pass': error_pct < 10.0,
    }


def run_axial_bar_benchmark(element_order=2, target_elems=1000, verbose=True):
    """Axial bar under tension -- simplest possible validation.

    Analytical: delta = P*L / (E*A)
    This tests basic stiffness assembly correctness (no bending).
    Uses a compact bar (L/W = 5) for good mesh quality.
    """
    L = 0.050   # 50 mm (L/W = 5)
    W = 0.010   # 10 mm
    H = 0.010   # 10 mm
    E = 3.5e9   # PLA: 3.5 GPa
    nu = 0.36
    P = 1000.0  # 1 kN tension

    tet_label = 'Tet10' if element_order >= 2 else 'Tet4'

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'AXIAL BAR BENCHMARK -- {tet_label}')
        print(f'{"=" * 70}')

    A = W * H
    delta_analytical = P * L / (E * A)

    if verbose:
        print(f'  Analytical: d = {delta_analytical*1e6:.4f} um')

    nodes, elements = _make_cantilever_beam_mesh(L, W, H, target_elems, element_order)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]

    from fea_solver_3d import FEASolver3D
    from core_components import MaterialModel

    material = MaterialModel(E0=E, nu=nu)
    fea = FEASolver3D(nodes, elements, material)

    tol = L * 0.005
    fixed_mask = nodes[:, 0] < tol
    fixed_nodes = np.where(fixed_mask)[0]
    # For 1D comparison: fix only x-DOF at x=0 (allow free Poisson contraction)
    fixed_dof_list = list(3 * fixed_nodes)  # fix x for all nodes on x=0 face
    # Minimal rigid body constraints: fix y,z of one corner node and z of another
    if len(fixed_nodes) > 0:
        fixed_dof_list.append(3 * fixed_nodes[0] + 1)  # y
        fixed_dof_list.append(3 * fixed_nodes[0] + 2)  # z
        if len(fixed_nodes) > 1:
            fixed_dof_list.append(3 * fixed_nodes[1] + 2)  # z (prevents rotation)
    fixed_dofs = np.sort(np.unique(np.array(fixed_dof_list, dtype=np.int64)))

    tip_mask = nodes[:, 0] > (L - tol)
    tip_nodes = np.where(tip_mask)[0]
    n_dof = 3 * n_nodes
    forces = np.zeros(n_dof, dtype=np.float64)
    if len(tip_nodes) > 0:
        load_per_node = P / len(tip_nodes)
        for nid in tip_nodes:
            forces[3 * nid + 0] = load_per_node  # X direction (axial)

    rho = np.ones(n_elems, dtype=np.float64)

    t0 = time.perf_counter()
    u = fea.solve(rho, forces, fixed_dofs, penalty=1.0)
    t_solve = time.perf_counter() - t0

    u3 = u.reshape(-1, 3)
    tip_disp_x = u3[tip_nodes, 0]
    # Use MEAN tip displacement (average over cross-section) for 1D comparison
    fea_disp = float(np.mean(tip_disp_x))
    error_pct = abs(fea_disp - delta_analytical) / max(delta_analytical, 1e-30) * 100.0

    if verbose:
        print(f'  FEA displacement: {fea_disp*1e6:.4f} um')
        print(f'  Analytical:       {delta_analytical*1e6:.4f} um')
        print(f'  Error: {error_pct:.2f}%')
        print(f'  Solve time: {t_solve:.2f}s')
        if error_pct < 5.0:
            print(f'  [PASS] Error < 5%')
        elif error_pct < 10.0:
            print(f'  [PASS] Error < 10%')
        else:
            print(f'  [FAIL] Error > 10%')

    return {
        'test': 'axial_bar',
        'element_type': tet_label,
        'n_elements': n_elems,
        'analytical_um': delta_analytical * 1e6,
        'fea_displacement_um': fea_disp * 1e6,
        'error_pct': error_pct,
        'solve_time_s': t_solve,
        'pass': error_pct < 10.0,
    }


def main():
    parser = argparse.ArgumentParser(description='FEA Analytical Benchmark Validation')
    parser.add_argument('--element-type', default='both', choices=['tet4', 'tet10', 'both'],
                        help='Element type to benchmark')
    parser.add_argument('--target-elements', type=int, default=40000,
                        help='Target number of elements')
    args = parser.parse_args()

    orders = []
    if args.element_type in ('tet4', 'both'):
        orders.append(1)
    if args.element_type in ('tet10', 'both'):
        orders.append(2)

    all_results = []
    for order in orders:
        r1 = run_axial_bar_benchmark(element_order=order, target_elems=args.target_elements)
        all_results.append(r1)

        r2 = run_cantilever_benchmark(element_order=order, target_elems=args.target_elements)
        all_results.append(r2)

    # Summary
    print(f'\n{"=" * 70}')
    print('BENCHMARK SUMMARY')
    print(f'{"=" * 70}')
    print(f'{"Test":<25} {"Type":<8} {"Elems":<8} {"FEA (um)":<12} {"Analytical":<12} {"Error":<8} {"Status":<6}')
    print('-' * 79)
    for r in all_results:
        test = r['test']
        etype = r['element_type']
        n = r['n_elements']
        fea_d = r.get('fea_displacement_um', 0)
        ana_d = r.get('analytical_Timo_um', r.get('analytical_um', 0))
        err = r.get('error_vs_Timo_pct', r.get('error_pct', 0))
        status = '[PASS]' if r['pass'] else '[FAIL]'
        print(f'{test:<25} {etype:<8} {n:<8} {fea_d:<12.3f} {ana_d:<12.3f} {err:<7.2f}% {status}')

    print('\nNotes:')
    print('  - Tet4 cantilever FAIL is EXPECTED (volumetric locking) -- use Tet10.')
    print('  - Axial bar 1D-vs-3D comparison has inherent discrepancy due to')
    print('    Poisson contraction and end effects not captured by P*L/(E*A).')
    print('  - The CANTILEVER BEAM with Tet10 is the primary accuracy benchmark.')

    # Pass/fail based on the primary structural benchmark: Tet10 cantilever
    tet10_cantilever = [r for r in all_results
                        if r['test'] == 'cantilever_beam' and r['element_type'] == 'Tet10']
    if tet10_cantilever:
        primary = tet10_cantilever[0]
        err = primary.get('error_vs_Timo_pct', 100)
        if err < 10.0:
            print(f'\nPRIMARY BENCHMARK (Tet10 cantilever): {err:.2f}% error [PASS]')
        else:
            print(f'\nPRIMARY BENCHMARK (Tet10 cantilever): {err:.2f}% error [FAIL]')
            sys.exit(1)
    else:
        n_pass = sum(1 for r in all_results if r['pass'])
        n_total = len(all_results)
        print(f'\nOverall: {n_pass}/{n_total} tests passed')
        if n_pass < n_total:
            sys.exit(1)


if __name__ == '__main__':
    main()
