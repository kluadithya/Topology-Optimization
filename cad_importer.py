"""
CAD Importer - Fixed to preserve actual STEP/STL geometry including hollow spaces.

ROOT CAUSE OF CUBIC SHAPE BUG:
  The old Mesh3DGenerator used scipy.spatial.Delaunay on just the vertex cloud,
  which always produces a CONVEX HULL filled with tetrahedra — a box/blob shape
  that erases all hollow regions, holes, channels, and internal voids.

FIX STRATEGY:
  1. CADImporter now also tries pythonocc-core (OCC) for native STEP reading.
  2. Mesh3DGenerator now uses tetgen (preferred) or gmsh to create a BOUNDARY-
     CONFORMING tetrahedral mesh that respects the actual surface geometry.
  3. A SurfaceMesh fallback is provided so the GUI can display the real shape
     even if tetrahedral meshing is unavailable.
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Optional heavy imports ──────────────────────────────────────────────────
try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    trimesh = None
    _HAS_TRIMESH = False
    logger.warning('trimesh not available - install with: pip install trimesh')

try:
    import tetgen
    _HAS_TETGEN = True
except ImportError:
    tetgen = None
    _HAS_TETGEN = False
    logger.warning('tetgen not available - install with: pip install tetgen')

try:
    import gmsh
    _HAS_GMSH = True
except ImportError:
    gmsh = None
    _HAS_GMSH = False

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    pv = None
    _HAS_PYVISTA = False


# ── CADImporter ─────────────────────────────────────────────────────────────

class CADImporter:
    """
    Import STEP / STP / STL / OBJ / PLY / GLB / GLTF files and extract
    the surface triangulation while preserving the actual geometry.
    """

    def __init__(self):
        self.mesh = None            # trimesh.Trimesh surface mesh
        self.file_path = None
        self.vertices = None        # (N, 3) float64  - surface vertices
        self.elements = None        # (M, 3) int      - surface triangles
        self.face_ids = None        # (M,) int32      - per-triangle face IDs
        self.edge_segments = None   # (K, 2) int32    - boundary edges for edge picks
        self.bounds = None
        self.is_watertight = False
        self.model_type = 'UNKNOWN'

    def get_supported_formats(self):
        return ['STEP', 'STP', 'STL', 'OBJ', 'DAE', 'PLY', 'GLB', 'GLTF']

    # ------------------------------------------------------------------ load
    def import_file(self, file_path, auto_heal=False):
        try:
            self.file_path = Path(file_path)
            if not self.file_path.exists():
                raise FileNotFoundError(f'File not found: {file_path}')

            ext = self.file_path.suffix.upper()
            logger.info(f'Importing: {self.file_path.name}  [{ext}]')

            # ── choose loader ─────────────────────────────────────────────
            if ext in ('.STEP', '.STP'):
                ok = self._load_step(file_path)
                if not ok:
                    ok = self._load_trimesh(file_path)
            else:
                ok = self._load_trimesh(file_path)

            if not ok or self.mesh is None:
                logger.error('Import failed - no mesh produced')
                return False

            # ── post-process ──────────────────────────────────────────────
            self._post_process(auto_heal)
            self._extract_arrays()
            self._log_summary()
            return True

        except Exception as e:
            logger.error(f'Import error: {e}')
            import traceback; traceback.print_exc()
            return False

    # ---------------------------------------------------------------- loaders
    def _load_trimesh(self, file_path):
        """
        Load via trimesh with process=False to keep hollow / multi-body geometry.
        """
        if not _HAS_TRIMESH:
            return False
        try:
            loaded = trimesh.load(str(file_path), process=False)

            if isinstance(loaded, trimesh.Scene):
                # Concatenate all geometry parts (preserves internal surfaces)
                parts = [g for g in loaded.geometry.values()
                         if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
                if not parts:
                    logger.error('Scene contains no valid mesh geometry')
                    return False
                self.mesh = trimesh.util.concatenate(parts)

            elif isinstance(loaded, trimesh.Trimesh):
                self.mesh = loaded

            else:
                logger.error(f'Unsupported trimesh object type: {type(loaded)}')
                return False

            return True
        except Exception as e:
            logger.warning(f'trimesh load failed: {e}')
            return False

    def _load_step(self, file_path):
        """
        Try pythonocc-core for native STEP reading (best quality).
        Falls back to trimesh on failure.
        """
        try:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location

            reader = STEPControl_Reader()
            status = reader.ReadFile(str(file_path))
            if status != IFSelect_RetDone:
                raise RuntimeError('STEP reader failed')

            reader.TransferRoots()
            shape = reader.OneShape()

            # Mesh the BRep shape
            mesh_algo = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
            mesh_algo.Perform()

            vertices_list, faces_list = [], []
            vertex_offset = 0

            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                location = TopLoc_Location()
                try:
                    triangulation = BRep_Tool.Triangulation(face, location)
                except AttributeError:
                    triangulation = BRep_Tool.Triangulation_s(face, location)

                if triangulation is not None:
                    trsf = location.IsIdentity()
                    n_nodes = triangulation.NbNodes()
                    n_tri = triangulation.NbTriangles()

                    verts = np.array([
                        [triangulation.Node(i + 1).X(),
                         triangulation.Node(i + 1).Y(),
                         triangulation.Node(i + 1).Z()]
                        for i in range(n_nodes)
                    ])
                    tris = np.array([
                        [triangulation.Triangle(i + 1).Get()[j] - 1 + vertex_offset
                         for j in range(3)]
                        for i in range(n_tri)
                    ])

                    vertices_list.append(verts)
                    faces_list.append(tris)
                    vertex_offset += n_nodes

                explorer.Next()

            if not vertices_list:
                raise RuntimeError('No triangles extracted from STEP faces')

            all_verts = np.vstack(vertices_list)
            all_faces = np.vstack(faces_list)

            # OCC triangulates each CAD face independently; merge coincident
            # vertices to avoid duplicate-node artifacts in display/remeshing.
            vq = np.round(all_verts, decimals=8)
            uniq_verts, inv = np.unique(vq, axis=0, return_inverse=True)
            all_faces = inv[all_faces]
            all_verts = uniq_verts.astype(np.float64)

            if _HAS_TRIMESH:
                self.mesh = trimesh.Trimesh(vertices=all_verts,
                                            faces=all_faces,
                                            process=False)
                try:
                    self.mesh.remove_duplicate_faces()
                except Exception:
                    pass
            else:
                # Minimal duck-typed container
                class _Mesh:
                    pass
                m = _Mesh()
                m.vertices = all_verts
                m.faces = all_faces
                m.is_watertight = False
                self.mesh = m

            logger.info('  Loaded via pythonocc (native STEP)')
            return True

        except ImportError:
            logger.info('  pythonocc not available - falling back to trimesh')
            return False
        except Exception as e:
            logger.warning(f'  pythonocc STEP load failed: {e} - trying trimesh')
            return False

    # ---------------------------------------------------------- post-process
    def _post_process(self, auto_heal):
        if not _HAS_TRIMESH or not isinstance(self.mesh, trimesh.Trimesh):
            return

        # Remove only degenerate (zero-area) faces - do NOT fill holes or
        # remove unreferenced vertices, as that would alter hollow geometry.
        try:
            self.mesh.remove_degenerate_faces()
        except AttributeError:
            # Newer trimesh versions removed remove_degenerate_faces()
            try:
                mask = self.mesh.nondegenerate_faces()
                if mask is not None and not np.all(mask):
                    self.mesh.update_faces(mask)
            except Exception:
                pass  # skip degenerate face removal if unavailable

        if auto_heal and not self.mesh.is_watertight:
            try:
                trimesh.repair.fix_normals(self.mesh)
                trimesh.repair.fix_winding(self.mesh)
            except Exception:
                pass

    def _extract_arrays(self):
        self.vertices = np.array(self.mesh.vertices, dtype=np.float64)
        self.elements = np.array(self.mesh.faces, dtype=np.int32)

        self.is_watertight = getattr(self.mesh, 'is_watertight', False)
        self.model_type = 'SOLID (watertight)' if self.is_watertight else 'HOLLOW / OPEN SHELL'

        self.bounds = {
            'min': self.vertices.min(axis=0),
            'max': self.vertices.max(axis=0),
            'size': self.vertices.max(axis=0) - self.vertices.min(axis=0)
        }

        self._build_face_edge_metadata()


    def _build_face_edge_metadata(self):
        """Compute face IDs and boundary edges for selection/picking."""
        if self.elements is None:
            return

        faces = np.asarray(self.elements, dtype=np.int32)

        if self.face_ids is None or len(self.face_ids) != len(faces):
            self.face_ids = np.arange(len(faces), dtype=np.int32)

        try:
            edges = np.sort(
                np.vstack([
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ]), axis=1)
            uniq, counts = np.unique(edges, axis=0, return_counts=True)
            self.edge_segments = uniq[counts == 1]
        except Exception:
            self.edge_segments = None

    def _log_summary(self):
        logger.info(f'  Type    : {self.model_type}')
        logger.info(f'  Vertices: {self.vertices.shape[0]:,}')
        logger.info(f'  Triangles:{self.elements.shape[0]:,}')
        size = self.bounds["size"]
        logger.info(f'  Size    : {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}')
        logger.info(f'Successfully loaded: {self.file_path.name}')

    def get_geometry_info(self):
        if self.mesh is None:
            return None
        return {
            'vertices': self.vertices.shape[0],
            'elements': self.elements.shape[0],
            'bounds': self.bounds,
            'is_watertight': self.is_watertight,
            'model_type': self.model_type
        }


# ── Mesh3DGenerator ─────────────────────────────────────────────────────────



class Mesh3DGenerator:
    """
    Generate a boundary-conforming tetrahedral mesh from a surface mesh.

    Priority order:
      1. tetgen  - fast, robust, industry-standard
      2. gmsh    - full-featured mesher
      3. FALLBACK: surface-only mode (no tets) so the GUI still works

    The OLD scipy.Delaunay approach has been REMOVED because it ignores the
    surface boundary and always produces a convex-hull blob.
    """

    def __init__(self, vertices, triangles):
        self.surf_vertices = np.array(vertices, dtype=np.float64)
        self.surf_triangles = np.array(triangles, dtype=np.int32)

        self.tet_vertices = None   # set after successful meshing
        self.tet_elements = None   # (N_tet, 4|10) int

        # Always keep the surface available for GUI display
        self.surface_vertices = self.surf_vertices
        self.surface_faces = self.surf_triangles

    # ---------------------------------------------------------------- public
    def generate_tetrahedrals(self,
                              quality_order=2.0,
                              max_volume=None,
                              element_order=2):
        """
        Try to create a volumetric tet mesh.  Returns True if successful.
        On failure, self.tet_vertices/tet_elements stay None but
        get_mesh(surface_fallback=True) will still return the surface mesh.
        """
        order = 2 if int(element_order) >= 2 else 1
        logger.info(f'Tet meshing mode: order={order} ({"Tet10" if order == 2 else "Tet4"})')

        if _HAS_GMSH:
            ok = self._mesh_with_gmsh(max_volume, element_order=order)
            if ok:
                return True
            logger.warning('gmsh meshing failed - using surface-only fallback')

        logger.warning(
            'No volumetric mesher available.\n'
            '  Install tetgen :  pip install tetgen\n'
            '  Install gmsh   :  pip install gmsh\n'
            'Proceeding with surface-only mesh for BC definition.')
        return False

    def get_mesh(self, surface_fallback=True):
        """
        Return (vertices, elements).
        If tetrahedral meshing succeeded → 3-D tet mesh.
        Else if surface_fallback=True  → surface triangle mesh (still useful).
        Else → None.
        """
        if self.tet_vertices is not None:
            return self.tet_vertices, self.tet_elements
        if surface_fallback:
            logger.info('Using surface mesh (triangles) as fallback')
            return self.surface_vertices, self.surface_faces
        return None

    def get_surface(self):
        """Always returns the original surface triangulation."""
        return self.surface_vertices, self.surface_faces

    # ------------------------------------------------------------- tetgen
    def _mesh_with_tetgen(self, quality_order, max_volume):
        logger.warning('tetgen path is disabled in this build; using gmsh for Tet4/Tet10.')
        return False

    # -------------------------------------------------------------- gmsh
    def _mesh_with_gmsh(self, max_volume, element_order=2):
        gmsh_active = False
        stl_path = None
        try:
            import gmsh
            import tempfile
            import os

            logger.info('Meshing with gmsh ...')

            gmsh.initialize()
            gmsh_active = True
            gmsh.option.setNumber('General.Terminal', 1)
            gmsh.option.setNumber('Mesh.Algorithm3D', 1)  # Delaunay 3D
            gmsh.option.setNumber('Mesh.Optimize', 1)
            gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
            gmsh.option.setNumber('Mesh.HighOrderOptimize', 0)
            gmsh.option.setNumber('Mesh.Smoothing', 10)
            logger.info('  gmsh: initializing and importing STL surface ...')

            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tf:
                stl_path = tf.name

            if _HAS_TRIMESH:
                tmp_mesh = trimesh.Trimesh(
                    vertices=self.surf_vertices,
                    faces=self.surf_triangles,
                    process=False,
                )
                tmp_mesh.export(stl_path)
            else:
                _write_stl(stl_path, self.surf_vertices, self.surf_triangles)

            gmsh.merge(stl_path)

            logger.info('  gmsh: classifying and creating CAD geometry ...')
            # Preserve feature edges better than the previous pi-radian blanket classification.
            feature_angle = float(np.deg2rad(40.0))
            gmsh.model.mesh.classifySurfaces(feature_angle, True, True, feature_angle)
            gmsh.model.mesh.createGeometry()

            surfaces = [entity[1] for entity in gmsh.model.getEntities(2)]
            if not surfaces:
                raise RuntimeError('gmsh found no surfaces after classification')

            surface_loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            gmsh.model.geo.addVolume([surface_loop])
            gmsh.model.geo.synchronize()

            if max_volume is not None and max_volume > 0:
                # Approximate target edge from tet volume; keeps mesh sizing stable.
                target_len = float((8.48528137423857 * max_volume) ** (1.0 / 3.0))
                gmsh.option.setNumber('Mesh.CharacteristicLengthMin', target_len * 0.7)
                gmsh.option.setNumber('Mesh.CharacteristicLengthMax', target_len * 1.3)

            order = 2 if int(element_order) >= 2 else 1
            logger.info(f'  gmsh: generating 3D tetra mesh (order={order}) ...')
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(order)
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            if len(node_tags) == 0:
                raise RuntimeError('gmsh produced no nodes')
            verts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

            elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
            tet_conn_tags = None
            expected_type = 11 if order == 2 else 4
            npe = 10 if order == 2 else 4
            for elem_type, conn in zip(elem_types, elem_node_tags):
                if int(elem_type) == expected_type:
                    tet_conn_tags = np.asarray(conn, dtype=np.int64).reshape(-1, npe)
                    break

            if tet_conn_tags is None or tet_conn_tags.size == 0:
                raise RuntimeError(f'gmsh produced no {"Tet10" if order == 2 else "Tet4"} elements')

            # Remap gmsh node tags to contiguous 0-based indices.
            tag_to_idx = {int(tag): i for i, tag in enumerate(np.asarray(node_tags, dtype=np.int64))}
            tet_elements = np.empty_like(tet_conn_tags, dtype=np.int32)
            for i in range(tet_conn_tags.shape[0]):
                for j in range(npe):
                    tag = int(tet_conn_tags[i, j])
                    if tag not in tag_to_idx:
                        raise RuntimeError(f'gmsh connectivity references unknown node tag {tag}')
                    tet_elements[i, j] = int(tag_to_idx[tag])

            self.tet_vertices = verts
            self.tet_elements = tet_elements

            # §FIX: Gmsh Tet10 node ordering correction.
            # Gmsh assigns midside nodes as:  4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(2,3) 9=(1,3)
            # Our shape functions expect:      4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3)
            # i.e. N8 = 4*L2*L4 (edge 1-3) and N9 = 4*L3*L4 (edge 2-3).
            # Without this swap, every Tet10 element gets corrupted shape derivatives,
            # causing displacement errors that grow with mesh refinement.
            if order == 2 and self.tet_elements.shape[1] == 10:
                self.tet_elements[:, [8, 9]] = self.tet_elements[:, [9, 8]]
                logger.info('  Applied gmsh->FEA Tet10 node reorder (swap nodes 8,9)')

            logger.info(f'  Tetrahedral mesh: {len(self.tet_vertices):,} nodes, '
                        f'{len(self.tet_elements):,} elements')
            return True

        except Exception as e:
            logger.warning(f'gmsh error: {e}')
            return False
        finally:
            try:
                if gmsh_active:
                    gmsh.finalize()
            except Exception:
                pass
            try:
                if stl_path is not None and os.path.exists(stl_path):
                    os.unlink(stl_path)
            except Exception:
                pass

def _write_stl(path, vertices, faces):
    """Minimal ASCII STL writer (fallback when trimesh unavailable)."""
    with open(path, 'w') as f:
        f.write('solid mesh\n')
        for tri in faces:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = v1 - v0; e2 = v2 - v0
            n = np.cross(e1, e2)
            nlen = np.linalg.norm(n)
            n = n / nlen if nlen > 1e-12 else n
            f.write(f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
            f.write('    outer loop\n')
            for v in (v0, v1, v2):
                f.write(f'      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n')
            f.write('    endloop\n')
            f.write('  endfacet\n')
        f.write('endsolid mesh\n')














