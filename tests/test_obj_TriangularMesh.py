import sys
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv

import magpylib as magpy
from magpylib._src.fields.field_BH_triangularmesh import fix_trimesh_orientation
from magpylib._src.fields.field_BH_triangularmesh import lines_end_in_trimesh
from magpylib._src.fields.field_BH_triangularmesh import magnet_trimesh_field


def test_TriangularMesh_repr():
    """TriangularMesh repr test"""
    trimesh = magpy.magnet.TriangularMesh.from_pyvista((0, 0, 1000), pv.Octahedron())
    assert repr(trimesh).startswith("TriangularMesh"), "TriangularMesh repr failed"


def test_TriangularMesh_barycenter():
    """test TriangluarMesh barycenter"""
    mag = (0, 0, 333)
    trimesh = magpy.magnet.TriangularMesh.from_pyvista(mag, pv.Octahedron()).move(
        (1, 2, 3)
    )
    bary = np.array([1, 2, 3])
    np.testing.assert_allclose(trimesh.barycenter, bary)


def test_TriangularMesh_getBH():
    """Compare meshed cube to magpylib cube"""
    dimension = (1, 1, 1)
    magnetization = (100, 200, 300)
    mesh3d = magpy.graphics.model3d.make_Cuboid()
    vertices = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    triangles = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T

    triangles[0] = triangles[0][[0, 2, 1]]  # flip one triangle in wrong orientation

    cube = magpy.magnet.Cuboid(magnetization=magnetization, dimension=dimension)
    cube.rotate_from_angax(19, (1, 2, 3))
    cube.move((1, 2, 3))

    cube_facet_reorient_true = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        magnetization=magnetization,
        vertices=vertices,
        triangles=triangles,
        reorient_triangles=True,
    )
    cube_misc_triangles = cube_facet_reorient_true.to_TrianglesCollection()
    cube_facet_reorient_false = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        magnetization=magnetization,
        vertices=vertices,
        triangles=triangles,
        reorient_triangles=False,
    )

    vertices = np.linspace((-2, 0, 0), (2, 0, 0), 50)
    B1 = cube.getB(vertices)
    B2 = cube_facet_reorient_true.getB(vertices)
    B3 = cube_misc_triangles.getB(vertices)
    B4 = cube_facet_reorient_false.getB(vertices)

    np.testing.assert_allclose(B1, B2)
    np.testing.assert_allclose(B1, B3)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B1, B4)

    H1 = cube.getH(vertices)
    H2 = cube_facet_reorient_true.getH(vertices)
    H3 = cube_misc_triangles.getH(vertices)

    np.testing.assert_allclose(H1, H2)
    np.testing.assert_allclose(H1, H3)


def test_TriangularMesh_getB_different_facet_shapes_mixed():
    """test different facet shapes, facet cube has
    shape (12,3,3) vs (4,3,3) for facet tetrahedron"""
    tetra_pv = pv.Tetrahedron()
    tetra = (
        magpy.magnet.Tetrahedron((444, 555, 666), vertices=tetra_pv.points)
        .move((-1, 1, 1))
        .rotate_from_angax([14, 65, 97], (4, 6, 9), anchor=0)
    )
    tetra_kwargs = {
        "magnetization": tetra.magnetization,
        "position": tetra.position,
        "orientation": tetra.orientation,
    }
    tetra_facets = magpy.magnet.TriangularMesh.from_pyvista(
        polydata=tetra_pv, **tetra_kwargs
    )
    assert tetra_facets.is_reoriented
    cube = (
        magpy.magnet.Cuboid((111, 222, 333), (1, 1, 1))
        .move((1, 1, 1))
        .rotate_from_angax([14, 65, 97], (4, 6, 9), anchor=0)
    )
    cube_kwargs = {
        "magnetization": cube.magnetization,
        "position": cube.position,
        "orientation": cube.orientation,
    }
    cube_facets = magpy.magnet.TriangularMesh.from_pyvista(
        polydata=pv.Cube(), **cube_kwargs
    )
    # create a sensor of which the pixel line corsses both bodies
    sens = magpy.Sensor(pixel=np.linspace((-2, 1, 1), (2, 1, 1))).rotate_from_angax(
        [14, 65, 97], (4, 6, 9), anchor=0
    )

    np.testing.assert_allclose(magpy.getB(cube, sens), magpy.getB(cube_facets, sens))
    np.testing.assert_allclose(magpy.getB(tetra, sens), magpy.getB(tetra_facets, sens))
    np.testing.assert_allclose(
        magpy.getB([tetra, cube], sens), magpy.getB([tetra_facets, cube_facets], sens)
    )


def test_magnet_trimesh_func():
    """test on manual inside"""
    mag = (111, 222, 333)
    dim = (10, 10, 10)
    cube = magpy.magnet.Cuboid(mag, dim)
    cube_facets = magpy.magnet.TriangularMesh.from_pyvista(
        mag, pv.Cube(cube.position, *dim)
    )

    pts_inside = np.array([[0, 0, 1]])
    B0 = cube.getB(pts_inside)
    B1 = cube_facets.getB(pts_inside)
    B2 = magnet_trimesh_field(
        "B",
        pts_inside,
        np.array([mag]),
        np.array([cube_facets.facets]),
        in_out="inside",
    )[0]
    np.testing.assert_allclose(B0, B1)
    np.testing.assert_allclose(B0, B2)


def test_bad_triangle_indices():
    "raise ValueError if triangles index > len(vertices)"
    vertices = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    triangles = [[1, 2, 3]]  # index 3 >= len(vertices)
    with pytest.raises(ValueError):
        magpy.magnet.TriangularMesh(
            magnetization=(0, 0, 1000),
            vertices=vertices,
            triangles=triangles,
        )


def test_minimum_vertices():
    "raise ValueError if triangles index > len(vertices)"
    vertices = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    triangles = [[1, 2, 3]]
    with pytest.raises(ValueError):
        magpy.magnet.TriangularMesh(
            magnetization=(0, 0, 1000),
            vertices=vertices,
            triangles=triangles,
        )


def test_open_mesh():
    """raises Error if mesh is open"""
    open_mesh = {
        "i": [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3],
        "j": [0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2],
        "k": [3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7],
        "x": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        "y": [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
        "z": [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    }
    vertices = np.array([v for k, v in open_mesh.items() if k in "xyz"]).T
    triangles = np.array([v for k, v in open_mesh.items() if k in "ijk"]).T
    with pytest.raises(ValueError):
        magpy.magnet.TriangularMesh(
            magnetization=(0, 0, 1000),
            vertices=vertices,
            triangles=triangles,
            validate_closed=True,
        )


def test_disjoint_mesh():
    """raises Error if mesh is not connected"""
    #  Multiple Text3D letters are disjoint
    with pytest.raises(ValueError):
        magpy.magnet.TriangularMesh.from_pyvista(
            magnetization=(0, 0, 1000),
            polydata=pv.Text3D("AB"),
        )


def test_TriangularMesh_from_pyvista():
    """Test from_pyvista classmethod"""

    def get_tri_from_pv(obj):
        return magpy.magnet.TriangularMesh.from_pyvista((0, 0, 1000), obj)

    # shoud work
    get_tri_from_pv(pv.Cube())

    # should fail
    with pytest.raises(TypeError):
        get_tri_from_pv("bad_pyvista_obj_input")

    # Should raise if pyvista is not installed
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ModuleNotFoundError):
            get_tri_from_pv(pv.Cube())


def test_TriangularMesh_from_facets_bad_inputs():
    """Test from_facets classmethod bad inputs"""
    mag = (0, 0, 1000)

    def get_tri_from_facets(facets):
        return magpy.magnet.TriangularMesh.from_triangular_facets(
            mag, facets, validate_open_mesh=False, reorient_triangles=False
        )

    triangle = magpy.misc.Triangle(mag, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    # good element type but not array-like
    with pytest.raises(TypeError):
        get_tri_from_facets(triangle)

    # array-like but bad shape
    with pytest.raises(ValueError):
        get_tri_from_facets(np.array([(0, 0, 0), (1, 0, 0)]))

    # element in list has wrong type
    with pytest.raises(TypeError):
        get_tri_from_facets(["bad_type"])

    # element in list is array like but has bad shape
    with pytest.raises(ValueError):
        get_tri_from_facets([triangle, [(0, 0, 0), (1, 0, 0)]])


def test_TriangularMesh_from_facets_good_inputs():
    """Test from_facets classmethod good inputs"""
    mag = (0, 0, 1000)

    def get_tri_from_facets(facets, **kwargs):
        return magpy.magnet.TriangularMesh.from_triangular_facets(mag, facets, **kwargs)

    # create Tetrahedron and move/orient randomly
    tetra = magpy.magnet.Tetrahedron(mag, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tetra.move((3, 4, 5)).rotate_from_angax([13, 37], (1, 2, 3), anchor=0)
    pos_orient = dict(orientation=tetra.orientation, position=tetra.position)

    # copy Tetrahedron from vertices and convexhull into a TriangularMesh object
    tetra_from_ConvexHull = magpy.magnet.TriangularMesh.from_ConvexHull(
        mag, tetra.vertices, **pos_orient
    )

    # 1-> test getB vs ConvexHull Tetrahedron
    points = [0, 0, 0]
    B0 = tetra.getB(points)
    B1 = tetra_from_ConvexHull.getB(points)
    np.testing.assert_allclose(B0, B1)

    # 2-> test getB vs ConvexHull Tetrahedron faces as facets
    facets = tetra_from_ConvexHull.facets
    src2 = get_tri_from_facets(facets, **pos_orient)
    B2 = src2.getB(points)
    np.testing.assert_allclose(B0, B2)

    # 3-> test getB vs ConvexHull Tetrahedron faces as magpylib.misc.Triangles
    facets = [magpy.misc.Triangle(mag, face) for face in tetra_from_ConvexHull.facets]
    src3 = get_tri_from_facets(facets, **pos_orient)
    B3 = src3.getB(points)
    np.testing.assert_allclose(B0, B3)

    # 4-> test getB mixed input
    facets = [magpy.misc.Triangle(mag, face) for face in tetra_from_ConvexHull.facets]
    facets[-1] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    src4 = get_tri_from_facets(facets, **pos_orient)
    B4 = src4.getB(points)
    np.testing.assert_allclose(B0, B4)


def test_lines_ends_in_trimesh():
    "test special cases"

    # line point coincides with facet point
    facets = np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]]])
    lines = np.array([[[-1, 1, -1], [1, 0, 0]]])

    assert bool(lines_end_in_trimesh(lines, facets)[0]) is True


def test_reorient_on_closed_but_disconnected_mesh():
    """Reorient edge case"""
    N = 3
    s1 = pv.Sphere(theta_resolution=N, phi_resolution=N)
    s2 = pv.Sphere(theta_resolution=N, phi_resolution=N, center=(2, 0, 0))
    polydata = s1.merge(s2)
    triangles = polydata.faces.reshape(-1, 4)[:, 1:]
    vertices = polydata.points

    # flip every 2nd normals
    bad_triangles = triangles.copy()
    bad_triangles[::2] = bad_triangles[::2, [0, 2, 1]]

    fixed_triangles = fix_trimesh_orientation(vertices, bad_triangles)
    np.testing.assert_array_equal(triangles, fixed_triangles)
