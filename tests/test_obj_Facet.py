import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibMissingInput


def test_Triangle_repr():
    """Triangle repr test"""
    line = magpy.misc.Triangle()
    assert line.__repr__()[:8] == "Triangle", "Triangle repr failed"


def test_triangle_input1():
    """test obj-oriented triangle vs cube"""
    obs = (1, 2, 3)
    mag = (0, 0, 333)
    vert = np.array(
        [
            [(-1, -1, 1), (1, -1, 1), (-1, 1, 1)],  # top1
            [(1, -1, -1), (-1, -1, -1), (-1, 1, -1)],  # bott1
            [(1, -1, 1), (1, 1, 1), (-1, 1, 1)],  # top2
            [(1, 1, -1), (1, -1, -1), (-1, 1, -1)],  # bott2
        ]
    )
    coll = magpy.Collection()
    for v in vert:
        coll.add(magpy.misc.Triangle(mag, v))
    cube = magpy.magnet.Cuboid(mag, (2, 2, 2))

    b = coll.getB(obs)
    bb = cube.getB(obs)

    np.testing.assert_allclose(b, bb)


def test_triangle_input3():
    """test core triangle vs objOriented triangle"""

    obs = np.array([(3, 4, 5)] * 4)
    mag = np.array([(111, 222, 333)] * 4)
    vert = np.array(
        [
            [(0, 0, 0), (3, 0, 0), (0, 10, 0)],
            [(3, 0, 0), (5, 0, 0), (0, 10, 0)],
            [(5, 0, 0), (6, 0, 0), (0, 10, 0)],
            [(6, 0, 0), (10, 0, 0), (0, 10, 0)],
        ]
    )
    b = magpy.core.triangle_field("B", obs, mag, vert)
    b = np.sum(b, axis=0)

    tri1 = magpy.misc.Triangle(mag[0], vertices=vert[0])
    tri2 = magpy.misc.Triangle(mag[0], vertices=vert[1])
    tri3 = magpy.misc.Triangle(mag[0], vertices=vert[2])
    tri4 = magpy.misc.Triangle(mag[0], vertices=vert[3])

    bb = magpy.getB([tri1, tri2, tri3, tri4], obs[0], sumup=True)

    np.testing.assert_allclose(b, bb)


def test_empty_object_initialization():
    """empty object init and error msg"""

    fac = magpy.misc.Triangle()

    def call_show():
        """dummy function call show"""
        fac.show()

    np.testing.assert_raises(MagpylibMissingInput, call_show)

    def call_getB():
        """dummy function call getB"""
        fac.getB()

    np.testing.assert_raises(MagpylibMissingInput, call_getB)


def test_barycenter():
    """call barycenter"""
    mag = (0, 0, 333)
    vert = ((-1, -1, 0), (1, -1, 0), (0, 2, 0))
    face = magpy.misc.Triangle(mag, vert)
    bary = np.array([0, 0, 0])
    np.testing.assert_allclose(face.barycenter, bary)


def test_triangular_mesh_body_getB():
    """Compare meshed cube to magpylib cube"""
    dimension = (1, 1, 1)
    magnetization = (100, 200, 300)
    mesh3d = magpy.graphics.model3d.make_Cuboid()
    points = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    triangles = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T

    triangles[0] = triangles[0][[0, 2, 1]]  # flip one triangle in wrong orientation

    cube = magpy.magnet.Cuboid(magnetization=magnetization, dimension=dimension)
    cube.rotate_from_angax(19, (1, 2, 3))
    cube.move((1, 2, 3))

    cube_facet_reorient_true = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        magnetization=magnetization,
        facets=points[triangles],
        reorient_facets=True,
    )
    cube_facet_reorient_false = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        magnetization=magnetization,
        facets=points[triangles],
        reorient_facets=False,
    )

    points = np.linspace((-2, 0, 0), (2, 0, 0), 50)
    B1 = cube.getB(points)
    B2 = cube_facet_reorient_true.getB(points)
    B3 = cube_facet_reorient_false.getB(points)

    np.testing.assert_allclose(B1, B2)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B1, B3)


def test_bad_triangle_indices():
    "raise ValueError if triangles index > len(vertices)"
    vertices = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    triangles = [[1, 2, 3]]
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


def test_self_intersecting_triangular_mesh():
    """raises Error if self intersecting"""
    self_intersecting_mesh3d = {
        "x": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0],
        "y": [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0],
        "z": [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -2.0],
        "i": [7, 0, 0, 0, 2, 6, 4, 0, 3, 7, 4, 5, 6, 7],
        "j": [0, 7, 1, 2, 1, 2, 5, 5, 2, 2, 5, 6, 7, 4],
        "k": [3, 4, 2, 3, 5, 5, 0, 1, 7, 6, 8, 8, 8, 8],
    }
    vertices = np.array(
        [v for k, v in self_intersecting_mesh3d.items() if k in "xyz"]
    ).T
    triangles = np.array(
        [v for k, v in self_intersecting_mesh3d.items() if k in "ijk"]
    ).T
    with pytest.raises(ValueError):
        magpy.magnet.TriangularMesh(
            magnetization=(0, 0, 1000),
            vertices=vertices,
            triangles=triangles,
            validate_mesh=True,
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
            validate_mesh=True,
        )
