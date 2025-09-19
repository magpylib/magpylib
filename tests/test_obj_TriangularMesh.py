import re
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_BH_triangularmesh import (
    BHJM_magnet_trimesh,
    fix_trimesh_orientation,
    lines_end_in_trimesh,
)


def test_TriangularMesh_repr():
    """TriangularMesh repr test"""
    trimesh = magpy.magnet.TriangularMesh.from_pyvista(
        polarization=(0, 0, 1), polydata=pv.Octahedron()
    )
    assert repr(trimesh).startswith("TriangularMesh"), "TriangularMesh repr failed"


def test_TriangularMesh_barycenter():
    """test TriangularMesh barycenter"""
    pol = (0, 0, 333)
    trimesh = magpy.magnet.TriangularMesh.from_pyvista(
        polarization=pol, polydata=pv.Octahedron()
    ).move((1, 2, 3))
    bary = np.array([1, 2, 3])
    np.testing.assert_allclose(trimesh.barycenter, bary)


def test_TriangularMesh_getBH():
    """Compare meshed cube to magpylib cube"""
    dimension = (1, 1, 1)
    polarization = (100, 200, 300)
    mesh3d = magpy.graphics.model3d.make_Cuboid()
    vertices = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    faces = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T

    faces[0] = faces[0][[0, 2, 1]]  # flip one triangle in wrong orientation

    cube = magpy.magnet.Cuboid(polarization=polarization, dimension=dimension)
    cube.rotate_from_angax(19, (1, 2, 3))
    cube.move((1, 2, 3))

    cube_facet_reorient_true = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        polarization=polarization,
        vertices=vertices,
        faces=faces,
        reorient_faces=True,
    )
    cube_misc_faces = cube_facet_reorient_true.to_TriangleCollection()
    cube_facet_reorient_false = magpy.magnet.TriangularMesh(
        position=cube.position,
        orientation=cube.orientation,
        polarization=polarization,
        vertices=vertices,
        faces=faces,
        reorient_faces=False,
    )

    vertices = np.linspace((-2, 0, 0), (2, 0, 0), 50)
    B1 = cube.getB(vertices)
    B2 = cube_facet_reorient_true.getB(vertices)
    B3 = cube_misc_faces.getB(vertices)
    B4 = cube_facet_reorient_false.getB(vertices)

    np.testing.assert_allclose(B1, B2)
    np.testing.assert_allclose(B1, B3)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B1, B4)

    H1 = cube.getH(vertices)
    H2 = cube_facet_reorient_true.getH(vertices)
    H3 = cube_misc_faces.getH(vertices)

    np.testing.assert_allclose(H1, H2)
    np.testing.assert_allclose(H1, H3)


def test_TriangularMesh_getB_different_facet_shapes_mixed():
    """test different facet shapes, facet cube has
    shape (12,3,3) vs (4,3,3) for facet tetrahedron"""
    tetra_pv = pv.Tetrahedron()
    tetra = (
        magpy.magnet.Tetrahedron(
            polarization=(0.444, 0.555, 0.666), vertices=tetra_pv.points
        )
        .move((-1, 1, 1))
        .rotate_from_angax([14, 65, 97], (4, 6, 9), anchor=0)
    )
    tetra_kwargs = {
        "polarization": tetra.polarization,
        "position": tetra.position,
        "orientation": tetra.orientation,
    }
    tmesh_tetra = magpy.magnet.TriangularMesh.from_pyvista(
        polydata=tetra_pv, **tetra_kwargs
    )
    assert tmesh_tetra.status_reoriented is True
    cube = (
        magpy.magnet.Cuboid(polarization=(0.111, 0.222, 0.333), dimension=(1, 1, 1))
        .move((1, 1, 1))
        .rotate_from_angax([14, 65, 97], (4, 6, 9), anchor=0)
    )
    cube_kwargs = {
        "polarization": cube.polarization,
        "position": cube.position,
        "orientation": cube.orientation,
    }
    tmesh_cube = magpy.magnet.TriangularMesh.from_pyvista(
        polydata=pv.Cube(), **cube_kwargs
    )
    # create a sensor of which the pixel line crosses both bodies
    sens = magpy.Sensor(pixel=np.linspace((-2, 1, 1), (2, 1, 1))).rotate_from_angax(
        [14, 65, 97], (4, 6, 9), anchor=0
    )

    np.testing.assert_allclose(magpy.getB(cube, sens), magpy.getB(tmesh_cube, sens))
    np.testing.assert_allclose(magpy.getB(tetra, sens), magpy.getB(tmesh_tetra, sens))
    np.testing.assert_allclose(
        magpy.getB([tetra, cube], sens), magpy.getB([tmesh_tetra, tmesh_cube], sens)
    )


def test_magnet_trimesh_func():
    """test on manual inside"""
    pol = (0.111, 0.222, 0.333)
    dim = (10, 10, 10)
    cube = magpy.magnet.Cuboid(polarization=pol, dimension=dim)
    tmesh_cube = magpy.magnet.TriangularMesh.from_pyvista(
        polarization=pol,
        polydata=pv.Cube(
            center=cube.position,
            x_length=10,
            y_length=10,
            z_length=10,
        ),
    )

    pts_inside = np.array([[0, 0, 1]])
    B0 = cube.getB(pts_inside)
    B1 = tmesh_cube.getB(pts_inside)
    B2 = BHJM_magnet_trimesh(
        field="B",
        observers=pts_inside,
        polarization=np.array([pol]),
        mesh=np.array([tmesh_cube.mesh]),
        in_out="inside",
    )[0]
    np.testing.assert_allclose(B0, B1)
    np.testing.assert_allclose(B0, B2)


def test_bad_triangle_indices():
    "raise ValueError if faces index > len(vertices)"
    vertices = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    faces = [[1, 2, 3]]  # index 3 >= len(vertices)
    with pytest.raises(IndexError):
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
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
    faces = np.array([v for k, v in open_mesh.items() if k in "ijk"]).T
    with pytest.raises(ValueError, match=r"Open mesh detected in .*."):
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_open="raise",
        )
    with pytest.raises(ValueError, match=r"Open mesh in .* detected."):
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_open="ignore",
            reorient_faces="raise",
        )
    with pytest.warns(UserWarning) as record:  # noqa: PT030, PT031
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_open="warn",
        )
        assert len(record) == 2
        assert re.match(r"Open mesh detected in .*.", str(record[0].message))
        assert re.match(r"Open mesh in .* detected.", str(record[1].message))

    with pytest.warns(UserWarning) as record:  # noqa: PT030, PT031
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_open="skip",
            reorient_faces="warn",
        )
        assert len(record) == 3
        assert re.match(
            r"Unchecked mesh status in .* detected. Now applying check_open()",
            str(record[0].message),
        )
        assert re.match(r"Open mesh detected in .*.", str(record[1].message))
        assert re.match(r"Open mesh in .* detected.", str(record[2].message))

    with warnings.catch_warnings():  # no warning should be issued!
        warnings.simplefilter("error")
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_open="ignore",
            reorient_faces="ignore",
        )

    mesh = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 1),
        vertices=vertices,
        faces=faces,
        check_open="ignore",
        reorient_faces="ignore",
    )
    with pytest.warns(
        UserWarning,
        match=r"Open mesh of .* detected",
    ):
        mesh.getB((0, 0, 0))

    mesh = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 1),
        vertices=vertices,
        faces=faces,
        check_open="skip",
        reorient_faces="skip",
    )
    with pytest.warns(UserWarning, match=r"Unchecked mesh status of .* detected"):
        mesh.getB((0, 0, 0))


def test_disconnected_mesh():
    """raises Error if mesh is not connected"""
    #  Multiple Text3D letters are disconnected
    with pytest.raises(ValueError, match=r"Disconnected mesh detected in .*."):
        magpy.magnet.TriangularMesh.from_pyvista(
            polarization=(0, 0, 1),
            polydata=pv.Text3D("AB"),
            check_disconnected="raise",
        )


def test_selfintersecting_triangular_mesh():
    """raises Error if self intersecting"""
    # cube with closed with an inverted pyramid crossing the opposite face.
    selfintersecting_mesh3d = {
        "x": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0],
        "y": [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0],
        "z": [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -2.0],
        "i": [7, 0, 0, 0, 2, 6, 4, 0, 3, 7, 4, 5, 6, 7],
        "j": [0, 7, 1, 2, 1, 2, 5, 5, 2, 2, 5, 6, 7, 4],
        "k": [3, 4, 2, 3, 5, 5, 0, 1, 7, 6, 8, 8, 8, 8],
    }
    vertices = np.array([v for k, v in selfintersecting_mesh3d.items() if k in "xyz"]).T
    faces = np.array([v for k, v in selfintersecting_mesh3d.items() if k in "ijk"]).T
    with pytest.raises(ValueError, match=r"Self-intersecting mesh detected in .*."):
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_selfintersecting="raise",
        )
    with pytest.warns(UserWarning, match=r"Self-intersecting mesh detected in .*."):
        magpy.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=vertices,
            faces=faces,
            check_selfintersecting="warn",
        )


def test_TriangularMesh_from_pyvista():
    """Test from_pyvista classmethod"""

    def get_tri_from_pv(obj):
        return magpy.magnet.TriangularMesh.from_pyvista(
            polarization=(0, 0, 1), polydata=obj
        )

    # should work
    get_tri_from_pv(pv.Cube())

    # should fail
    with pytest.raises(TypeError):
        get_tri_from_pv("bad_pyvista_obj_input")

    # Should raise if pyvista is not installed
    with patch.dict(sys.modules, {"pyvista": None}), pytest.raises(ModuleNotFoundError):
        get_tri_from_pv(pv.Cube())


def test_TriangularMesh_from_faces_bad_inputs():
    """Test from_faces classmethod bad inputs"""
    pol = (0, 0, 1)
    kw = {
        "polarization": pol,
        "check_open": False,
        "check_disconnected": False,
        "reorient_faces": False,
    }

    def get_tri_from_triangles(trias):
        return magpy.magnet.TriangularMesh.from_triangles(triangles=trias, **kw)

    def get_tri_from_mesh(mesh):
        return magpy.magnet.TriangularMesh.from_mesh(mesh=mesh, **kw)

    triangle = magpy.misc.Triangle(
        polarization=pol, vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    )

    # good element type but not array-like
    with pytest.raises(
        TypeError,
        match=r"The `triangles` parameter must be a list or Collection of `Triangle` objects*.",
    ):
        get_tri_from_triangles(triangle)

    # element in list has wrong type
    with pytest.raises(
        TypeError, match=r"All elements of `triangles` must be `Triangle` objects*."
    ):
        get_tri_from_triangles(["bad_type"])

    # bad type input
    with pytest.raises(MagpylibBadUserInput):
        get_tri_from_mesh(1)

    # bad shape input
    msh = [((0, 0), (1, 0), (0, 1))] * 2
    with pytest.raises(ValueError, match=r"Input parameter `mesh` has bad shape*."):
        get_tri_from_mesh(msh)

    # bad shape input
    msh = [((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))] * 2
    with pytest.raises(ValueError, match=r"Input parameter `mesh` has bad shape*."):
        get_tri_from_mesh(msh)


def test_TriangularMesh_from_faces_good_inputs():
    """Test from_faces classmethod good inputs"""
    pol = (0, 0, 1)

    # create Tetrahedron and move/orient randomly
    tetra = magpy.magnet.Tetrahedron(
        polarization=pol, vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    )
    tetra.move((3, 4, 5)).rotate_from_angax([13, 37], (1, 2, 3), anchor=0)
    pos_ori = {"orientation": tetra.orientation, "position": tetra.position}

    tmesh1 = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=pol, points=tetra.vertices, **pos_ori
    )

    # from triangle list
    trias = [
        magpy.misc.Triangle(polarization=pol, vertices=face) for face in tmesh1.mesh
    ]
    tmesh2 = magpy.magnet.TriangularMesh.from_triangles(
        polarization=pol, triangles=trias, **pos_ori
    )

    # from collection
    coll = magpy.Collection(trias)
    tmesh3 = magpy.magnet.TriangularMesh.from_triangles(
        polarization=pol, triangles=coll, **pos_ori
    )

    # from mesh
    msh = [t.vertices for t in coll]
    tmesh4 = magpy.magnet.TriangularMesh.from_mesh(
        polarization=pol, mesh=msh, **pos_ori
    )

    points = [0, 0, 0]
    B0 = tetra.getB(points)
    B1 = tmesh1.getB(points)
    B2 = tmesh2.getB(points)
    B3 = tmesh3.getB(points)
    B4 = tmesh4.getB(points)

    np.testing.assert_allclose(B0, B1)
    np.testing.assert_allclose(B0, B2)
    np.testing.assert_allclose(B0, B3)
    np.testing.assert_allclose(B0, B4)

    # # 2-> test getB vs ConvexHull Tetrahedron faces as msh
    # msh = tetra_from_ConvexHull.mesh
    # src2 = get_tri_from_faces(msh, **pos_orient)
    # B2 = src2.getB(points)
    # np.testing.assert_allclose(B0, B2)

    # # 3-> test getB vs ConvexHull Tetrahedron faces as magpylib.misc.Triangles
    # msh = [magpy.misc.Triangle(mag, face) for face in tetra_from_ConvexHull.mesh]
    # src3 = get_tri_from_faces(msh, **pos_orient)
    # B3 = src3.getB(points)
    # np.testing.assert_allclose(B0, B3)

    # # 4-> test getB mixed input
    # msh = [magpy.misc.Triangle(mag, face) for face in tetra_from_ConvexHull.mesh]
    # msh[-1] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    # src4 = get_tri_from_faces(msh, **pos_orient)
    # B4 = src4.getB(points)
    # np.testing.assert_allclose(B0, B4)


def test_lines_ends_in_trimesh():
    "test special cases"

    # line point coincides with facet point
    msh = np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]]])
    lines = np.array([[[-1, 1, -1], [1, 0, 0]]])

    assert bool(lines_end_in_trimesh(lines, msh)[0]) is True


def test_reorient_on_closed_but_disconnected_mesh():
    """Reorient edge case"""
    N = 3
    s1 = pv.Sphere(theta_resolution=N, phi_resolution=N)
    s2 = pv.Sphere(theta_resolution=N, phi_resolution=N, center=(2, 0, 0))
    polydata = s1.merge(s2)
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    vertices = polydata.points

    # flip every 2nd normals
    bad_faces = faces.copy()
    bad_faces[::2] = bad_faces[::2, [0, 2, 1]]

    fixed_faces = fix_trimesh_orientation(vertices, bad_faces)
    np.testing.assert_array_equal(faces, fixed_faces)


def test_bad_mode_input():
    """test bad mode input"""
    with pytest.raises(
        ValueError,
        match=r"The `check_open mode` argument .*, instead received 'badinput'.",
    ):
        magpy.magnet.TriangularMesh.from_pyvista(
            polarization=(0, 0, 1), polydata=pv.Octahedron(), check_open="badinput"
        )


def test_orientation_edge_case():
    """test reorientation edge case"""

    # reorientation may fail if the face orientation vector is two small
    # see issue #636

    def points(r0):
        return [(r0 * np.cos(t), r0 * np.sin(t), 10) for t in ts] + [(0, 0, 0)]

    ts = np.linspace(0, 2 * np.pi, 5)
    cone1 = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=(0, 0, 1), points=points(12)
    )
    cone2 = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=(0, 0, 1), points=points(13)
    )

    np.testing.assert_array_equal(cone1.faces, cone2.faces)


def test_TriangularMesh_volume():
    """Test TriangularMesh volume calculation."""
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    mesh = magpy.magnet.TriangularMesh(
        vertices=vertices, faces=faces, polarization=(0, 0, 1)
    )
    calculated = mesh.volume
    expected = 1.0 / 6.0
    assert abs(calculated - expected) < 1e-10


def test_TriangularMesh_volume_complex():
    """Test TriangularMesh volume calculation with complex body."""
    # Create a complex Pyvista PolyData object using a boolean operation
    cyl = pv.Cylinder(radius=0.4, height=10.0, resolution=20).triangulate().subdivide(2)
    cube = pv.Cube().triangulate().subdivide(2)
    obj = cube.boolean_difference(cyl)
    obj = obj.clean()

    # Construct magnet from PolyData object
    magnet = magpy.magnet.TriangularMesh.from_pyvista(
        polarization=(0, 0, 0.1),
        polydata=obj,
        style_label="magnet",
    )
    calculated = magnet.volume
    expected = obj.volume  # Pyvista calculates volume correctly
    assert abs(calculated - expected) < 1e-10


def test_TriangularMesh_centroid():
    """Test TriangularMesh centroid - should return barycenter if available"""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    tri_mesh = magpy.magnet.TriangularMesh(
        vertices=vertices, faces=faces, polarization=(0, 0, 1), position=(6, 7, 8)
    )
    expected = [6.26289171, 7.26289171, 8.26289171]  # barycenter offset from position
    assert np.allclose(tri_mesh.centroid, expected)
