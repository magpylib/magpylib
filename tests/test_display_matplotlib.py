import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest
import pyvista as pv

import magpylib as magpy
from magpylib.graphics.model3d import make_Cuboid
from magpylib.magnet import Cuboid
from magpylib.magnet import Cylinder
from magpylib.magnet import CylinderSegment
from magpylib.magnet import Sphere


# pylint: disable=assignment-from-no-return

magpy.defaults.reset()


def test_Cuboid_display():
    """testing display"""
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.1, 0.1), (2, 2, 2), 20), start=-1)
    src.show(style_path_frames=5, return_fig=True)

    with plt.ion():
        x = src.show(style_path_show=False)
    assert x is None  # only place where return_fig=False, for testcov


def test_Cylinder_display():
    """testing display"""
    # path should revert to True
    ax = plt.subplot(projection="3d")
    src = Cylinder((1, 2, 3), (1, 2))
    src.show(canvas=ax, style_path_frames=15, backend="matplotlib")

    # hide path
    src.move(np.linspace((0.4, 0.4, 0.4), (12, 12, 12), 30), start=-1)
    src.show(canvas=ax, style_path_show=False, backend="matplotlib")

    # empty frames, ind>path_len, should display last position
    src.show(canvas=ax, style_path_frames=[], backend="matplotlib")

    src.show(
        canvas=ax, style_path_frames=[1, 5, 6], backend="matplotlib", return_fig=True
    )


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    src.show(canvas=ax, style_path_frames=15, return_fig=True)

    src.move(np.linspace((0.4, 0.4, 0.4), (13.2, 13.2, 13.2), 33), start=-1)
    src.show(canvas=ax, style_path_show=False, return_fig=True)


def test_Sphere_display():
    """testing display"""
    # path should revert to True
    ax = plt.subplot(projection="3d")
    src = Sphere((1, 2, 3), 2)
    src.show(canvas=ax, style_path_frames=15, return_fig=True)

    src.move(np.linspace((0.4, 0.4, 0.4), (8, 8, 8), 20), start=-1)
    src.show(canvas=ax, style_path_show=False, return_fig=True)


def test_Tetrahedron_display():
    """testing Tetrahedron display"""
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    src = magpy.magnet.Tetrahedron(magnetization=(100, 200, 300), vertices=verts)
    src.show(return_fig=True)


def test_Sensor_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.style.arrows.z.color = "magenta"
    sens.style.arrows.z.show = False
    poz = np.linspace((0.4, 0.4, 0.4), (13.2, 13.2, 13.2), 33)
    sens.move(poz, start=-1)
    sens.show(
        canvas=ax, markers=[(100, 100, 100)], style_path_frames=15, return_fig=True
    )

    sens.pixel = [(2, 3, 4)]  # one non-zero pixel
    sens.show(
        canvas=ax, markers=[(100, 100, 100)], style_path_show=False, return_fig=True
    )


def test_CustomSource_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    cs = magpy.misc.CustomSource()
    cs.show(canvas=ax, return_fig=True)


def test_Loop_display():
    """testing display for Loop source"""
    ax = plt.subplot(projection="3d")
    src = magpy.current.Loop(current=1, diameter=1)
    src.show(canvas=ax, return_fig=True)

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    src.show(canvas=ax, style_path_frames=3, return_fig=True)


def test_Triangle_display():
    """testing display for Triangle source built from vertices"""
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    mesh3d = magpy.graphics.model3d.make_TriangularMesh(vertices=verts)
    # note: triangles are built by scipy.Convexhull since triangles=None
    # ConvexHull DOES NOT GUARRANTY proper orientation of triangles when building a body
    points = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    triangles = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T
    src = magpy.Collection(
        [
            magpy.misc.Triangle(magnetization=(1000, 0, 0), vertices=v)
            for v in points[triangles]
        ]
    )
    magpy.show(
        *src,
        backend="matplotlib",
        style_orientation_offset=0.5,
        style_orientation_size=2,
        style_orientation_color="yellow",
        style_orientation_symbol="cone",
        style_magnetization_mode="color+arrow",
        return_fig=True,
    )


def test_TringularMesh_display():
    """testing display for TriangleMesh source built from vertices"""
    # test  classic trimesh display
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    src = magpy.magnet.TriangularMesh.from_ConvexHull(
        magnetization=(1000, 0, 0), points=points
    )
    src.show(backend="matplotlib", style_description_show=False, return_fig=True)

    # test display of disjoint and open mesh elements
    polydata = pv.Text3D("AB")  # create disjoint mesh
    polydata = polydata.triangulate()
    vertices = polydata.points
    triangles = polydata.faces.reshape(-1, 4)[:, 1:]
    triangles = triangles[1:]  # open the mesh
    src = magpy.magnet.TriangularMesh(
        (0, 0, 1000),
        vertices,
        triangles,
        validate_closed=False,
        validate_connected=False,
        reorient_triangles=False,
        style_mesh_grid_show=True,
    )

    src.show(backend="matplotlib", return_fig=True)


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax = plt.subplot(projection="3d")
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = pm1.copy(position=(2, 0, 0))
    pm3 = pm1.copy(position=(4, 0, 0))
    nested_col = (pm1 + pm2 + pm3).set_children_styles(color="magenta")
    nested_col.show(canvas=ax, return_fig=True)


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    dip = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    dip.show(canvas=ax2, return_fig=True)
    dip.show(canvas=ax2, style_path_frames=2, return_fig=True)


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    src1 = magpy.current.Loop(1, 2)
    src2 = magpy.current.Loop(1, 2)
    src1.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, start=-1)
    src1.show(canvas=ax2, style_path_frames=2, style_arrow_size=0, return_fig=True)
    src2.show(canvas=ax2, return_fig=True)
    src3.show(canvas=ax2, style_arrow_size=0, return_fig=True)
    src4.show(canvas=ax2, style_path_frames=2, return_fig=True)


def test_matplotlib_model3d_extra():
    """test display extra model3d"""

    # using "plot"
    xs, ys, zs = [(1, 2)] * 3
    trace1 = dict(
        backend="matplotlib",
        constructor="plot",
        args=(xs, ys, zs),
        kwargs=dict(ls="-"),
    )
    obj1 = magpy.misc.Dipole(moment=(0, 0, 1))
    obj1.style.model3d.add_trace(**trace1)

    # using "plot_surface"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, 0 : np.pi : 6j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    trace2 = dict(
        backend="matplotlib",
        constructor="plot_surface",
        args=(xs, ys, zs),
        kwargs=dict(
            cmap=plt.cm.YlGnBu_r,  # pylint: disable=no-member
        ),
    )
    obj2 = magpy.Collection()
    obj2.style.model3d.add_trace(**trace2)

    # using "plot_trisurf"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, -0.5:0.5:6j]
    u, v = u.flatten(), v.flatten()
    xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    zs = 0.5 * v * np.sin(u / 2.0)
    tri = mtri.Triangulation(u, v)
    trace3 = dict(
        backend="matplotlib",
        constructor="plot_trisurf",
        args=lambda: (xs, ys, zs),  # test callable args
        kwargs=dict(
            triangles=tri.triangles,
            cmap=plt.cm.Spectral,  # pylint: disable=no-member
        ),
    )
    obj3 = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3, 0, 0))
    obj3.style.model3d.add_trace(**trace3)

    ax = plt.subplot(projection="3d")
    magpy.show(obj1, obj2, obj3, canvas=ax, return_fig=True)


def test_matplotlib_model3d_extra_bad_input():
    """test display extra model3d"""

    xs, ys, zs = [(1, 2)] * 3
    trace = dict(
        backend="matplotlib",
        constructor="plot",
        kwargs={"xs": xs, "ys": ys, "zs": zs},
        coordsargs={"x": "xs", "y": "ys", "z": "Z"},  # bad Z input
    )
    obj = magpy.misc.Dipole(moment=(0, 0, 1))
    with pytest.raises(ValueError):
        obj.style.model3d.add_trace(**trace)
        ax = plt.subplot(projection="3d")
        obj.show(canvas=ax, return_fig=True)


def test_matplotlib_model3d_extra_updatefunc():
    """test display extra model3d"""
    ax = plt.subplot(projection="3d")
    obj = magpy.misc.Dipole(moment=(0, 0, 1))
    updatefunc = lambda: make_Cuboid("matplotlib", position=(2, 0, 0))
    obj.style.model3d.data = updatefunc
    ax = plt.subplot(projection="3d")
    obj.show(canvas=ax, return_fig=True)

    with pytest.raises(ValueError):
        updatefunc = "not callable"
        obj.style.model3d.add_trace(updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = "not callable"
        obj.style.model3d.add_trace(updatefunc=updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = lambda: "bad output type"
        obj.style.model3d.add_trace(updatefunc=updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = lambda: {"bad_key": "some_value"}
        obj.style.model3d.add_trace(updatefunc=updatefunc)


def test_empty_display():
    """should not fail if nothing to display"""
    ax = plt.subplot(projection="3d")
    magpy.show(canvas=ax, backend="matplotlib", return_fig=True)


def test_graphics_model_mpl():
    """test base extra graphics with mpl"""
    ax = plt.subplot(projection="3d")
    c = magpy.magnet.Cuboid((0, 1, 0), (1, 1, 1))
    c.rotate_from_angax(33, "x", anchor=0)
    c.style.model3d.add_trace(**make_Cuboid("matplotlib", position=(2, 0, 0)))
    c.show(canvas=ax, style_path_frames=1, backend="matplotlib", return_fig=True)


def test_graphics_model_generic_to_mpl():
    """test generic base extra graphics with mpl"""
    c = magpy.magnet.Cuboid((0, 1, 0), (1, 1, 1))
    c.move([[i, 0, 0] for i in range(2)])
    model3d = make_Cuboid(position=(2, 0, 0))
    model3d["kwargs"]["facecolor"] = np.array(["blue"] * 12)
    c.style.model3d.add_trace(**model3d)
    fig = c.show(style_path_frames=1, backend="matplotlib", return_fig=True)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_mpl_animation():
    """test animation with matplotib"""
    c = magpy.magnet.Cuboid((0, 1, 0), (1, 1, 1))
    c.move([[i, 0, 0] for i in range(2)])
    fig, anim = c.show(
        backend="matplotlib", animation=True, return_animation=True, return_fig=True
    )
    anim._draw_was_started = True  # avoid mpl test warning
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
