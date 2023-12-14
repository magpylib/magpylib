# pylint: disable="wrong-import-position"
import re
from unittest.mock import patch

import matplotlib  # noreorder

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest
import pyvista as pv
from matplotlib.figure import Figure as mplFig

import magpylib as magpy
from magpylib._src.display.display import ctx
from magpylib.graphics.model3d import make_Cuboid


# pylint: disable=assignment-from-no-return
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=no-member

magpy.defaults.reset()


def test_Cuboid_display():
    """testing display"""
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.1, 0.1), (2, 2, 2), 20), start=-1)
    src.show(
        style_path_frames=5,
        style_magnetization_arrow_sizemode="absolute",
        style_magnetization_arrow_color="cyan",
        style_magnetization_arrow_style="dashed",
        style_magnetization_arrow_width=3,
        return_fig=True,
    )

    with patch("matplotlib.pyplot.show"):
        x = src.show(style_path_show=False, style_magnetization_mode="color+arrow")
    assert x is None  # only place where return_fig=False, for testcov


def test_Cylinder_display():
    """testing display"""
    # path should revert to True
    ax = plt.subplot(projection="3d")
    src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    src.show(canvas=ax, style_path_frames=15, backend="matplotlib")

    # hide path
    src.move(np.linspace((0.4, 0.4, 0.4), (12, 12, 12), 30), start=-1)
    src.show(canvas=ax, style_path_show=False, backend="matplotlib")

    # empty frames, ind>path_len, should display last position
    src.show(canvas=ax, style_path_frames=[], backend="matplotlib")

    src.show(
        canvas=ax,
        style_path_frames=[1, 5, 6],
        style_path_numbering=True,
        backend="matplotlib",
        return_fig=True,
    )


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = magpy.magnet.CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    src.show(canvas=ax, style_path_frames=15, return_fig=True)

    src.move(np.linspace((0.4, 0.4, 0.4), (13.2, 13.2, 13.2), 33), start=-1)
    src.show(canvas=ax, style_path_show=False, return_fig=True)


def test_Sphere_display():
    """testing display"""
    # path should revert to True
    ax = plt.subplot(projection="3d")
    src = magpy.magnet.Sphere((1, 2, 3), 2)
    src.show(canvas=ax, style_path_frames=15, return_fig=True)

    src.move(np.linspace((0.4, 0.4, 0.4), (8, 8, 8), 20), start=-1)
    src.show(
        canvas=ax,
        style_path_show=False,
        style_magnetization_mode="color+arrow",
        return_fig=True,
    )


def test_Tetrahedron_display():
    """testing Tetrahedron display"""
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    src = magpy.magnet.Tetrahedron(magnetization=(100, 200, 300), vertices=verts)
    src.show(return_fig=True, style_magnetization_mode="color+arrow")


def test_Sensor_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)], handedness="left")
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


def test_Circle_display():
    """testing display for Circle source"""
    ax = plt.subplot(projection="3d")
    src = magpy.current.Circle(current=1, diameter=1)
    src.show(canvas=ax, return_fig=True)

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    src.show(canvas=ax, style_path_frames=3, return_fig=True)


def test_Triangle_display():
    """testing display for Triangle source built from vertices"""
    mesh3d = magpy.graphics.model3d.make_Cuboid()
    # note: triangles are built by scipy.Convexhull since triangles=None
    points = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    triangles = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T
    src = magpy.Collection(
        [
            magpy.misc.Triangle(magnetization=(1000, 1000, 0), vertices=v)
            for v in points[triangles]
        ]
    )
    # make north/south limit pass an ege by bicolor mode and (45° mag)
    magpy.show(
        *src,
        backend="matplotlib",
        style_magnetization_color_mode="bicolor",
        style_orientation_offset=0.5,
        style_orientation_size=2,
        style_orientation_color="yellow",
        style_orientation_symbol="cone",
        style_magnetization_mode="color+arrow",
        return_fig=True,
    )


def test_Triangle_display_from_convexhull():
    """testing display for Triangle source built from vertices"""
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    mesh3d = magpy.graphics.model3d.make_TriangularMesh(vertices=verts)
    # note: faces are built by scipy.Convexhull since faces=None
    # ConvexHull DOES NOT GUARRANTY proper orientation of faces when building a body
    points = np.array([v for k, v in mesh3d["kwargs"].items() if k in "xyz"]).T
    faces = np.array([v for k, v in mesh3d["kwargs"].items() if k in "ijk"]).T
    src = magpy.Collection(
        [
            magpy.misc.Triangle(magnetization=(1000, 0, 0), vertices=v)
            for v in points[faces]
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
    src.show(
        backend="matplotlib",
        style_description_show=False,
        style_mesh_open_show=True,
        style_mesh_disconnected_show=True,
        style_mesh_selfintersecting_show=True,
        style_orientation_show=True,
        return_fig=True,
    )

    # test display of disconnected and open mesh elements
    polydata = pv.Text3D("AB")  # create disconnected mesh
    polydata = polydata.triangulate()
    vertices = polydata.points
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    faces = faces[1:]  # open the mesh
    src = magpy.magnet.TriangularMesh(
        (0, 0, 1000),
        vertices,
        faces,
        check_open="ignore",
        check_disconnected="ignore",
        reorient_faces=False,
        style_mesh_grid_show=True,
    )

    src.show(
        style_mesh_open_show=True,
        style_mesh_disconnected_show=True,
        style_orientation_show=True,
        style_magnetization_mode="color+arrow",
        backend="matplotlib",
        return_fig=True,
    )

    with pytest.warns(UserWarning) as record:
        magpy.magnet.TriangularMesh(
            (0, 0, 1000),
            vertices,
            faces,
            check_open="skip",
            check_disconnected="skip",
            reorient_faces=False,
            style_mesh_grid_show=True,
        ).show(
            style_mesh_open_show=True,
            style_mesh_disconnected_show=True,
            backend="matplotlib",
            return_fig=True,
        )
        assert len(record) == 4
        assert re.match(
            r"Unchecked open mesh status in .* detected", str(record[0].message)
        )
        assert re.match(r"Open mesh detected in .*.", str(record[1].message))
        assert re.match(
            r"Unchecked disconnected mesh status in .* detected", str(record[2].message)
        )
        assert re.match(r"Disconnected mesh detected in .*.", str(record[3].message))

    # test self-intersecting display
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
    with pytest.warns(UserWarning) as record:
        magpy.magnet.TriangularMesh(
            (0, 0, 1000),
            vertices=vertices,
            faces=faces,
            check_open="warn",
            check_disconnected="warn",
            check_selfintersecting="skip",
            reorient_faces=True,
        ).show(
            style_mesh_selfintersecting_show=True,
            backend="matplotlib",
            return_fig=True,
        )
        assert len(record) == 2
        assert re.match(
            r"Unchecked selfintersecting mesh status in .* detected",
            str(record[0].message),
        )
        assert re.match(
            r"Self-intersecting mesh detected in .*.", str(record[1].message)
        )


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
    src1 = magpy.current.Circle(1, 2)
    src2 = magpy.current.Circle(1, 2)
    src1.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    src3 = magpy.current.Polyline(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Polyline(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, start=-1)
    src1.show(canvas=ax2, style_path_frames=2, style_arrow_size=0, return_fig=True)
    src2.show(canvas=ax2, style_arrow_sizemode="absolute", return_fig=True)
    src3.show(
        canvas=ax2, style_arrow_sizemode="absolute", style_arrow_size=0, return_fig=True
    )
    src4.show(canvas=ax2, style_path_frames=2, return_fig=True)


def test_matplotlib_model3d_extra():
    """test display extra model3d"""

    # using "plot"
    xs, ys, zs = [(1, 2)] * 3
    trace1 = {
        "backend": "matplotlib",
        "constructor": "plot",
        "args": (xs, ys, zs),
        "kwargs": {"ls": "-"},
    }
    obj1 = magpy.misc.Dipole(moment=(0, 0, 1))
    obj1.style.model3d.add_trace(**trace1)

    # using "plot_surface"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, 0 : np.pi : 6j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    trace2 = {
        "backend": "matplotlib",
        "constructor": "plot_surface",
        "args": (xs, ys, zs),
        "kwargs": {"cmap": plt.cm.YlGnBu_r},  # pylint: disable=no-member},
    }
    obj2 = magpy.Collection()
    obj2.style.model3d.add_trace(**trace2)

    # using "plot_trisurf"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, -0.5:0.5:6j]
    u, v = u.flatten(), v.flatten()
    xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    zs = 0.5 * v * np.sin(u / 2.0)
    tri = mtri.Triangulation(u, v)
    trace3 = {
        "backend": "matplotlib",
        "constructor": "plot_trisurf",
        "args": lambda: (xs, ys, zs),  # test callable args,
        "kwargs": {
            "triangles": tri.triangles,
            "cmap": plt.cm.Spectral,  # pylint: disable=no-member
        },
    }
    obj3 = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3, 0, 0))
    obj3.style.model3d.add_trace(**trace3)

    ax = plt.subplot(projection="3d")
    magpy.show(obj1, obj2, obj3, canvas=ax, return_fig=True)


def test_matplotlib_model3d_extra_bad_input():
    """test display extra model3d"""

    xs, ys, zs = [(1, 2)] * 3
    trace = {
        "backend": "matplotlib",
        "constructor": "plot",
        "kwargs": {"xs": xs, "ys": ys, "zs": zs},
        "coordsargs": {"x": "xs", "y": "ys", "z": "Z"},  # bad Z input
    }
    obj = magpy.misc.Dipole(moment=(0, 0, 1))
    with pytest.raises(ValueError):
        obj.style.model3d.add_trace(**trace)
        ax = plt.subplot(projection="3d")
        obj.show(canvas=ax, return_fig=True)


def test_matplotlib_model3d_extra_updatefunc():
    """test display extra model3d"""
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
    # pylint: disable=protected-access
    anim._draw_was_started = True  # avoid mpl test warning
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(anim, matplotlib.animation.FuncAnimation)


def test_subplots():
    """test subplots"""
    sensor = magpy.Sensor(
        pixel=np.linspace((0, 0, -0.2), (0, 0, 0.2), 2), style_size=1.5
    )
    sensor.style.label = "Sensor1"
    cyl1 = magpy.magnet.Cylinder(
        magnetization=(100, 0, 0), dimension=(1, 2), style_label="Cylinder1"
    )

    # define paths
    sensor.position = np.linspace((0, 0, -3), (0, 0, 3), 100)
    cyl1.position = (4, 0, 0)
    cyl1.rotate_from_angax(angle=np.linspace(0, 300, 100), start=0, axis="z", anchor=0)
    cyl2 = cyl1.copy().move((0, 0, 5))
    objs = cyl1, cyl2, sensor

    # with implicit axes
    fig = plt.figure(figsize=(20, 4))
    with magpy.show_context(
        backend="matplotlib", canvas=fig, animation=False, sumup=True, pixel_agg="mean"
    ) as s:
        s.show(*objs, col=1, output=("Bx", "By", "Bz"))  # from context
        magpy.show(cyl1, col=2)  # directly
        magpy.show({"objects": [cyl1, cyl2], "col": 3})  # as dict

    # with given axes in figure
    fig = plt.figure(figsize=(20, 4))
    fig.add_subplot(121, projection="3d")
    magpy.show(cyl1, col=2, canvas=fig)


def test_bad_show_inputs():
    """bad show inputs"""

    cyl1 = magpy.magnet.Cylinder(
        magnetization=(100, 0, 0), dimension=(1, 2), style_label="Cylinder1"
    )

    # test bad canvas
    with pytest.raises(TypeError, match=r"The `canvas` parameter must be one of .*"):
        magpy.show(cyl1, canvas="bad_canvas_input", backend="matplotlib")

    # test bad axes canvas with rows
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(131, projection="3d")
    with pytest.raises(
        ValueError,
        match=(
            r"Provided canvas is an instance of `matplotlib.axes.Axes` "
            r"and does not support `rows`.*"
        ),
    ):
        magpy.show(cyl1, canvas=ax, col=2, backend="matplotlib")

    # test conflicting output types
    sensor = magpy.Sensor(
        pixel=np.linspace((0, 0, -0.2), (0, 0, 0.2), 2), style_size=1.5
    )
    cyl1 = magpy.magnet.Cylinder(
        magnetization=(100, 0, 0), dimension=(1, 2), style_label="Cylinder1"
    )
    with pytest.raises(
        ValueError,
        match=r"Row/Col .* received conflicting output types.*",
    ):
        with magpy.show_context(animation=False, sumup=True, pixel_agg="mean") as s:
            s.show(cyl1, sensor, col=1, output="Bx")
            s.show(cyl1, sensor, col=1)

    # test unsupported specific args for some backends
    with pytest.warns(
        UserWarning,
        match=r"The 'plotly' backend does not support 'animation_output'.*",
    ):
        sensor = magpy.Sensor(
            position=np.linspace((0, 0, -0.2), (0, 0, 0.2), 200), style_size=1.5
        )
        magpy.show(
            sensor,
            backend="plotly",
            col=1,
            animation=True,
            animation_output="gif",
            return_fig=True,
        )


def test_show_context_reset():
    """show context reset"""
    ctx.reset(reset_show_return_value=True)
    with magpy.show_context(backend="matplotlib") as s:
        assert s.show_return_value is None
        s.show(magpy.Sensor(), return_fig=True)
    assert isinstance(s.show_return_value, mplFig)


def test_unset_excitations():
    """test show if mag, curr or mom are not set"""

    objs = [
        magpy.magnet.Cuboid(dimension=(1, 1, 1)),
        magpy.magnet.Cylinder(dimension=(1, 1)),
        magpy.magnet.CylinderSegment(dimension=(0, 1, 1, 45, 120)),
        magpy.magnet.Tetrahedron(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        magpy.magnet.TriangularMesh(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            faces=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        ),
        magpy.magnet.Sphere(diameter=1),
        magpy.misc.Triangle(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)]),
        magpy.misc.Dipole(),
        magpy.current.Polyline(vertices=[[0, -1, 0], [0, 1, 0]]),
        magpy.current.Circle(diameter=1, current=0),
    ]
    for i, o in enumerate(objs):
        o.move((i * 1.5, 0, 0))
    magpy.show(
        *objs,
        style_magnetization_mode="color+arrow",
        return_fig=True,
    )


def test_unset_objs():
    """test completely unset objects"""
    objs = [
        magpy.magnet.Cuboid(),
        magpy.magnet.Cylinder(),
        magpy.magnet.CylinderSegment(),
        magpy.magnet.Sphere(),
        magpy.magnet.Tetrahedron(),
        # magpy.magnet.TriangularMesh(), not possible yet
        magpy.misc.Triangle(),
        magpy.misc.Dipole(),
        magpy.current.Polyline(),
        magpy.current.Circle(),
    ]

    for i, o in enumerate(objs):
        o.move((1.5 * i, 0, 0))
    magpy.show(
        *objs,
        return_fig=True,
    )
