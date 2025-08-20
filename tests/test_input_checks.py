import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibDeprecationWarning,
    MagpylibMissingInput,
)
from magpylib._src.fields.field_BH_dipole import BHJM_dipole

# pylint: disable=unnecessary-lambda-assignment

###########################################################
###########################################################
# OBJECT INPUTS


@pytest.mark.parametrize(
    "position",
    [
        (1, 2, 3),
        (0, 0, 0),
        ((1, 2, 3), (2, 3, 4)),
        [(2, 3, 4)],
        [2, 3, 4],
        [[2, 3, 4], [3, 4, 5]],
        [(2, 3, 4), (3, 4, 5)],
        np.array((1, 2, 3)),
        np.array(((1, 2, 3), (2, 3, 4))),
    ],
)
def test_input_objects_position_good(position):
    """good input: magpy.Sensor(position=position)"""

    sens = magpy.Sensor(position=position)
    np.testing.assert_allclose(sens.position, np.squeeze(np.array(position)))


@pytest.mark.parametrize(
    "position",
    [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        (((1, 2, 3), (1, 2, 3)), ((1, 2, 3), (1, 2, 3))),
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        True,
    ],
)
def test_input_objects_position_bad(position):
    """bad input: magpy.Sensor(position=position)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.Sensor(position=position)


@pytest.mark.parametrize(
    "pixel",
    [
        (1, -2, 3),
        (0, 0, 0),
        ((1, 2, 3), (2, 3, 4)),
        (((1, 2, 3), (2, -3, 4)), ((1, 2, 3), (2, 3, 4))),
        [(2, 3, 4)],
        [2, 3, 4],
        [[-2, 3, 4], [3, 4, 5]],
        [[[2, 3, 4], [3, 4, 5]]] * 4,
        [(2, 3, 4), (3, 4, 5)],
        np.array((1, 2, -3)),
        np.array(((1, -2, 3), (2, 3, 4))),
    ],
)
def test_input_objects_pixel_good(pixel):
    """good input: magpy.Sensor(pixel=pixel)"""

    sens = magpy.Sensor(pixel=pixel)
    np.testing.assert_allclose(sens.pixel, pixel)


@pytest.mark.parametrize(
    "pixel",
    [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        True,
    ],
)
def test_input_objects_pixel_bad(pixel):
    """bad input: magpy.Sensor(pixel=pixel)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.Sensor(position=(0, 0, 0), pixel=pixel)


@pytest.mark.parametrize(
    "orientation_rotvec",
    [
        None,
        (0.1, 0.2, 0.3),
        (0, 0, 0),
        [(0.1, 0.2, 0.3)],
        [(0.1, 0.2, 0.3)] * 5,
    ],
)
def test_input_objects_orientation_good(orientation_rotvec):
    """good input: magpy.Sensor(orientation=orientation_rotvec)"""

    if orientation_rotvec is None:
        sens = magpy.Sensor(orientation=None)
        np.testing.assert_allclose(sens.orientation.as_rotvec(), (0, 0, 0))
    else:
        sens = magpy.Sensor(orientation=R.from_rotvec(orientation_rotvec))
        np.testing.assert_allclose(
            sens.orientation.as_rotvec(), np.squeeze(np.array(orientation_rotvec))
        )


@pytest.mark.parametrize(
    "orientation_rotvec",
    [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        True,
    ],
)
def test_input_objects_orientation_bad(orientation_rotvec):
    """bad input: magpy.Sensor(orientation=orientation_rotvec)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.Sensor(
            position=(0, 0, 0), pixel=(0, 0, 0), orientation=orientation_rotvec
        )


@pytest.mark.parametrize(
    "current",
    [
        None,
        0,
        1,
        1.2,
        np.array([1, 2, 3])[1],
        -1,
        -1.123,
        True,
    ],
)
def test_input_objects_current_good(current):
    """good input: magpy.current.Circle(current)"""

    src = magpy.current.Circle(current=current)
    if current is None:
        assert src.current is None
    else:
        np.testing.assert_allclose(src.current, current)


@pytest.mark.parametrize(
    "current",
    [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        {"woot": 15},
    ],
)
def test_input_objects_current_bad(current):
    """bad input: magpy.current.Circle(current)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.current.Circle(current)


@pytest.mark.parametrize(
    "diameter",
    [
        None,
        0,
        1,
        1.2,
        np.array([1, 2, 3])[1],
        True,
    ],
)
def test_input_objects_diameter_good(diameter):
    """good input: magpy.current.Circle(diameter=inp)"""

    src = magpy.current.Circle(diameter=diameter)
    if diameter is None:
        assert src.diameter is None
    else:
        np.testing.assert_allclose(src.diameter, diameter)


@pytest.mark.parametrize(
    "diameter",
    [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        -1,
        -1.123,
    ],
)
def test_input_objects_diameter_bad(diameter):
    """bad input: magpy.current.Circle(diameter=diameter)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.current.Circle(diameter=diameter)


@pytest.mark.parametrize(
    "vertices",
    [
        None,
        ((0, 0, 0), (0, 0, 0)),
        ((1, 2, 3), (2, 3, 4)),
        [(2, 3, 4), (-1, -2, -3)] * 2,
        [[2, 3, 4], [3, 4, 5]],
        np.array(((1, 2, 3), (2, 3, 4))),
    ],
)
def test_input_objects_vertices_good(vertices):
    """good input: magpy.current.Polyline(vertices=vertices)"""

    src = magpy.current.Polyline(vertices=vertices)
    if vertices is None:
        assert src.vertices is None
    else:
        np.testing.assert_allclose(src.vertices, vertices)


@pytest.mark.parametrize(
    "vertices",
    [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        [(1, 2, 3)],
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        0,
        -1.123,
        True,
    ],
)
def test_input_objects_vertices_bad(vertices):
    """bad input: magpy.current.Polyline(vertices=vertices)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.current.Polyline(vertices=vertices)


@pytest.mark.parametrize(
    "pol_or_mom",
    [
        None,
        (1, 2, 3),
        (0, 0, 0),
        [-1, -2, -3],
        np.array((1, 2, 3)),
    ],
)
def test_input_objects_magnetization_moment_good(pol_or_mom):
    """
    good input:
        magpy.magnet.Cuboid(magnetization=moment),
        magpy.misc.Dipole(moment=moment)
    """

    src = magpy.magnet.Cuboid(polarization=pol_or_mom)
    src2 = magpy.misc.Dipole(moment=pol_or_mom)
    if pol_or_mom is None:
        assert src.polarization is None
        assert src2.moment is None
    else:
        np.testing.assert_allclose(src.polarization, pol_or_mom)
        np.testing.assert_allclose(src2.moment, pol_or_mom)


@pytest.mark.parametrize(
    "moment",
    [
        (1, 2),
        [1, 2, 3, 4],
        [(1, 2, 3)] * 2,
        np.array([(1, 2, 3)] * 2),
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        0,
        -1.123,
        True,
    ],
)
def test_input_objects_magnetization_moment_bad(moment):
    """
    bad input:
        magpy.magnet.Cuboid(magnetization=moment),
        magpy.misc.Dipole(moment=moment)
    """

    with pytest.raises(MagpylibBadUserInput):
        magpy.magnet.Cuboid(magnetization=moment)
    with pytest.raises(MagpylibBadUserInput):
        magpy.misc.Dipole(moment=moment)


@pytest.mark.parametrize(
    "dimension",
    [
        None,
        (1, 2, 3),
        [11, 22, 33],
        np.array((1, 2, 3)),
    ],
)
def test_input_objects_dimension_cuboid_good(dimension):
    """good input: magpy.magnet.Cuboid(dimension=dimension)"""

    src = magpy.magnet.Cuboid(dimension=dimension)
    if dimension is None:
        assert src.dimension is None
    else:
        np.testing.assert_allclose(src.dimension, dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        [-1, 2, 3],
        (0, 1, 2),
        (1, 2),
        [1, 2, 3, 4],
        [(1, 2, 3)] * 2,
        np.array([(1, 2, 3)] * 2),
        "x",
        ["x", "y", "z"],
        {"woot": 15},
        0,
        True,
    ],
)
def test_input_objects_dimension_cuboid_bad(dimension):
    """bad input: magpy.magnet.Cuboid(dimension=dimension)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.magnet.Cuboid(dimension=dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        None,
        (1, 2),
        [11, 22],
        np.array((1, 2)),
    ],
)
def test_input_objects_dimension_cylinder_good(dimension):
    """good input: magpy.magnet.Cylinder(dimension=dimension)"""

    src = magpy.magnet.Cylinder(dimension=dimension)
    if dimension is None:
        assert src.dimension is None
    else:
        np.testing.assert_allclose(src.dimension, dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        [-1, 2],
        (0, 1),
        (1,),
        [1, 2, 3],
        [(1, 2)] * 2,
        np.array([(2, 3)] * 2),
        "x",
        ["x", "y"],
        {"woot": 15},
        0,
        True,
    ],
)
def test_input_objects_dimension_cylinder_bad(dimension):
    """bad input: magpy.magnet.Cylinder(dimension=dimension)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.magnet.Cylinder(dimension=dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        None,
        (0, 2, 3, 0, 50),
        (1, 2, 3, 40, 50),
        [11, 22, 33, 44, 360],
        [11, 22, 33, -44, 55],
        np.array((1, 2, 3, 4, 5)),
        [11, 22, 33, -44, -33],
        (0, 2, 3, -10, 0),
    ],
)
def test_input_objects_dimension_cylinderSegment_good(dimension):
    """good input: magpy.magnet.CylinderSegment(dimension=dimension)"""

    src = magpy.magnet.CylinderSegment(dimension=dimension)
    if dimension is None:
        assert src.dimension is None
    else:
        np.testing.assert_allclose(src.dimension, dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5, 6),
        (0, 0, 3, 4, 5),
        (2, 1, 3, 4, 5),
        (-1, 2, 3, 4, 5),
        (1, 2, 0, 4, 5),
        (1, 2, -1, 4, 5),
        (1, 2, 3, 5, 4),
        [(1, 2, 3, 4, 5)] * 2,
        np.array([(1, 2, 3, 4, 5)] * 2),
        "x",
        ["x", "y", "z", 1, 2],
        {"woot": 15},
        0,
        True,
    ],
)
def test_input_objects_dimension_cylinderSegment_bad(dimension):
    """good input: magpy.magnet.CylinderSegment(dimension=dimension)"""

    with pytest.raises(MagpylibBadUserInput):
        magpy.magnet.CylinderSegment(dimension=dimension)


def test_input_objects_field_func_good():
    """good input: magpy.misc.CustomSource(field_func=f)"""
    # pylint: disable=unused-argument

    # init empty = None
    src = magpy.misc.CustomSource()
    np.testing.assert_raises(MagpylibMissingInput, src.getB, (1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, src.getH, (1, 2, 3))

    # None
    src = magpy.misc.CustomSource(field_func=None)
    np.testing.assert_raises(MagpylibMissingInput, src.getB, (1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, src.getH, (1, 2, 3))

    # acceptable func with B and H return
    def f(field, observers):  # noqa : ARG001
        """3 in 3 out"""
        return observers

    src = magpy.misc.CustomSource(field_func=f)
    np.testing.assert_allclose(src.getB((1, 2, 3)), (1, 2, 3))
    np.testing.assert_allclose(src.getH((1, 2, 3)), (1, 2, 3))

    # acceptable func with only B return
    def ff(field, observers):
        """3 in 3 out"""
        if field == "B":
            return observers
        return None

    src = magpy.misc.CustomSource(field_func=ff)
    np.testing.assert_allclose(src.getB((1, 2, 3)), (1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, src.getH, (1, 2, 3))

    # acceptable func with only B return
    def fff(field, observers):
        """3 in 3 out"""
        if field == "H":
            return observers
        return None

    src = magpy.misc.CustomSource(field_func=fff)
    np.testing.assert_raises(MagpylibMissingInput, src.getB, (1, 2, 3))
    np.testing.assert_allclose(src.getH((1, 2, 3)), (1, 2, 3))


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(1, id="non-callable"),
        pytest.param(lambda fieldd, observers, whatever: None, id="bad-arg-names"),  # noqa: ARG005
        pytest.param(
            lambda field, observers: 1 if field == "B" else None,  # noqa: ARG005
            id="no-ndarray-return-on-B",
        ),
        pytest.param(
            lambda field, observers: (1 if field == "H" else observers),
            id="no-ndarray-return-on-H",
        ),
        pytest.param(
            lambda field, observers: (  # noqa: ARG005
                np.array([1, 2, 3]) if field == "B" else None
            ),
            id="bad-return-shape-on-B",
        ),
        pytest.param(
            lambda field, observers: (
                np.array([1, 2, 3]) if field == "H" else observers
            ),
            id="bad-return-shape-on-H",
        ),
    ],
)
def test_input_objects_field_func_bad(func):
    """bad input: magpy.misc.CustomSource(field_func=f)"""
    with pytest.raises(
        MagpylibBadUserInput, match=r"Input parameter `field_func` must .*."
    ):
        magpy.misc.CustomSource(field_func=func)


def test_missing_input_triangular_mesh():
    """missing input checks for TriangularMesh"""

    verts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tris = np.array([(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)])

    with pytest.raises(MagpylibMissingInput):
        magpy.magnet.TriangularMesh(faces=tris)

    with pytest.raises(MagpylibMissingInput):
        magpy.magnet.TriangularMesh(vertices=verts)


###########################################################
###########################################################
# DISPLAY


@pytest.mark.parametrize(
    "zoom",
    [
        (1, 2, 3),
        -1,
    ],
)
def test_input_show_zoom_bad(zoom):
    """bad show zoom inputs"""
    x = magpy.Sensor()
    with pytest.raises(MagpylibBadUserInput):
        magpy.show(x, zoom=zoom, return_fig=True, backend="plotly")


@pytest.mark.parametrize(
    "animation",
    [
        (1, 2, 3),
        -1,
    ],
)
def test_input_show_animation_bad(animation):
    """bad show animation inputs"""
    x = magpy.Sensor()
    with pytest.raises(MagpylibBadUserInput):
        magpy.show(x, animation=animation)


@pytest.mark.parametrize(
    "backend",
    [
        (1, 2, 3),
        -1,
        "x",
        True,
    ],
)
def test_input_show_backend_bad(backend):
    """bad show backend inputs"""
    x = magpy.Sensor()
    with pytest.raises(MagpylibBadUserInput):
        magpy.show(x, backend=backend)


###########################################################
###########################################################
# MOVE ROTATE


@pytest.mark.parametrize(
    "start_value",
    [
        "auto",
        0,
        1,
        15,
        -2,
        -250,
        np.array((1, 2, 3))[0],
    ],
)
def test_input_move_start_good(start_value):
    """good start inputs"""
    x = magpy.Sensor(position=[(0, 0, i) for i in range(10)])
    x.move((1, 0, 0), start=start_value)
    assert isinstance(x.position, np.ndarray)


@pytest.mark.parametrize(
    "start_value",
    [
        1.1,
        1.0,
        "x",
        None,
        [11],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        {"woot": 15},
    ],
)
def test_input_move_start_bad(start_value):
    """bad start inputs"""
    x = magpy.Sensor(position=[(0, 0, i) for i in range(10)])
    with pytest.raises(MagpylibBadUserInput):
        x.move((1, 1, 1), start=start_value)


@pytest.mark.parametrize("degrees", [True, False])
def test_input_rotate_degrees_good(degrees):
    """good degrees inputs"""
    x = magpy.Sensor(position=(0, 0, 1))
    x.rotate_from_angax(ang := 1.2345, "y", degrees=degrees, anchor=0)
    if degrees:
        ang = np.deg2rad(ang)
    np.testing.assert_allclose(x.position, [np.sin(ang), 0, np.cos(ang)])


@pytest.mark.parametrize(
    "degrees",
    [
        1,
        0,
        1.1,
        1.0,
        "x",
        None,
        [True],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        {"woot": 15},
    ],
)
def test_input_rotate_degrees_bad(degrees):
    """bad degrees inputs"""
    x = magpy.Sensor()
    with pytest.raises(MagpylibBadUserInput):
        x.rotate_from_angax(10, "z", degrees=degrees)


@pytest.mark.parametrize(
    "axis",
    [
        (1, 2, 3),
        (0, 0, 1),
        [0, 0, 1],
        np.array([0, 0, 1]),
        "x",
        "y",
        "z",
    ],
)
def test_input_rotate_axis_good(axis):
    """good rotate axis inputs"""
    x = magpy.Sensor()
    x.rotate_from_angax(10, axis)
    assert isinstance(x.position, np.ndarray)


@pytest.mark.parametrize(
    "axis",
    [
        (0, 0, 0),
        (1, 2),
        (1, 2, 3, 4),
        1.1,
        1,
        "xx",
        None,
        True,
        np.array([(1, 2, 3, 4, 5)] * 2),
        {"woot": 15},
    ],
)
def test_input_rotate_axis_bad(axis):
    """bad rotate axis inputs"""
    x = magpy.Sensor()
    with pytest.raises(MagpylibBadUserInput):
        x.rotate_from_angax(10, axis)


@pytest.mark.parametrize(
    "observers",
    [
        magpy.Sensor(position=(1, 1, 1)),
        magpy.Collection(magpy.Sensor(position=(1, 1, 1))),
        magpy.Collection(magpy.Sensor(), magpy.Sensor()),
        (1, 2, 3),
        [(1, 2, 3)] * 2,
        [[(1, 2, 3)] * 2] * 3,
        [magpy.Sensor(), magpy.Collection(magpy.Sensor())],
        [magpy.Sensor(), magpy.Collection(magpy.Sensor(), magpy.Sensor())],
        [magpy.Sensor(), (1, 2, 3)],
        [magpy.Sensor(pixel=[[(1, 2, 3)] * 2] * 3), [[(1, 2, 3)] * 2] * 3],
        [(1, 2, 3), magpy.Collection(magpy.Sensor())],
        [(1, 2, 3), magpy.Collection(magpy.Sensor(), magpy.Sensor())],
        [magpy.Sensor(), magpy.Collection(magpy.Sensor()), (1, 2, 3)],
        [magpy.Sensor(), magpy.Collection(magpy.Sensor()), magpy.Sensor(), (1, 2, 3)],
    ],
)
def test_input_observers_good(observers):
    """good observers input"""
    src = magpy.misc.Dipole(moment=(1, 2, 3))
    B = src.getB(observers)
    print(B)
    assert isinstance(B, np.ndarray)


@pytest.mark.parametrize(
    "observers",
    [
        "a",
        None,
        [],
        ("a", "b", "c"),
        [("a", "b", "c")],
        magpy.misc.Dipole(moment=(1, 2, 3)),
        [(1, 2, 3), [(1, 2, 3)] * 2],
        [magpy.Sensor(), [(1, 2, 3)] * 2],
        [[(1, 2, 3)] * 2, magpy.Collection(magpy.Sensor())],
        [magpy.Sensor(pixel=(1, 2, 3)), ("a", "b", "c")],
    ],
)
def test_input_observers_bad(observers):
    """bad observers input"""
    src = magpy.misc.Dipole(moment=(1, 2, 3))
    with pytest.raises(MagpylibBadUserInput):
        src.getB(observers)


@pytest.mark.parametrize(
    "children",
    [
        [magpy.Sensor()],
        [magpy.magnet.Cuboid()],
        [magpy.Collection()],
        [magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection()],
        [
            magpy.Sensor(),
            magpy.Sensor(),
            magpy.magnet.Cuboid(),
            magpy.magnet.Cuboid(),
            magpy.Collection(),
            magpy.Collection(),
        ],
        [[magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection()]],
        [(magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection())],
    ],
)
def test_input_collection_good(children):
    """good inputs: collection(inp)"""
    col = magpy.Collection(*children)
    assert isinstance(col, magpy.Collection)


@pytest.mark.parametrize(
    "children",
    [
        "some_string",
        None,
        True,
        1,
        np.array((1, 2, 3)),
        [magpy.Sensor(), [magpy.magnet.Cuboid(), magpy.Collection()]],
    ],
)
def test_input_collection_bad(children):
    """bad inputs: collection(inp)"""
    with pytest.raises(MagpylibBadUserInput):
        magpy.Collection(children)


@pytest.mark.parametrize(
    "children",
    [
        [magpy.Sensor()],
        [magpy.magnet.Cuboid()],
        [magpy.Collection()],
        [magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection()],
        [
            magpy.Sensor(),
            magpy.Sensor(),
            magpy.magnet.Cuboid(),
            magpy.magnet.Cuboid(),
            magpy.Collection(),
            magpy.Collection(),
        ],
        [[magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection()]],
        [(magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection())],
    ],
)
def test_input_collection_add_good(children):
    """good inputs: collection.add(children)"""
    col = magpy.Collection()
    col.add(*children)
    assert isinstance(col, magpy.Collection)


@pytest.mark.parametrize(
    "children",
    [
        "some_string",
        None,
        True,
        1,
        np.array((1, 2, 3)),
        ([magpy.Sensor(), [magpy.magnet.Cuboid(), magpy.Collection()]],),
    ],
)
def test_input_collection_add_bad(children):
    """bad inputs: collection.add(children)"""
    col = magpy.Collection()
    with pytest.raises(MagpylibBadUserInput):
        col.add(children)


@pytest.mark.parametrize(
    "children",
    [
        [magpy.Sensor()],
        [magpy.magnet.Cuboid()],
        [magpy.Collection()],
        [magpy.Sensor(), magpy.magnet.Cuboid(), magpy.Collection()],
        [[magpy.Sensor(), magpy.magnet.Cuboid()]],
        [(magpy.Sensor(), magpy.magnet.Cuboid())],
    ],
)
def test_input_collection_remove_good(children):
    """good inputs: collection.remove(children)"""
    col = magpy.Collection(*children)
    assert col.children == (
        list(children[0]) if isinstance(children[0], tuple | list) else children
    )
    col.remove(*children)
    assert not col.children


@pytest.mark.parametrize(
    "children",
    [
        "some_string",
        None,
        True,
        1,
        np.array((1, 2, 3)),
        [magpy.Sensor(), [magpy.Sensor()]],
    ],
)
def test_input_collection_remove_bad(children):
    """bad inputs: collection.remove(children)"""
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cuboid()
    c1 = magpy.Collection()
    col = magpy.Collection(x1, x2, s1, s2, c1)

    with pytest.raises(MagpylibBadUserInput):
        col.remove(children)


def test_input_collection_bad_errors_arg():
    """bad errors input"""
    x1 = magpy.Sensor()
    col = magpy.Collection()
    with pytest.raises(MagpylibBadUserInput):
        col.remove(x1, errors="w00t")


@pytest.mark.parametrize("parent", [magpy.Collection(), None])
def test_input_basegeo_parent_setter_good(parent):
    """good inputs: obj.parent=parent"""
    x = magpy.Sensor()
    x.parent = parent
    assert x.parent == parent


@pytest.mark.parametrize(
    "parent",
    [
        "some_string",
        [],
        True,
        1,
        np.array((1, 2, 3)),
        [magpy.Collection()],
        magpy.Sensor(),
        magpy.magnet.Cuboid(),
    ],
)
def test_input_basegeo_parent_setter_bad(parent):
    """bad inputs: obj.parent=parent"""
    x = magpy.Sensor()

    with pytest.raises(MagpylibBadUserInput):
        x.parent = parent

    # when obj is good but has already a parent
    x = magpy.Sensor()
    magpy.Collection(x)

    with pytest.raises(MagpylibBadUserInput):
        magpy.Collection(x)


###########################################################
###########################################################
# GET BH


@pytest.mark.parametrize("field", ["B", "H"])
def test_input_getBH_field_good(field):
    """good getBH field inputs"""
    moms = np.array([[1, 2, 3]])
    obs = np.array([[1, 2, 3]])
    B = BHJM_dipole(field=field, observers=obs, moment=moms)
    assert isinstance(B, np.ndarray)


@pytest.mark.parametrize(
    "field",
    [
        1,
        0,
        1.1,
        1.0,
        "x",
        None,
        [True],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        {"woot": 15},
    ],
)
def test_input_getBH_field_bad(field):
    """bad getBH field inputs"""
    moms = np.array([[1, 2, 3]])
    obs = np.array([[1, 2, 3]])
    with pytest.raises(MagpylibBadUserInput):
        BHJM_dipole(field=field, observers=obs, moment=moms)


def test_sensor_handedness():
    """Test if handedness input"""
    magpy.Sensor(handedness="right")
    magpy.Sensor(handedness="left")
    with pytest.raises(
        MagpylibBadUserInput,
        match=r"Sensor `handedness` must be either `'right'` or `'left'`",
    ):
        magpy.Sensor(handedness="not_right_or_left")


def test_magnet_polarization_magnetization_input():
    """test codependency and magnetization polarization inputs"""
    # warning when magnetization is too low -> polarization confusion
    mag = np.array([1, 2, 3]) * 1e6

    with pytest.warns(
        MagpylibDeprecationWarning,
        match=r".* received a very low magnetization. .*",
    ):
        magpy.magnet.Cuboid(magnetization=[1, 2, 3])

    # both polarization and magnetization at the same time
    with pytest.raises(
        ValueError,
        match=r"The attributes magnetization and polarization are dependent. .*",
    ):
        magpy.magnet.Cuboid(polarization=[1, 2, 3], magnetization=mag)

    # setting magnetization afterwards
    c = magpy.magnet.Cuboid()
    c.magnetization = mag
    np.testing.assert_allclose(mag, c.magnetization)
    np.testing.assert_allclose(mag * (4 * np.pi * 1e-7), c.polarization)


def test_current_sheet_init():
    """good inputs"""
    cds = [(1, 0, 0)] * 5
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 3))

    magpy.current.TriangleSheet(
        current_densities=cds,
        vertices=verts,
        faces=facs,
    )


def test_current_strip_init():
    """good inputs"""
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    magpy.current.TriangleStrip(
        current=1,
        vertices=verts,
    )


def test_current_sheet_init_bad():
    """bad inputs"""

    # verts.len < 3
    cds = [(1, 0, 0)] * 5
    verts = ((0, 0, 0), (1, 0, 0))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 3))

    with pytest.raises(ValueError):  # noqa: PT011
        magpy.current.TriangleSheet(
            current_densities=cds,
            vertices=verts,
            faces=facs,
        )

    # faces.len != cds.len
    cds = [(1, 0, 0)] * 3
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 3))

    with pytest.raises(ValueError):  # noqa: PT011
        magpy.current.TriangleSheet(
            current_densities=cds,
            vertices=verts,
            faces=facs,
        )

    # bad face indices
    cds = [(1, 0, 0)] * 5
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 33))

    with pytest.raises(IndexError):
        magpy.current.TriangleSheet(
            current_densities=cds,
            vertices=verts,
            faces=facs,
        )

    # bad input formats
    cds = 1
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 33))

    with pytest.raises(MagpylibBadUserInput):
        magpy.current.TriangleSheet(
            current_densities=cds,
            vertices=verts,
            faces=facs,
        )

    # bad input formats
    cds = [(1, 2)] * 5
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    facs = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3), (1, 2, 33))

    with pytest.raises(MagpylibBadUserInput):
        magpy.current.TriangleSheet(
            current_densities=cds,
            vertices=verts,
            faces=facs,
        )


def test_current_strip_init_bad():
    """bad inputs"""
    verts = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    curr = [1] * 4
    with pytest.raises(MagpylibBadUserInput):
        magpy.current.TriangleStrip(
            current=curr,
            vertices=verts,
        )

    verts = (
        (0, 0, 0),
        (1, 0, 0),
    )
    curr = 1
    with pytest.raises(MagpylibBadUserInput):
        magpy.current.TriangleStrip(
            current=curr,
            vertices=verts,
        )
