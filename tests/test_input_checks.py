import numpy as np
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.exceptions import MagpylibMissingInput


###########################################################
###########################################################
# OBJECT INPUTS


def test_input_objects_position_good():
    """good input: magpy.Sensor(position=inp)"""
    goods = [
        (1, 2, 3),
        (0, 0, 0),
        ((1, 2, 3), (2, 3, 4)),
        [(2, 3, 4)],
        [2, 3, 4],
        [[2, 3, 4], [3, 4, 5]],
        [(2, 3, 4), (3, 4, 5)],
        np.array((1, 2, 3)),
        np.array(((1, 2, 3), (2, 3, 4))),
    ]
    for good in goods:
        sens = magpy.Sensor(position=good)
        np.testing.assert_allclose(sens.position, np.squeeze(np.array(good)))


def test_input_objects_position_bad():
    """bad input: magpy.Sensor(position=inp)"""
    bads = [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        (((1, 2, 3), (1, 2, 3)), ((1, 2, 3), (1, 2, 3))),
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        True,
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Sensor, bad)


def test_input_objects_pixel_good():
    """good input: magpy.Sensor(pixel=inp)"""
    goods = [
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
    ]
    for good in goods:
        sens = magpy.Sensor(pixel=good)
        np.testing.assert_allclose(sens.pixel, good)


def test_input_objects_pixel_bad():
    """bad input: magpy.Sensor(pixel=inp)"""
    bads = [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        True,
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Sensor, (0, 0, 0), bad)


def test_input_objects_orientation_good():
    """good input: magpy.Sensor(orientation=inp)"""
    goods = [
        None,
        (0.1, 0.2, 0.3),
        (0, 0, 0),
        [(0.1, 0.2, 0.3)],
        [(0.1, 0.2, 0.3)] * 5,
    ]
    for good in goods:
        if good is None:
            sens = magpy.Sensor(orientation=None)
            np.testing.assert_allclose(sens.orientation.as_rotvec(), (0, 0, 0))
        else:
            sens = magpy.Sensor(orientation=R.from_rotvec(good))
            np.testing.assert_allclose(
                sens.orientation.as_rotvec(), np.squeeze(np.array(good))
            )


def test_input_objects_orientation_bad():
    """bad input: magpy.Sensor(orientation=inp)"""
    bads = [
        (1, 2),
        (1, 2, 3, 4),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        True,
    ]
    for bad in bads:
        np.testing.assert_raises(
            MagpylibBadUserInput, magpy.Sensor, (0, 0, 0), (0, 0, 0), bad
        )


def test_input_objects_current_good():
    """good input: magpy.current.Loop(inp)"""
    goods = [
        None,
        0,
        1,
        1.2,
        np.array([1, 2, 3])[1],
        -1,
        -1.123,
        True,
    ]
    for good in goods:
        src = magpy.current.Loop(good)
        if good is None:
            assert src.current is None
        else:
            np.testing.assert_allclose(src.current, good)


def test_input_objects_current_bad():
    """bad input: magpy.current.Loop(inp)"""
    bads = [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        dict(woot=15),
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.current.Loop, bad)


def test_input_objects_diameter_good():
    """good input: magpy.current.Loop(diameter=inp)"""
    goods = [
        None,
        0,
        1,
        1.2,
        np.array([1, 2, 3])[1],
        True,
    ]
    for good in goods:
        src = magpy.current.Loop(diameter=good)
        if good is None:
            assert src.diameter is None
        else:
            np.testing.assert_allclose(src.diameter, good)


def test_input_objects_diameter_bad():
    """bad input: magpy.current.Loop(diameter=inp)"""
    bads = [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        -1,
        -1.123,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.current.Loop(diameter=bad)


def test_input_objects_vertices_good():
    """good input: magpy.current.Line(vertices=inp)"""
    goods = [
        None,
        ((0, 0, 0), (0, 0, 0)),
        ((1, 2, 3), (2, 3, 4)),
        [(2, 3, 4), (-1, -2, -3)] * 2,
        [[2, 3, 4], [3, 4, 5]],
        np.array(((1, 2, 3), (2, 3, 4))),
    ]
    for good in goods:
        src = magpy.current.Line(vertices=good)
        if good is None:
            assert src.vertices is None
        else:
            np.testing.assert_allclose(src.vertices, good)


def test_input_objects_vertices_bad():
    """bad input: magpy.current.Line(vertices=inp)"""
    bads = [
        (1, 2),
        [(1, 2, 3, 4)] * 2,
        [(1, 2, 3)],
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        0,
        -1.123,
        True,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.current.Line(vertices=bad)


def test_input_objects_magnetization_moment_good():
    """
    good input:
        magpy.magnet.Cuboid(magnetization=inp),
        magpy.misc.Dipole(moment=inp)
    """
    goods = [
        None,
        (1, 2, 3),
        (0, 0, 0),
        [-1, -2, -3],
        np.array((1, 2, 3)),
    ]
    for good in goods:
        src = magpy.magnet.Cuboid(good)
        src2 = magpy.misc.Dipole(good)
        if good is None:
            assert src.magnetization is None
            assert src2.moment is None
        else:
            np.testing.assert_allclose(src.magnetization, good)
            np.testing.assert_allclose(src2.moment, good)


def test_input_objects_magnetization_moment_bad():
    """
    bad input:
        magpy.magnet.Cuboid(magnetization=inp),
        magpy.misc.Dipole(moment=inp)
    """
    bads = [
        (1, 2),
        [1, 2, 3, 4],
        [(1, 2, 3)] * 2,
        np.array([(1, 2, 3)] * 2),
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        0,
        -1.123,
        True,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.magnet.Cuboid(magnetization=bad)
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.misc.Dipole(moment=bad)


def test_input_objects_dimension_cuboid_good():
    """good input: magpy.magnet.Cuboid(dimension=inp)"""
    goods = [
        None,
        (1, 2, 3),
        [11, 22, 33],
        np.array((1, 2, 3)),
    ]
    for good in goods:
        src = magpy.magnet.Cuboid(dimension=good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cuboid_bad():
    """bad input: magpy.magnet.Cuboid(dimension=inp)"""
    bads = [
        [-1, 2, 3],
        (0, 1, 2),
        (1, 2),
        [1, 2, 3, 4],
        [(1, 2, 3)] * 2,
        np.array([(1, 2, 3)] * 2),
        "x",
        ["x", "y", "z"],
        dict(woot=15),
        0,
        True,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.magnet.Cuboid(dimension=bad)


def test_input_objects_dimension_cylinder_good():
    """good input: magpy.magnet.Cylinder(dimension=inp)"""
    goods = [
        None,
        (1, 2),
        [11, 22],
        np.array((1, 2)),
    ]
    for good in goods:
        src = magpy.magnet.Cylinder(dimension=good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cylinder_bad():
    """bad input: magpy.magnet.Cylinder(dimension=inp)"""
    bads = [
        [-1, 2],
        (0, 1),
        (1,),
        [1, 2, 3],
        [(1, 2)] * 2,
        np.array([(2, 3)] * 2),
        "x",
        ["x", "y"],
        dict(woot=15),
        0,
        True,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.magnet.Cylinder(dimension=bad)


def test_input_objects_dimension_cylinderSegment_good():
    """good input: magpy.magnet.CylinderSegment(dimension=inp)"""
    goods = [
        None,
        (0, 2, 3, 0, 50),
        (1, 2, 3, 40, 50),
        [11, 22, 33, 44, 360],
        [11, 22, 33, -44, 55],
        np.array((1, 2, 3, 4, 5)),
        [11, 22, 33, -44, -33],
        (0, 2, 3, -10, 0),
    ]
    for good in goods:
        src = magpy.magnet.CylinderSegment(dimension=good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cylinderSegment_bad():
    """good input: magpy.magnet.CylinderSegment(dimension=inp)"""
    bads = [
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
        dict(woot=15),
        0,
        True,
    ]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            magpy.magnet.CylinderSegment(dimension=bad)


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
    def f(field, observers):
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


def test_input_objects_field_func_bad():
    """bad input: magpy.misc.CustomSource(field_func=f)"""
    # pylint: disable=unused-argument

    # non callable
    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, 1)

    # bad arg names
    def ff(fieldd, observers, whatever):
        """ff"""

    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, ff)

    # no ndarray return on B
    def fff(field, observers):
        """fff"""
        if field == "B":
            return 1

    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, fff)

    # no ndarray return on H
    def ffff(field, observers):
        """ffff"""
        if field == "H":
            return 1
        return observers

    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, ffff)

    # bad return shape on B
    def g(field, observers):
        """g"""
        if field == "B":
            return np.array([1, 2, 3])

    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, g)

    # bad return shape on H
    def gg(field, observers):
        """gg"""
        if field == "H":
            return np.array([1, 2, 3])
        return observers

    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, gg)


###########################################################
###########################################################
# DISPLAY


def test_input_show_zoom_bad():
    """bad show zoom inputs"""
    x = magpy.Sensor()
    bads = [
        (1, 2, 3),
        -1,
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.show, x, zoom=bad)


def test_input_show_animation_bad():
    """bad show animation inputs"""
    x = magpy.Sensor()
    bads = [
        (1, 2, 3),
        -1,
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.show, x, animation=bad)


def test_input_show_backend_bad():
    """bad show backend inputs"""
    x = magpy.Sensor()
    bads = [
        (1, 2, 3),
        -1,
        "x",
        True,
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.show, x, backend=bad)


def test_input_show_missing_parameters1():
    """missing inputs"""
    s = magpy.magnet.Cuboid()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Cylinder()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.CylinderSegment()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Sphere()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Loop()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Line()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.misc.Dipole()
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)


def test_input_show_missing_parameters2():
    """missing inputs"""
    s = magpy.magnet.Cuboid(dimension=(1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Cylinder(dimension=(1, 2))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.CylinderSegment(dimension=(1, 2, 3, 4, 5))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Sphere(diameter=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Loop(diameter=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Line(vertices=[(1, 2, 3)] * 2)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)


def test_input_show_missing_parameters3():
    """missing inputs"""
    s = magpy.magnet.Cuboid(magnetization=(1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Cylinder(magnetization=(1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.CylinderSegment(magnetization=(1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Sphere(magnetization=(1, 2, 3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Loop(current=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Line(current=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)


###########################################################
###########################################################
# MOVE ROTATE


def test_input_move_start_good():
    """good start inputs"""
    goods = [
        "auto",
        0,
        1,
        15,
        -2,
        -250,
        np.array((1, 2, 3))[0],
    ]
    for good in goods:
        x = magpy.Sensor(position=[(0, 0, i) for i in range(10)])
        x.move((1, 0, 0), start=good)
        assert isinstance(x.position, np.ndarray)


def test_input_move_start_bad():
    """bad start inputs"""
    bads = [
        1.1,
        1.0,
        "x",
        None,
        [11],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        dict(woot=15),
    ]
    for bad in bads:
        x = magpy.Sensor(position=[(0, 0, i) for i in range(10)])
        np.testing.assert_raises(MagpylibBadUserInput, x.move, (1, 1, 1), start=bad)


def test_input_rotate_degrees_good():
    """good degrees inputs"""
    goods = [
        True,
        False,
    ]
    for good in goods:
        x = magpy.Sensor()
        x.rotate_from_angax(10, "z", degrees=good)
        assert isinstance(x.position, np.ndarray)


def test_input_rotate_degrees_bad():
    """bad degrees inputs"""
    bads = [
        1,
        0,
        1.1,
        1.0,
        "x",
        None,
        [True],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        dict(woot=15),
    ]
    for bad in bads:
        x = magpy.Sensor()
        np.testing.assert_raises(
            MagpylibBadUserInput, x.rotate_from_angax, 10, "z", degrees=bad
        )


def test_input_rotate_axis_good():
    """good rotate axis inputs"""
    goods = [
        (1, 2, 3),
        (0, 0, 1),
        [0, 0, 1],
        np.array([0, 0, 1]),
        "x",
        "y",
        "z",
    ]
    for good in goods:
        x = magpy.Sensor()
        x.rotate_from_angax(10, good)
        assert isinstance(x.position, np.ndarray)


def test_input_rotate_axis_bad():
    """bad rotate axis inputs"""
    bads = [
        (0, 0, 0),
        (1, 2),
        (1, 2, 3, 4),
        1.1,
        1,
        "xx",
        None,
        True,
        np.array([(1, 2, 3, 4, 5)] * 2),
        dict(woot=15),
    ]
    for bad in bads:
        x = magpy.Sensor()
        np.testing.assert_raises(MagpylibBadUserInput, x.rotate_from_angax, 10, bad)


def test_input_observers_good():
    """good observers input"""
    pos_vec1 = (1, 2, 3)
    pos_vec2 = [(1, 2, 3)] * 2
    pos_vec3 = [[(1, 2, 3)] * 2] * 3
    sens1 = magpy.Sensor()
    sens2 = magpy.Sensor()
    sens3 = magpy.Sensor()
    sens4 = magpy.Sensor(pixel=pos_vec3)
    coll1 = magpy.Collection(sens1)
    coll2 = magpy.Collection(sens2, sens3)

    goods = [
        sens1,
        coll1,
        coll2,
        pos_vec1,
        pos_vec2,
        pos_vec3,
        [sens1, coll1],
        [sens1, coll2],
        [sens1, pos_vec1],
        [sens4, pos_vec3],
        [pos_vec1, coll1],
        [pos_vec1, coll2],
        [sens1, coll1, pos_vec1],
        [sens1, coll1, sens2, pos_vec1],
    ]

    src = magpy.misc.Dipole((1, 2, 3))
    for good in goods:
        B = src.getB(good)
        assert isinstance(B, np.ndarray)


def test_input_observers_bad():
    """bad observers input"""
    pos_vec1 = (1, 2, 3)
    pos_vec2 = [(1, 2, 3)] * 2
    sens1 = magpy.Sensor()
    coll1 = magpy.Collection(sens1)

    bads = [
        "a",
        None,
        [],
        ("a", "b", "c"),
        [("a", "b", "c")],
        magpy.misc.Dipole((1, 2, 3)),
        [pos_vec1, pos_vec2],
        [sens1, pos_vec2],
        [pos_vec2, coll1],
        [magpy.Sensor(pixel=(1, 2, 3)), ("a", "b", "c")],
    ]
    src = magpy.misc.Dipole((1, 2, 3))
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, src.getB, bad)


def test_input_collection_good():
    """good inputs: collection(inp)"""
    # pylint: disable=unnecessary-lambda
    x = lambda: magpy.Sensor()
    s = lambda: magpy.magnet.Cuboid()
    c = lambda: magpy.Collection()

    goods = [  # unpacked
        [x()],
        [s()],
        [c()],
        [x(), s(), c()],
        [x(), x(), s(), s(), c(), c()],
        [[x(), s(), c()]],
        [(x(), s(), c())],
    ]

    for good in goods:
        col = magpy.Collection(*good)
        assert getattr(col, "_object_type", "") == "Collection"


def test_input_collection_bad():
    """bad inputs: collection(inp)"""
    # pylint: disable=unnecessary-lambda
    x = lambda: magpy.Sensor()
    s = lambda: magpy.magnet.Cuboid()
    c = lambda: magpy.Collection()

    bads = [
        "some_string",
        None,
        True,
        1,
        np.array((1, 2, 3)),
        [x(), [s(), c()]],
    ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Collection, bad)


def test_input_collection_add_good():
    """good inputs: collection.add(inp)"""
    # pylint: disable=unnecessary-lambda
    x = lambda: magpy.Sensor()
    s = lambda: magpy.magnet.Cuboid()
    c = lambda: magpy.Collection()

    goods = [  # unpacked
        [x()],
        [s()],
        [c()],
        [x(), s(), c()],
        [x(), x(), s(), s(), c(), c()],
        [[x(), s(), c()]],
        [(x(), s(), c())],
    ]

    for good in goods:
        col = magpy.Collection()
        col.add(*good)
        assert getattr(col, "_object_type", "") == "Collection"


def test_input_collection_add_bad():
    """bad inputs: collection.add(inp)"""
    # pylint: disable=unnecessary-lambda
    x = lambda: magpy.Sensor()
    s = lambda: magpy.magnet.Cuboid()
    c = lambda: magpy.Collection()

    bads = [
        "some_string",
        None,
        True,
        1,
        np.array((1, 2, 3)),
        [x(), [s(), c()]],
    ]
    for bad in bads:
        col = magpy.Collection()
        np.testing.assert_raises(MagpylibBadUserInput, col.add, bad)


def test_input_collection_remove_good():
    """good inputs: collection.remove(inp)"""
    x = magpy.Sensor()
    s = magpy.magnet.Cuboid()
    c = magpy.Collection()

    goods = [  # unpacked
        [x],
        [s],
        [c],
        [x, s, c],
        [[x, s]],
        [(x, s)],
    ]

    for good in goods:
        col = magpy.Collection(*good)
        assert col.children == (
            list(good[0]) if isinstance(good[0], (tuple, list)) else good
        )
        col.remove(*good)
        assert not col.children


def test_input_collection_remove_bad():
    """bad inputs: collection.remove(inp)"""
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cuboid()
    c1 = magpy.Collection()
    c2 = magpy.Collection()
    col = magpy.Collection(x1, x2, s1, s2, c1)

    bads = ["some_string", None, True, 1, np.array((1, 2, 3)), [x1, [x2]]]
    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            col.remove(bad)

    # bad errors input
    with np.testing.assert_raises(MagpylibBadUserInput):
        col.remove(c2, errors="w00t")


def test_input_basegeo_parent_setter_good():
    """good inputs: obj.parent=inp"""
    x = magpy.Sensor()
    c = magpy.Collection()

    goods = [
        c,
        None,
    ]

    for good in goods:
        x.parent = good
        assert x.parent == good


def test_input_basegeo_parent_setter_bad():
    """bad inputs: obj.parent=inp"""
    x = magpy.Sensor()
    c = magpy.Collection()

    bads = [
        "some_string",
        [],
        True,
        1,
        np.array((1, 2, 3)),
        [c],
        magpy.Sensor(),
        magpy.magnet.Cuboid(),
    ]

    for bad in bads:
        with np.testing.assert_raises(MagpylibBadUserInput):
            x.parent = bad

    # when obj is good but has already a parent
    x = magpy.Sensor()
    magpy.Collection(x)
    with np.testing.assert_raises(MagpylibBadUserInput):
        magpy.Collection(x)


###########################################################
###########################################################
# GET BH


def test_input_getBH_field_good():
    """good getBH field inputs"""
    goods = [
        "B",
        "H",
    ]
    for good in goods:
        moms = np.array([[1, 2, 3]])
        obs = np.array([[1, 2, 3]])
        B = magpy.core.dipole_field(good, obs, moms)
        assert isinstance(B, np.ndarray)


def test_input_getBH_field_bad():
    """bad getBH field inputs"""
    bads = [
        1,
        0,
        1.1,
        1.0,
        "x",
        None,
        [True],
        (1,),
        np.array([(1, 2, 3, 4, 5)] * 2),
        dict(woot=15),
    ]
    for bad in bads:
        moms = np.array([[1, 2, 3]])
        obs = np.array([[1, 2, 3]])
        np.testing.assert_raises(
            MagpylibBadUserInput,
            magpy.core.dipole_field,
            bad,
            obs,
            moms,
        )
