import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput, MagpylibMissingInput
import magpylib as magpy

###########################################################
###########################################################
# OBJECT INPUTS


def test_input_objects_position_good():
    """good positions"""
    goods = [
        (1,2,3),
        (0,0,0),
        ((1,2,3),(2,3,4)),
        [(2,3,4)],
        [2,3,4],
        [[2,3,4],[3,4,5]],
        [(2,3,4),(3,4,5)],
        np.array((1,2,3)),
        np.array(((1,2,3),(2,3,4))),
        ]
    for good in goods:
        sens = magpy.Sensor(position=good)
        np.testing.assert_allclose(sens.position, np.squeeze(np.array(good)))


def test_input_objects_position_bad():
    """bad positions"""
    bads = [
        (1,2),
        (1,2,3,4),
        [(1,2,3,4)]*2,
        (((1,2,3), (1,2,3)), ((1,2,3), (1,2,3))),
        'x',
        ['x','y','z'],
        dict(woot=15),
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Sensor, bad)


def test_input_objects_pixel_good():
    """good pixel"""
    goods = [
        (1,-2,3),
        (0,0,0),
        ((1,2,3),(2,3,4)),
        (((1,2,3),(2,-3,4)), ((1,2,3),(2,3,4))),
        [(2,3,4)],
        [2,3,4],
        [[-2,3,4],[3,4,5]],
        [[[2,3,4],[3,4,5]]]*4,
        [(2,3,4),(3,4,5)],
        np.array((1,2,-3)),
        np.array(((1,-2,3),(2,3,4))),
        ]
    for good in goods:
        sens = magpy.Sensor(pixel=good)
        np.testing.assert_allclose(sens.pixel, np.squeeze(np.array(good)))


def test_input_objects_pixel_bad():
    """bad pixel"""
    bads = [
        (1,2),
        (1,2,3,4),
        [(1,2,3,4)]*2,
        'x',
        ['x','y','z'],
        dict(woot=15),
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Sensor, (0,0,0), bad)


def test_input_objects_orientation_good():
    """good orientations (from rotvec)"""
    goods = [
        None,
        (.1,.2,.3),
        (0,0,0),
        [(.1,.2,.3)],
        [(.1,.2,.3)]*5,
        ]
    for good in goods:
        if good is None:
            sens = magpy.Sensor(orientation=None)
            np.testing.assert_allclose(sens.orientation.as_rotvec(), (0,0,0))
        else:
            sens = magpy.Sensor(orientation=R.from_rotvec(good))
            np.testing.assert_allclose(sens.orientation.as_rotvec(), np.squeeze(np.array(good)))


def test_input_objects_orientation_bad():
    """bad orienations"""
    bads = [
        (1,2),
        (1,2,3,4),
        [(1,2,3,4)]*2,
        'x',
        ['x','y','z'],
        dict(woot=15),
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.Sensor, (0,0,0), (0,0,0), bad)


def test_input_objects_current_good():
    """good currents"""
    goods = [
        None,
        0,
        1,
        1.2,
        np.array([1,2,3])[1],
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
    """bad current"""
    bads = [
        (1,2),
        [(1,2,3,4)]*2,
        'x',
        ['x','y','z'],
        dict(woot=15),
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.current.Loop, bad)


def test_input_objects_diameter_good():
    """good diameter"""
    goods = [
        None,
        0,
        1,
        1.2,
        np.array([1,2,3])[1],
        True,
        ]
    for good in goods:
        src = magpy.current.Loop(1, good)
        if good is None:
            assert src.diameter is None
        else:
            np.testing.assert_allclose(src.diameter, good)


def test_input_objects_diameter_bad():
    """bad diameter"""
    bads = [
        (1,2),
        [(1,2,3,4)]*2,
        'x',
        ['x','y','z'],
        dict(woot=15),
        -1,
        -1.123,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.current.Loop, 1, bad)


def test_input_objects_vertices_good():
    """good vertices"""
    goods = [
        None,
        ((0,0,0),(0,0,0)),
        ((1,2,3),(2,3,4)),
        [(2,3,4), (-1,-2,-3)]*2,
        [[2,3,4],[3,4,5]],
        np.array(((1,2,3),(2,3,4))),
        ]
    for good in goods:
        src = magpy.current.Line(1, good)
        if good is None:
            assert src.vertices is None
        else:
            np.testing.assert_allclose(src.vertices, good)


def test_input_objects_vertices_bad():
    """bad vertices"""
    bads = [
        (1,2),
        [(1,2,3,4)]*2,
        [(1,2,3)],
        'x',
        ['x','y','z'],
        dict(woot=15),
        0,
        -1.123,
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.current.Line, 1, bad)


def test_input_objects_magnetization_moment_good():
    """good magnetization and moment"""
    goods = [
        None,
        (1,2,3),
        (0,0,0),
        [-1,-2,-3],
        np.array((1,2,3)),
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
    """bad magnetization and moment"""
    bads = [
        (1,2),
        [1,2,3,4],
        [(1,2,3)]*2,
        np.array([(1,2,3)]*2),
        'x',
        ['x','y','z'],
        dict(woot=15),
        0,
        -1.123,
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.magnet.Cuboid, bad)
        np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.Dipole, bad)


def test_input_objects_dimension_cuboid_good():
    """good cuboid dimension"""
    goods = [
        None,
        (1,2,3),
        [11,22,33],
        np.array((1,2,3)),
        ]
    for good in goods:
        src = magpy.magnet.Cuboid((1,1,1), good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cuboid_bad():
    """bad cuboid dimension"""
    bads = [
        [-1,2,3],
        (0,1,2),
        (1,2),
        [1,2,3,4],
        [(1,2,3)]*2,
        np.array([(1,2,3)]*2),
        'x',
        ['x','y','z'],
        dict(woot=15),
        0,
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.magnet.Cuboid, (1,1,1), bad)


def test_input_objects_dimension_cylinder_good():
    """good cylinder dimension"""
    goods = [
        None,
        (1,2),
        [11,22],
        np.array((1,2)),
        ]
    for good in goods:
        src = magpy.magnet.Cylinder((1,1,1), good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cylinder_bad():
    """bad cylinder dimension"""
    bads = [
        [-1,2],
        (0,1),
        (1,),
        [1,2,3],
        [(1,2)]*2,
        np.array([(2,3)]*2),
        'x',
        ['x','y'],
        dict(woot=15),
        0,
        True,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.magnet.Cylinder, (1,1,1), bad)


def test_input_objects_dimension_cylinderSegment_good():
    """good cylinder segment dimension"""
    goods = [
        None,
        (0,2,3,0,50),
        (1,2,3,40,50),
        [11,22,33,44,360],
        [11,22,33,-44,55],
        np.array((1,2,3,4,5)),
        [11,22,33,-44,-33],
        (0,2,3,-10,0),
        ]
    for good in goods:
        src = magpy.magnet.CylinderSegment((1,1,1), good)
        if good is None:
            assert src.dimension is None
        else:
            np.testing.assert_allclose(src.dimension, good)


def test_input_objects_dimension_cylinderSegment_bad():
    """bad cylinder segment dimension"""
    bads = [
        (1,2,3,4),
        (1,2,3,4,5,6),
        (0,0,3,4,5),
        (2,1,3,4,5),
        (-1,2,3,4,5),
        (1,2,0,4,5),
        (1,2,-1,4,5),
        (1,2,3,5,4),
        [(1,2,3,4,5)]*2,
        np.array([(1,2,3,4,5)]*2),
        'x',
        ['x','y','z',1,2],
        dict(woot=15),
        0,
        True
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.magnet.CylinderSegment, (1,1,1), bad)


def test_input_objects_fiedBHlambda_good():
    """good custom fiedBHlambda"""
    def f(x):
        """3 in 3 out"""
        return x
    src = magpy.misc.CustomSource(field_B_lambda=f, field_H_lambda=f)

    np.testing.assert_allclose(src.getB((1,2,3)), (1,2,3))
    np.testing.assert_allclose(src.getH((1,2,3)), (1,2,3))


def test_input_objects_fiedBHlambda_bad():
    """bad custom fiedBlambda"""
    def f(x):
        """bad fieldBH lambda"""
        return 1
    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, f)
    np.testing.assert_raises(MagpylibBadUserInput, magpy.misc.CustomSource, field_H_lambda=f)



###########################################################
###########################################################
# DISPLAY

def test_input_show_zoom_bad():
    """bad show zoom inputs"""
    x = magpy.Sensor()
    bads = [
        (1,2,3),
        -1,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.show, x, zoom=bad)


def test_input_show_animation_bad():
    """bad show animation inputs"""
    x = magpy.Sensor()
    bads = [
        (1,2,3),
        -1,
        ]
    for bad in bads:
        np.testing.assert_raises(MagpylibBadUserInput, magpy.show, x, animation=bad)


def test_input_show_backend_bad():
    """bad show backend inputs"""
    x = magpy.Sensor()
    bads = [
        (1,2,3),
        -1,
        'x',
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
    s = magpy.magnet.Cuboid(dimension=(1,2,3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Cylinder(dimension=(1,2))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.CylinderSegment(dimension=(1,2,3,4,5))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Sphere(diameter=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Loop(diameter=1)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.current.Line(vertices=[(1,2,3)]*2)
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)


def test_input_show_missing_parameters3():
    """missing inputs"""
    s = magpy.magnet.Cuboid(magnetization=(1,2,3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Cylinder(magnetization=(1,2,3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.CylinderSegment(magnetization=(1,2,3))
    np.testing.assert_raises(MagpylibMissingInput, magpy.show, s)
    s = magpy.magnet.Sphere(magnetization=(1,2,3))
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
        'auto',
        0,
        1,
        15,
        -2,
        -250,
        np.array((1,2,3))[0],
        ]
    for good in goods:
        x = magpy.Sensor(position=[(0,0,i) for i in range(10)])
        x.move((1,0,0), start=good)
        assert isinstance(x.position, np.ndarray)


def test_input_move_start_bad():
    """bad start inputs"""
    bads = [
        1.1,
        1.,
        'x',
        None,
        [11],
        (1,),
        np.array([(1,2,3,4,5)]*2),
        dict(woot=15),
        ]
    for bad in bads:
        x = magpy.Sensor(position=[(0,0,i) for i in range(10)])
        np.testing.assert_raises(MagpylibBadUserInput, x.move, (1,1,1), start=bad)


def test_input_rotate_degrees_good():
    """good degrees inputs"""
    goods = [
        True,
        False,
        ]
    for good in goods:
        x = magpy.Sensor()
        x.rotate_from_angax(10, 'z', degrees=good)
        assert isinstance(x.position, np.ndarray)


def test_input_rotate_degrees_bad():
    """bad degrees inputs"""
    bads = [
        1,
        0,
        1.1,
        1.,
        'x',
        None,
        [True],
        (1,),
        np.array([(1,2,3,4,5)]*2),
        dict(woot=15),
        ]
    for bad in bads:
        x = magpy.Sensor()
        np.testing.assert_raises(MagpylibBadUserInput, x.rotate_from_angax, 10, 'z', degrees=bad)


def test_input_rotate_axis_good():
    """good rotate axis inputs"""
    goods = [
        (1,2,3),
        (0,0,1),
        [0,0,1],
        np.array([0,0,1]),
        'x',
        'y',
        'z',
        ]
    for good in goods:
        x = magpy.Sensor()
        x.rotate_from_angax(10, good)
        assert isinstance(x.position, np.ndarray)


def test_input_rotate_axis_bad():
    """bad rotate axis inputs"""
    bads = [
        (0,0,0),
        (1,2),
        (1,2,3,4),
        1.1,
        1,
        'xx',
        None,
        True,
        np.array([(1,2,3,4,5)]*2),
        dict(woot=15),
        ]
    for bad in bads:
        x = magpy.Sensor()
        np.testing.assert_raises(MagpylibBadUserInput, x.rotate_from_angax, 10, bad)


###########################################################
###########################################################
# GET BH


def test_input_getBH_field_good():
    """good getBH field inputs"""
    goods = [
        'B',
        'H',
        ]
    for good in goods:
        moms = np.array([[1,2,3]])
        obs = np.array([[1,2,3]])
        B = magpy.core.dipole_field(moms, obs, field=good)
        assert isinstance(B, np.ndarray)


def test_input_getBH_field_bad():
    """bad getBH field inputs"""
    bads = [
        1,
        0,
        1.1,
        1.,
        'x',
        None,
        [True],
        (1,),
        np.array([(1,2,3,4,5)]*2),
        dict(woot=15),
        ]
    for bad in bads:
        moms = np.array([[1,2,3]])
        obs = np.array([[1,2,3]])
        np.testing.assert_raises(MagpylibBadUserInput, magpy.core.dipole_field, moms, obs, field=bad)
