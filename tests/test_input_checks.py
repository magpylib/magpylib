import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput
import magpylib as magpy


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
        (1,2,3),
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


# a3 = np.array([1,2,3])
# a2 = np.array([1,2])
# a234 = np.array([[[1,2]]*3]*4)
# r0 = R.from_quat((0,0,0,1))

# # Check error throwing when dimension or excitation is not initialized
# def init_dim_display1():
#     """throw dimension init error from display"""
#     src1 = magpy.current.Loop(current=1)
#     src2 = magpy.magnet.Sphere(magnetization=(1,2,3))
#     magpy.show(src1, src2)

# def init_dim_display2():
#     """throw dimension init error from display"""
#     src1 = magpy.current.Loop(current=1, diameter=2)
#     src2 = magpy.magnet.Cuboid(magnetization=(1,2,3))
#     magpy.show(src1, src2)

# def init_dim_display3():
#     """throw dimension init error from display"""
#     src1 = magpy.current.Loop(current=1, diameter=2)
#     src2 = magpy.current.Line(current=1)
#     magpy.show(src1, src2)

# def init_dim_getBH1():
#     """throw dimension init error from getB"""
#     src1 = magpy.current.Loop(current=1)
#     src2 = magpy.magnet.Sphere(magnetization=(1,2,3))
#     magpy.getB([src1,src2], (1,2,3))

# def init_excite_display1():
#     """throw excitation init error from display"""
#     src1 = magpy.current.Loop(diameter=1)
#     src2 = magpy.magnet.Sphere(diameter=2)
#     magpy.show(src1, src2, style_magnetization_show=True, style_arrow_show=True)

# def init_excite_getBH1():
#     """throw excitation init error from getB"""
#     src1 = magpy.current.Loop(diameter=1, current=1)
#     src2 = magpy.magnet.Sphere(diameter=2)
#     magpy.getB([src1, src2], (1,2,3))

# def init_excite_getBH2():
#     """throw excitation init error from getB"""
#     src1 = magpy.current.Loop(current=1, diameter=2)
#     src2 = magpy.magnet.Cuboid(dimension=(1,2,3))
#     magpy.getB([src1, src2], (1,2,3))

# def init_excite_getBH3():
#     """throw excitation init error from getB"""
#     src1 = magpy.current.Loop(current=1, diameter=2)
#     src2 = magpy.current.Line(vertices=[(1,2,3),(2,3,4)])
#     magpy.getB([src1, src2], (1,2,3))

# def init_excite_getBH4():
#     """throw excitation init error from getB"""
#     src1 = magpy.current.Loop(current=1, diameter=2)
#     src2 = magpy.misc.Dipole()
#     magpy.getB([src1, src2], (1,2,3))

# # BAD BASEGEO INPUTS --------------------------------------
# def badInput_BaseGeo1():
#     """ bad pos input type"""
#     BaseGeo(123, r0)

# def badInput_BaseGeo2():
#     """ bad pos input format1"""
#     BaseGeo(a2, r0)

# def badInput_BaseGeo3():
#     """ bad pos input format2"""
#     BaseGeo(a234, r0)

# def badInput_BaseGeo4():
#     """ bad rot input type"""
#     BaseGeo(a3, 123)


# # BAD BASEGEO .MOVE INPUT--------------------------------------------------
# def badInput_move1():
#     """ bad displacement input type"""
#     bg = BaseGeo(a3, r0)
#     bg.move(123)

# def badInput_move2():
#     """ bad start input type"""
#     bg = BaseGeo(a3, r0)
#     bg.move([1,2,3], start=1.1)

# def badInput_move3():
#     """ bad displacement input format1"""
#     bg = BaseGeo(a3, r0)
#     bg.move([1,2])

# def badInput_move4():
#     """ bad displacement input format2"""
#     bg = BaseGeo(a3, r0)
#     bg.move([[[1,2,3]]*3]*4)


# # BAD BASEGEO .ROTATE INPUT------------------------------------------------
# def badInput_rotate1():
#     """ bad rot input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate(123)

# def badInput_rotate2():
#     """ bad anchor input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate(r0, anchor=123)

# def badInput_rotate3():
#     """ bad start input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate(r0, start=1.23)

# def badInput_rotate5():
#     """ bad anchor input format"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate(r0, anchor=(1,2,3,4))


# # BAD BASEGEO .rotate_from_angax INPUT-----------------------------------------
# def badInput_rotate_from_angax1():
#     """ bad ang input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax('123', (1,2,3))

# def badInput_rotate_from_angax2():
#     """ bad axis input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, 1)

# def badInput_rotate_from_angax3():
#     """ bad anchor input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, (1,2,3), anchor=1)

# def badInput_rotate_from_angax4():
#     """ bad start input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, (1,2,3), start=1.1)

# def badInput_rotate_from_angax6():
#     """ bad degrees input type"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, (1,2,3), degrees=None)

# def badInput_rotate_from_angax7():
#     """ bad angle input format"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax([[1,2,3]]*2, (1,2,3))

# def badInput_rotate_from_angax8():
#     """ bad axis input format"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax([1,2,3], [1,2,3,4])

# def badInput_rotate_from_angax9():
#     """ bad axis input format"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, [0,0,0])

# def badInput_rotate_from_angax10():
#     """ bad axis input format"""
#     bg = BaseGeo(a3, r0)
#     bg.rotate_from_angax(123, [1,2,3], anchor=[[(1,2,3)]*2]*2)


# # EXCITATIONS -----------------------------------------

# def badMag_input1():
#     """bad magnetization input type"""
#     magpy.magnet.Cuboid(magnetization='woot', dimension=a3)

# def badMag_input2():
#     """bad magnetization input format"""
#     magpy.magnet.Cuboid(magnetization=(1,2,3,4), dimension=a3)

# def badCurrent_input1():
#     """bad current input type"""
#     magpy.current.Loop(current='1', diameter=1)

# # DIMENSIONS --------------------------------------------

# def bad_dim_input1():
#     """cuboid dim type"""
#     magpy.magnet.Cuboid(magnetization=a3, dimension=1)

# def bad_dim_input3():
#     """cuboid dim format"""
#     magpy.magnet.Cuboid(magnetization=a3, dimension=(1,2))

# def bad_dim_input4():
#     """cylinder dim type"""
#     magpy.magnet.Cylinder(magnetization=a3, dimension=1)
# def bad_dim_input6():
#     """cylinder bad dim shape"""
#     magpy.magnet.Cylinder(magnetization=a3, dimension=(1,2,0,45))

# def bad_dim_input7():
#     """Sphere dim type"""
#     magpy.magnet.Sphere(magnetization=a3, diameter=(1,1))

# def bad_dim_input9():
#     """Loop dim type"""
#     magpy.current.Loop(current=1, diameter=(1,1))

# def bad_dim_input10():
#     """cylinder segment bad dim shape"""
#     magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,0,45))
# def bad_dim_input11():
#     """cylinder segment bad dim d1>d2"""
#     magpy.magnet.CylinderSegment(magnetization=a3, dimension=(3,2,2,0,45))
# def bad_dim_input12():
#     """cylinder segment bad dim phi1>phi2"""
#     magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,2,100,45))
# def bad_dim_input13():
#     """cylinder segment bad dim phi2-phi1>360"""
#     magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,2,0,1145))


# # MISC SOURCE ------------------------------------------------------------

# def bad_misc_input1():
#     """Sensor pixel type"""
#     magpy.Sensor(pixel=1)
# def bad_misc_input2():
#     """Sensor pixel format"""
#     magpy.Sensor(pixel=[[1,2]]*3)

# def bad_misc_input3():
#     """Line vertex type"""
#     magpy.current.Line(1, '1')
# def bad_misc_input4():
#     """Line vertex format 1"""
#     magpy.current.Line(1, [(1,2,3)])
# def bad_misc_input5():
#     """Line vertex format 2"""
#     magpy.current.Line(1, [[(1,2,3)]*2]*2)
# def bad_misc_input6():
#     """Line vertex format 3"""
#     magpy.current.Line(1, [(1,2)]*2)


# # OBSERVER ------------------------------------------------------------

# src = magpy.current.Loop(current=1, diameter=1)
# sens = magpy.Sensor()
# def bad_observer_input1():
#     """getBH observer format"""
#     magpy.getB(src, a234)
# def bad_observer_input2():
#     """getBH observer format"""
#     magpy.getB(src, [sens, [(1,2),(1,2)]])
# def bad_observer_input3():
#     """getBH observer format"""
#     magpy.getB(src, [(1,2),(1,2)])
# def bad_observer_input4():
#     """bad observer type"""
#     magpy.getB(src, 123)


# # test bad level0 inputs
# def bad_level0_field_input_0():
#     """sphere"""
#     mag = np.array([(100,200,300)])
#     dim1 = np.array((2,))
#     obs = np.array([(1,2,3)])
#     magpy.core.magnet_sphere_field(mag, dim1, obs, field='x')

# def bad_level0_field_input_1():
#     """cylinder"""
#     mag = np.array([(100,200,300)])
#     dim2 = np.array([(2,2)])
#     obs = np.array([(1,2,3)])
#     magpy.core.magnet_cylinder_field(mag, dim2, obs, field='y')

# def bad_level0_field_input_2():
#     """cylinder segment"""
#     mag = np.array([(100,200,300)])
#     dim5 = np.array([(0,1,2,0,360)])
#     obs = np.array([(1,2,3)])
#     magpy.core.magnet_cylinder_segment_field(mag, dim5, obs, field=123)

# def bad_level0_field_input_3():
#     """dipole"""
#     mag = np.array([(100,200,300)])
#     obs = np.array([(1,2,3)])
#     magpy.core.dipole_field(mag, obs, field=mag)

# def bad_level0_field_input_4():
#     """loop"""
#     cur = np.array((20,))
#     dim1 = np.array((2,))
#     obs = np.array([(1,2,3)])
#     magpy.core.current_loop_field(cur, dim1, obs, field=None)

# def bad_level0_field_input_5():
#     """line"""
#     cur = np.array((20,))
#     start = np.array([(1,1,1)])
#     end = np.array([(3,3,3)])
#     obs = np.array([(1,2,3)])
#     magpy.core.current_line_field(cur, start, end, obs, field=(1,2,3))

# def bad_level0_field_input_6():
#     """cuboid"""
#     mag = np.array([(100,200,300)])
#     dim3 = np.array([(2,2,2)])
#     obs = np.array([(1,2,3)])
#     magpy.core.magnet_cuboid_field(mag, dim3, obs, field='F')

# class TestExceptions(unittest.TestCase):
#     """ test class for exception testing """
#     def test_level0_field_inputs(self):
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_0)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_1)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_2)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_3)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_4)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_5)
#         self.assertRaises(MagpylibBadUserInput, bad_level0_field_input_6)

#     def test_init(self):
#         """ missing inputs when calling display and getB"""
#         self.assertRaises(MagpylibMissingInput, init_dim_display1)
#         self.assertRaises(MagpylibMissingInput, init_dim_display2)
#         self.assertRaises(MagpylibMissingInput, init_dim_display3)
#         self.assertRaises(MagpylibMissingInput, init_dim_getBH1)
#         self.assertRaises(MagpylibMissingInput, init_excite_display1)
#         self.assertRaises(MagpylibMissingInput, init_excite_getBH1)
#         self.assertRaises(MagpylibMissingInput, init_excite_getBH2)
#         self.assertRaises(MagpylibMissingInput, init_excite_getBH3)
#         self.assertRaises(MagpylibMissingInput, init_excite_getBH4)

#     def test_BaseGeo(self):
#         """ bad BG inputs"""
#         self.assertRaises(MagpylibBadUserInput, badInput_BaseGeo1)
#         self.assertRaises(MagpylibBadInputShape, badInput_BaseGeo2)
#         self.assertRaises(MagpylibBadInputShape, badInput_BaseGeo3)
#         self.assertRaises(MagpylibBadUserInput, badInput_BaseGeo4)

#     def test_move(self):
#         """ bad .move inputs"""
#         self.assertRaises(MagpylibBadUserInput, badInput_move1)
#         self.assertRaises(MagpylibBadUserInput, badInput_move2)
#         self.assertRaises(MagpylibBadInputShape, badInput_move3)
#         self.assertRaises(MagpylibBadInputShape, badInput_move4)

#     def test_rotate(self):
#         """ bad .rotate inputs"""
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate1)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate2)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate3)
#         self.assertRaises(MagpylibBadInputShape, badInput_rotate5)

#     def test_rotate_from_angax(self):
#         """ bad .rotate_from_angax inputs"""
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax1)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax2)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax3)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax4)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax6)
#         self.assertRaises(MagpylibBadInputShape, badInput_rotate_from_angax7)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax8)
#         self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax9)
#         self.assertRaises(MagpylibBadInputShape, badInput_rotate_from_angax10)

#     def test_magnetization_input(self):
#         """ bad magnetization inputs"""
#         self.assertRaises(MagpylibBadUserInput, badMag_input1)
#         self.assertRaises(MagpylibBadInputShape, badMag_input2)
#         self.assertRaises(MagpylibBadUserInput, badCurrent_input1)

#     def test_dim_inputs(self):
#         """ bad dimension inputs"""
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input1)
#         self.assertRaises(MagpylibBadInputShape, bad_dim_input3)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input4)
#         self.assertRaises(MagpylibBadInputShape, bad_dim_input6)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input7)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input9)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input10)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input11)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input12)
#         self.assertRaises(MagpylibBadUserInput, bad_dim_input13)

#     def test_misc_source_inputs(self):
#         """ bad misc source inputs"""
#         self.assertRaises(MagpylibBadUserInput, bad_misc_input1)
#         self.assertRaises(MagpylibBadInputShape, bad_misc_input2)
#         self.assertRaises(MagpylibBadUserInput, bad_misc_input3)
#         self.assertRaises(MagpylibBadInputShape, bad_misc_input4)
#         self.assertRaises(MagpylibBadInputShape, bad_misc_input5)
#         self.assertRaises(MagpylibBadInputShape, bad_misc_input6)

#     def test_observer_inputs(self):
#         """ bad observer inputs"""
#         self.assertRaises(MagpylibBadInputShape, bad_observer_input1)
#         self.assertRaises(MagpylibBadInputShape, bad_observer_input2)
#         self.assertRaises(MagpylibBadInputShape, bad_observer_input3)
#         self.assertRaises(MagpylibBadUserInput, bad_observer_input4)
