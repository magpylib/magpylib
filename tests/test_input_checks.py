import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.exceptions import (MagpylibBadUserInput,
    MagpylibBadInputShape, MagpylibMissingInput)
import magpylib as magpy

a3 = np.array([1,2,3])
a2 = np.array([1,2])
a234 = np.array([[[1,2]]*3]*4)
r0 = R.from_quat((0,0,0,1))

# Check error throwing when dimension or excitation is not initialized
def init_dim_display1():
    """throw dimension init error from display"""
    src1 = magpy.current.Circular(current=1)
    src2 = magpy.magnet.Sphere(magnetization=(1,2,3))
    magpy.display(src1, src2)

def init_dim_display2():
    """throw dimension init error from display"""
    src1 = magpy.current.Circular(current=1, diameter=2)
    src2 = magpy.magnet.Cuboid(magnetization=(1,2,3))
    magpy.display(src1, src2)

def init_dim_display3():
    """throw dimension init error from display"""
    src1 = magpy.current.Circular(current=1, diameter=2)
    src2 = magpy.current.Line(current=1)
    magpy.display(src1, src2)

def init_dim_getBH1():
    """throw dimension init error from getB"""
    src1 = magpy.current.Circular(current=1)
    src2 = magpy.magnet.Sphere(magnetization=(1,2,3))
    magpy.getB([src1,src2], (1,2,3))

def init_exite_display1():
    """throw excitation init error from display"""
    src1 = magpy.current.Circular(diameter=1)
    src2 = magpy.magnet.Sphere(diameter=2)
    magpy.display(src1, src2, style_magnetization_show=True, style_current_show=True)

def init_exite_getBH1():
    """throw excitation init error from getB"""
    src1 = magpy.current.Circular(diameter=1, current=1)
    src2 = magpy.magnet.Sphere(diameter=2)
    magpy.getB([src1, src2], (1,2,3))

def init_exite_getBH2():
    """throw excitation init error from getB"""
    src1 = magpy.current.Circular(current=1, diameter=2)
    src2 = magpy.magnet.Cuboid(dimension=(1,2,3))
    magpy.getB([src1, src2], (1,2,3))

def init_exite_getBH3():
    """throw excitation init error from getB"""
    src1 = magpy.current.Circular(current=1, diameter=2)
    src2 = magpy.current.Line(vertices=[(1,2,3),(2,3,4)])
    magpy.getB([src1, src2], (1,2,3))

def init_exite_getBH4():
    """throw excitation init error from getB"""
    src1 = magpy.current.Circular(current=1, diameter=2)
    src2 = magpy.misc.Dipole()
    magpy.getB([src1, src2], (1,2,3))

# BAD BASEGEO INPUTS --------------------------------------
def badInput_BaseGeo1():
    """ bad pos input type"""
    BaseGeo(123, r0)

def badInput_BaseGeo2():
    """ bad pos input format1"""
    BaseGeo(a2, r0)

def badInput_BaseGeo3():
    """ bad pos input format2"""
    BaseGeo(a234, r0)

def badInput_BaseGeo4():
    """ bad rot input type"""
    BaseGeo(a3, 123)


# BAD BASEGEO .MOVE INPUT--------------------------------------------------
def badInput_move1():
    """ bad displacement input type"""
    bg = BaseGeo(a3, r0)
    bg.move(123)

def badInput_move2():
    """ bad start input type"""
    bg = BaseGeo(a3, r0)
    bg.move([1,2,3], start=1.1)

def badInput_move3():
    """ bad increment input type"""
    bg = BaseGeo(a3, r0)
    bg.move([1,2,3], increment=5)

def badInput_move4():
    """ bad displacement input format1"""
    bg = BaseGeo(a3, r0)
    bg.move([1,2])

def badInput_move5():
    """ bad displacement input format2"""
    bg = BaseGeo(a3, r0)
    bg.move([[[1,2,3]]*3]*4)


# BAD BASEGEO .ROTATE INPUT------------------------------------------------
def badInput_rotate1():
    """ bad rot input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate(123)

def badInput_rotate2():
    """ bad anchor input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate(r0, anchor=123)

def badInput_rotate3():
    """ bad start input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate(r0, start=1.23)

def badInput_rotate4():
    """ bad increment input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate(r0, increment=1)

def badInput_rotate5():
    """ bad anchor input format"""
    bg = BaseGeo(a3, r0)
    bg.rotate(r0, anchor=(1,2,3,4))


# BAD BASEGEO .ROTATE_FROM_ANGAX INPUT-----------------------------------------
def badInput_rotate_from_angax1():
    """ bad ang input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax('123', (1,2,3))

def badInput_rotate_from_angax2():
    """ bad axis input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, 1)

def badInput_rotate_from_angax3():
    """ bad anchor input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, (1,2,3), anchor=1)

def badInput_rotate_from_angax4():
    """ bad start input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, (1,2,3), start=1.1)

def badInput_rotate_from_angax5():
    """ bad increment input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, (1,2,3), increment=None)

def badInput_rotate_from_angax6():
    """ bad degrees input type"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, (1,2,3), degrees=None)

def badInput_rotate_from_angax7():
    """ bad angle input format"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax([[1,2,3]]*2, (1,2,3))

def badInput_rotate_from_angax8():
    """ bad axis input format"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax([1,2,3], [1,2,3,4])

def badInput_rotate_from_angax9():
    """ bad axis input format"""
    bg = BaseGeo(a3, r0)
    bg.rotate_from_angax(123, [0,0,0])


# EXCITATIONS -----------------------------------------

def badMag_input1():
    """bad magnetization input type"""
    magpy.magnet.Cuboid(magnetization='woot', dimension=a3)

def badMag_input2():
    """bad magnetization input format"""
    magpy.magnet.Cuboid(magnetization=(1,2,3,4), dimension=a3)

def badCurrent_input1():
    """bad current input type"""
    magpy.current.Circular(current='1', diameter=1)

# DIMENSIONS --------------------------------------------

def bad_dim_input1():
    """cuboid dim type"""
    magpy.magnet.Cuboid(magnetization=a3, dimension=1)

def bad_dim_input3():
    """cuboid dim format"""
    magpy.magnet.Cuboid(magnetization=a3, dimension=(1,2))

def bad_dim_input4():
    """cylinder dim type"""
    magpy.magnet.Cylinder(magnetization=a3, dimension=1)
def bad_dim_input6():
    """cylinder bad dim shape"""
    magpy.magnet.Cylinder(magnetization=a3, dimension=(1,2,0,45))

def bad_dim_input7():
    """Sphere dim type"""
    magpy.magnet.Sphere(magnetization=a3, diameter=(1,1))

def bad_dim_input9():
    """Circular dim type"""
    magpy.current.Circular(current=1, diameter=(1,1))

def bad_dim_input10():
    """cylinder section bad dim shape"""
    magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,0,45))
def bad_dim_input11():
    """cylinder section bad dim d1>d2"""
    magpy.magnet.CylinderSegment(magnetization=a3, dimension=(3,2,2,0,45))
def bad_dim_input12():
    """cylinder section bad dim phi1>phi2"""
    magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,2,100,45))
def bad_dim_input13():
    """cylinder section bad dim phi2-phi1>360"""
    magpy.magnet.CylinderSegment(magnetization=a3, dimension=(1,2,2,0,1145))


# MISC SOURCE ------------------------------------------------------------

def bad_misc_input1():
    """Sensor pixel type"""
    magpy.Sensor(pixel=1)
def bad_misc_input2():
    """Sensor pixel format"""
    magpy.Sensor(pixel=[[1,2]]*3)

def bad_misc_input3():
    """Line vertex type"""
    magpy.current.Line(1, '1')
def bad_misc_input4():
    """Line vertex format 1"""
    magpy.current.Line(1, [(1,2,3)])
def bad_misc_input5():
    """Line vertex format 2"""
    magpy.current.Line(1, [[(1,2,3)]*2]*2)
def bad_misc_input6():
    """Line vertex format 3"""
    magpy.current.Line(1, [(1,2)]*2)


# OBSERVER ------------------------------------------------------------

src = magpy.current.Circular(current=1, diameter=1)
sens = magpy.Sensor()
def bad_observer_input1():
    """getBH observer format"""
    magpy.getB(src, a234)
def bad_observer_input2():
    """getBH observer format"""
    magpy.getB(src, [sens, [(1,2),(1,2)]])
def bad_observer_input3():
    """getBH observer format"""
    magpy.getB(src, [(1,2),(1,2)])
def bad_observer_input4():
    """bad observer type"""
    magpy.getB(src, 123)


class TestExceptions(unittest.TestCase):
    """ test class for exception testing """

    def test_init(self):
        """ missing inputs when calling display and getB"""
        self.assertRaises(MagpylibMissingInput, init_dim_display1)
        self.assertRaises(MagpylibMissingInput, init_dim_display2)
        self.assertRaises(MagpylibMissingInput, init_dim_display3)
        self.assertRaises(MagpylibMissingInput, init_dim_getBH1)
        self.assertRaises(MagpylibMissingInput, init_exite_display1)
        self.assertRaises(MagpylibMissingInput, init_exite_getBH1)
        self.assertRaises(MagpylibMissingInput, init_exite_getBH2)
        self.assertRaises(MagpylibMissingInput, init_exite_getBH3)
        self.assertRaises(MagpylibMissingInput, init_exite_getBH4)

    def test_BaseGeo(self):
        """ bad BG inputs"""
        self.assertRaises(MagpylibBadUserInput, badInput_BaseGeo1)
        self.assertRaises(MagpylibBadInputShape, badInput_BaseGeo2)
        self.assertRaises(MagpylibBadInputShape, badInput_BaseGeo3)
        self.assertRaises(MagpylibBadUserInput, badInput_BaseGeo4)

    def test_move(self):
        """ bad .move inputs"""
        self.assertRaises(MagpylibBadUserInput, badInput_move1)
        self.assertRaises(MagpylibBadUserInput, badInput_move2)
        self.assertRaises(MagpylibBadUserInput, badInput_move3)
        self.assertRaises(MagpylibBadInputShape, badInput_move4)
        self.assertRaises(MagpylibBadInputShape, badInput_move5)

    def test_rotate(self):
        """ bad .rotate inputs"""
        self.assertRaises(MagpylibBadUserInput, badInput_rotate1)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate2)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate3)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate4)
        self.assertRaises(MagpylibBadInputShape, badInput_rotate5)

    def test_rotate_from_angax(self):
        """ bad .rotate_from_angax inputs"""
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax1)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax2)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax3)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax4)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax5)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax6)
        self.assertRaises(MagpylibBadInputShape, badInput_rotate_from_angax7)
        self.assertRaises(MagpylibBadInputShape, badInput_rotate_from_angax8)
        self.assertRaises(MagpylibBadUserInput, badInput_rotate_from_angax9)

    def test_magnetization_input(self):
        """ bad magnetization inputs"""
        self.assertRaises(MagpylibBadUserInput, badMag_input1)
        self.assertRaises(MagpylibBadInputShape, badMag_input2)
        self.assertRaises(MagpylibBadUserInput, badCurrent_input1)

    def test_dim_inputs(self):
        """ bad dimension inputs"""
        self.assertRaises(MagpylibBadUserInput, bad_dim_input1)
        self.assertRaises(MagpylibBadInputShape, bad_dim_input3)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input4)
        self.assertRaises(MagpylibBadInputShape, bad_dim_input6)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input7)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input9)
        self.assertRaises(MagpylibBadInputShape, bad_dim_input10)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input11)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input12)
        self.assertRaises(MagpylibBadUserInput, bad_dim_input13)

    def test_misc_source_inputs(self):
        """ bad misc source inputs"""
        self.assertRaises(MagpylibBadUserInput, bad_misc_input1)
        self.assertRaises(MagpylibBadInputShape, bad_misc_input2)
        self.assertRaises(MagpylibBadUserInput, bad_misc_input3)
        self.assertRaises(MagpylibBadInputShape, bad_misc_input4)
        self.assertRaises(MagpylibBadInputShape, bad_misc_input5)
        self.assertRaises(MagpylibBadInputShape, bad_misc_input6)

    def test_observer_inputs(self):
        """ bad observer inputs"""
        self.assertRaises(MagpylibBadInputShape, bad_observer_input1)
        self.assertRaises(MagpylibBadInputShape, bad_observer_input2)
        self.assertRaises(MagpylibBadInputShape, bad_observer_input3)
        self.assertRaises(MagpylibBadUserInput, bad_observer_input4)
