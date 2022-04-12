import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.exceptions import MagpylibInternalError
from magpylib._src.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.fields.field_wrap_BH_level2_dict import getBH_dict_level2
from magpylib._src.input_checks import check_format_input_observers
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import format_src_inputs
from magpylib._src.utility import test_path_format as tpf


def getBHv_unknown_source_type():
    """unknown source type"""
    getBH_dict_level2(
        source_type="badName",
        magnetization=(1, 0, 0),
        dimension=(0, 2, 1, 0, 360),
        position=(0, 0, -0.5),
        observers=(1.5, 0, -0.1),
        field="B",
    )


def getBH_level1_internal_error():
    """bad source_type input should not happen"""
    x = np.array([(1, 2, 3)])
    rot = R.from_quat((0, 0, 0, 1))
    getBH_level1(
        field="B",
        source_type="woot",
        magnetization=x,
        dimension=x,
        observers=x,
        position=x,
        orientation=rot,
    )


def getBH_level2_bad_input1():
    """test BadUserInput error at getBH_level2"""
    src = magpy.magnet.Cuboid((1, 1, 2), (1, 1, 1))
    sens = magpy.Sensor()
    getBH_level2([src, sens], (0, 0, 0), sumup=False, squeeze=True, field="B")


def getBH_level2_bad_input2():
    """different pixel shapes"""
    mag = (1, 2, 3)
    dim_cuboid = (1, 2, 3)
    pm1 = magpy.magnet.Cuboid(mag, dim_cuboid)
    sens1 = magpy.Sensor()
    sens2 = magpy.Sensor(pixel=[(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    magpy.getB(pm1, [sens1, sens2])


def getBH_level2_internal_error1():
    """somehow an unrecognized objects end up in get_src_dict"""
    # pylint: disable=protected-access
    sens = magpy.Sensor()
    x = np.zeros((10, 3))
    magpy._src.fields.field_wrap_BH_level2.get_src_dict([sens], 10, 10, x)


# getBHv missing inputs ------------------------------------------------------
def getBHv_missing_input1():
    """missing bh"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(source_type="Cuboid", observers=x, magnetization=x, dimension=x)


def getBHv_missing_input2():
    """missing source_type"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, observers=x, magnetization=x, dimension=x)


def getBHv_missing_input3():
    """missing observers"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Cuboid", magnetization=x, dimension=x)


def getBHv_missing_input4_cuboid():
    """missing Cuboid mag"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Cuboid", observers=x, dimension=x)


def getBHv_missing_input5_cuboid():
    """missing Cuboid dim"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Cuboid", observers=x, magnetization=x)


def getBHv_missing_input4_cyl():
    """missing Cylinder mag"""
    x = np.array([(1, 2, 3)])
    y = np.array([(1, 2)])
    getBH_dict_level2(bh=True, source_type="Cylinder", observers=x, dimension=y)


def getBHv_missing_input5_cyl():
    """missing Cylinder dim"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Cylinder", observers=x, magnetization=x)


def getBHv_missing_input4_sphere():
    """missing Sphere mag"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Sphere", observers=x, dimension=1)


def getBHv_missing_input5_sphere():
    """missing Sphere dim"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(bh=True, source_type="Sphere", observers=x, magnetization=x)


# bad inputs -------------------------------------------------------------------
def getBHv_bad_input1():
    """different input lengths"""
    x = np.array([(1, 2, 3)])
    x2 = np.array([(1, 2, 3)] * 2)
    getBH_dict_level2(
        bh=True, source_type="Cuboid", observers=x, magnetization=x2, dimension=x
    )


def getBHv_bad_input2():
    """bad source_type string"""
    x = np.array([(1, 2, 3)])
    getBH_dict_level2(
        bh=True, source_type="Cubooid", observers=x, magnetization=x, dimension=x
    )


def getBHv_bad_input3():
    """mixed input"""
    x = np.array([(1, 2, 3)])
    s = magpy.Sensor()
    getBH_dict_level2(
        bh=True, source_type="Cuboid", observers=s, magnetization=x, dimension=x
    )


def utility_format_obj_input():
    """bad input object"""
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    format_obj_input([pm1, pm2, 333])


def utility_format_src_inputs():
    """bad src input"""
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    format_src_inputs([pm1, pm2, 1])


def utility_format_obs_inputs():
    """bad src input"""
    sens1 = magpy.Sensor()
    sens2 = magpy.Sensor()
    possis = [1, 2, 3]
    check_format_input_observers([sens1, sens2, possis, "whatever"])


def utility_test_path_format():
    """bad path format input"""
    # pylint: disable=protected-access
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm1._position = [(1, 2, 3), (1, 2, 3)]
    tpf(pm1)


###############################################################################
# BAD INPUT SHAPE EXCEPTIONS
def bad_input_shape_basegeo_pos():
    """bad position input shape"""
    vec3 = (1, 2, 3)
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Cuboid(vec3, vec3, vec4)


def bad_input_shape_cuboid_dim():
    """bad cuboid dimension shape"""
    vec3 = (1, 2, 3)
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Cuboid(vec3, vec4)


def bad_input_shape_cuboid_mag():
    """bad cuboid magnetization shape"""
    vec3 = (1, 2, 3)
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Cuboid(vec4, vec3)


def bad_input_shape_cyl_dim():
    """bad cylinder dimension shape"""
    vec3 = (1, 2, 3)
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Cylinder(vec3, vec4)


def bad_input_shape_cyl_mag():
    """bad cylinder magnetization shape"""
    vec3 = (1, 2, 3)
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Cylinder(vec4, vec3)


def bad_input_shape_sphere_mag():
    """bad sphere magnetization shape"""
    vec4 = (1, 2, 3, 4)
    magpy.magnet.Sphere(vec4, 1)


def bad_input_shape_sensor_pix_pos():
    """bad sensor pix_pos input shape"""
    vec4 = (1, 2, 3, 4)
    vec3 = (1, 2, 3)
    magpy.Sensor(vec3, vec4)


def bad_input_shape_dipole_mom():
    """bad sphere magnetization shape"""
    vec4 = (1, 2, 3, 4)
    magpy.misc.Dipole(moment=vec4)


#####################################################################
class TestExceptions(unittest.TestCase):
    """test class for exception testing"""

    def test_except_utility(self):
        """utility"""
        self.assertRaises(MagpylibBadUserInput, utility_test_path_format)
        self.assertRaises(MagpylibBadUserInput, utility_format_obj_input)
        self.assertRaises(MagpylibBadUserInput, utility_format_src_inputs)
        self.assertRaises(MagpylibBadUserInput, utility_format_obs_inputs)

    def test_except_getBHv(self):
        """getBHv"""
        self.assertRaises(KeyError, getBHv_missing_input1)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input2)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input3)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_cuboid)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_cyl)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_sphere)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_cuboid)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_cyl)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_sphere)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input1)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input2)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input3)
        self.assertRaises(MagpylibBadUserInput, getBHv_unknown_source_type)

    def test_except_getBH_lev1(self):
        """getBH_level1 exception testing"""
        self.assertRaises(MagpylibInternalError, getBH_level1_internal_error)

    def test_except_getBH_lev2(self):
        """getBH_level2 exception testing"""
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input1)
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input2)
        self.assertRaises(MagpylibInternalError, getBH_level2_internal_error1)

    def test_except_bad_input_shape_basegeo(self):
        """BaseGeo bad input shapes"""
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_basegeo_pos)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_cuboid_dim)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_cuboid_mag)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_cyl_dim)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_cyl_mag)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_sphere_mag)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_sensor_pix_pos)
        self.assertRaises(MagpylibBadUserInput, bad_input_shape_dipole_mom)
