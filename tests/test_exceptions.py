import unittest

import numpy as np

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_wrap_BH import getBH_level2
from magpylib._src.input_checks import check_format_input_observers
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import format_src_inputs
from magpylib._src.utility import test_path_format as tpf

GETBH_KWARGS = {"sumup": False, "squeeze": True, "pixel_agg": None, "output": "ndarray"}


def getBHv_unknown_source_type():
    """unknown source type"""
    getBH_level2(
        sources="badName",
        observers=(0, 0, 0),
        magnetization=(1, 0, 0),
        dimension=(0, 2, 1, 0, 360),
        position=(0, 0, -0.5),
        field="B",
        **GETBH_KWARGS,
    )


def getBH_level2_bad_input1():
    """test BadUserInput error at getBH_level2"""
    src = magpy.magnet.Cuboid((1, 1, 2), (1, 1, 1))
    sens = magpy.Sensor()
    getBH_level2(
        [src, sens],
        (0, 0, 0),
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        field="B",
        output="ndarray",
    )


def getBH_level2_bad_input2():
    """different pixel shapes"""
    mag = (1, 2, 3)
    dim_cuboid = (1, 2, 3)
    pm1 = magpy.magnet.Cuboid(mag, dim_cuboid)
    sens1 = magpy.Sensor()
    sens2 = magpy.Sensor(pixel=[(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    magpy.getB(pm1, [sens1, sens2])


# getBHv missing inputs ------------------------------------------------------
def getBHv_missing_input1():
    """missing field"""
    x = np.array([(1, 2, 3)])
    # pylint: disable=missing-kwoa
    getBH_level2(
        sources="Cuboid", observers=x, magnetization=x, dimension=x, **GETBH_KWARGS
    )


def getBHv_missing_input2():
    """missing source_type"""
    x = np.array([(1, 2, 3)])
    getBH_level2(observers=x, field="B", magnetization=x, dimension=x, **GETBH_KWARGS)


def getBHv_missing_input3():
    """missing observer"""
    x = np.array([(1, 2, 3)])
    getBH_level2(
        sources="Cuboid", field="B", magnetization=x, dimension=x, **GETBH_KWARGS
    )


def getBHv_missing_input4_cuboid():
    """missing Cuboid mag"""
    x = np.array([(1, 2, 3)])
    getBH_level2(sources="Cuboid", observers=x, field="B", dimension=x, **GETBH_KWARGS)


def getBHv_missing_input5_cuboid():
    """missing Cuboid dim"""
    x = np.array([(1, 2, 3)])
    getBH_level2(
        sources="Cuboid", observers=x, field="B", magnetization=x, **GETBH_KWARGS
    )


def getBHv_missing_input4_cyl():
    """missing Cylinder mag"""
    x = np.array([(1, 2, 3)])
    y = np.array([(1, 2)])
    getBH_level2(
        sources="Cylinder", observers=x, field="B", dimension=y, **GETBH_KWARGS
    )


def getBHv_missing_input5_cyl():
    """missing Cylinder dim"""
    x = np.array([(1, 2, 3)])
    getBH_level2(
        sources="Cylinder", observers=x, field="B", magnetization=x, **GETBH_KWARGS
    )


def getBHv_missing_input4_sphere():
    """missing Sphere mag"""
    x = np.array([(1, 2, 3)])
    getBH_level2(sources="Sphere", observers=x, field="B", dimension=1, **GETBH_KWARGS)


def getBHv_missing_input5_sphere():
    """missing Sphere dim"""
    x = np.array([(1, 2, 3)])
    getBH_level2(
        sources="Sphere", observers=x, field="B", magnetization=x, **GETBH_KWARGS
    )


# bad inputs -------------------------------------------------------------------
def getBHv_bad_input1():
    """different input lengths"""
    x = np.array([(1, 2, 3)] * 3)
    x2 = np.array([(1, 2, 3)] * 2)
    getBH_level2(
        sources="Cuboid",
        observers=x,
        field="B",
        magnetization=x2,
        dimension=x,
        **GETBH_KWARGS,
    )


def getBHv_bad_input2():
    """bad source_type string"""
    x = np.array([(1, 2, 3)])
    getBH_level2(
        sources="Cubooid",
        observers=x,
        field="B",
        magnetization=x,
        dimension=x,
        **GETBH_KWARGS,
    )


def getBHv_bad_input3():
    """mixed input"""
    x = np.array([(1, 2, 3)])
    s = magpy.Sensor()
    getBH_level2(
        sources="Cuboid",
        observers=s,
        field="B",
        magnetization=x,
        dimension=x,
        **GETBH_KWARGS,
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
        self.assertRaises(TypeError, getBHv_missing_input1)
        self.assertRaises(TypeError, getBHv_missing_input2)
        self.assertRaises(TypeError, getBHv_missing_input3)
        self.assertRaises(TypeError, getBHv_missing_input4_cuboid)
        self.assertRaises(TypeError, getBHv_missing_input4_cyl)
        self.assertRaises(TypeError, getBHv_missing_input4_sphere)
        self.assertRaises(TypeError, getBHv_missing_input5_cuboid)
        self.assertRaises(TypeError, getBHv_missing_input5_cyl)
        self.assertRaises(TypeError, getBHv_missing_input5_sphere)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input1)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input2)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input3)
        self.assertRaises(MagpylibBadUserInput, getBHv_unknown_source_type)

    def test_except_getBH_lev2(self):
        """getBH_level2 exception testing"""
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input1)
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input2)

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
