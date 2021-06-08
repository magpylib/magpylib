import unittest
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import magpylib as mag3
from magpylib._lib.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._lib.fields.field_wrap_BH_v import getBHv_level2
from magpylib._lib.exceptions import (MagpylibInternalError, MagpylibBadUserInput,
    MagpylibBadInputShape)
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.utility import format_obj_input, format_src_inputs, format_obs_inputs
from magpylib._lib.utility import test_path_format as tpf


def getBH_level1_internal_error():
    """ bad src_type input should not happen
    """
    x = np.array([(1,2,3)])
    rot = R.from_quat((0,0,0,1))
    getBH_level1(bh=True,src_type='woot', magnetization=x, dimension=x, pos_obs=x, position=x,rot=rot)


def getBH_level2_bad_input1():
    """ test BadUserInput error at getBH_level2
    """
    src = mag3.magnet.Box((1,1,2),(1,1,1))
    sens = mag3.Sensor()
    getBH_level2(True, [src,sens],(0,0,0),False,True)


def getBH_level2_bad_input2():
    """ different pixel shapes
    """
    mag = (1,2,3)
    dim_box = (1,2,3)
    pm1 = mag3.magnet.Box(mag,dim_box)
    sens1 = mag3.Sensor()
    sens2 = mag3.Sensor(pos_pix=[(0,0,0),(0,0,1),(0,0,2)])
    mag3.getB(pm1,[sens1,sens2])


def getBH_level2_internal_error1():
    """ somhow an unrecognized objects end up in get_src_dict
    """
    # pylint: disable=protected-access
    sens = mag3.Sensor()
    x = np.zeros((10,3))
    mag3._lib.fields.field_wrap_BH_level2.get_src_dict([sens],10,10,x)


def getBHv_missing_input1():
    """ missing bh
    """
    x=np.array([(1,2,3)])
    getBHv_level2(src_type='Box', pos_obs=x, magnetization=x, dimension=x)


def getBHv_missing_input2():
    """ missing src_type
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, pos_obs=x, magnetization=x, dimension=x)


def getBHv_missing_input3():
    """ missing pos_obs
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Box', magnetization=x, dimension=x)


def getBHv_missing_input4_box():
    """ missing Box mag
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Box', pos_obs=x, dimension=x)


def getBHv_missing_input5_box():
    """ missing Box dim
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Box', pos_obs=x, magnetization=x)


def getBHv_missing_input4_cyl():
    """ missing Cylinder mag
    """
    x=np.array([(1,2,3)])
    y = np.array([(1,2)])
    getBHv_level2(bh=True, src_type='Cylinder', pos_obs=x, dimension=y)


def getBHv_missing_input5_cyl():
    """ missing Cylinder dim
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Cylinder', pos_obs=x, magnetization=x)


def getBHv_missing_input4_sphere():
    """ missing Sphere mag
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Sphere', pos_obs=x, dimension=1)


def getBHv_missing_input5_sphere():
    """ missing Sphere dim
    """
    x=np.array([(1,2,3)])
    getBHv_level2(bh=True, src_type='Sphere', pos_obs=x, magnetization=x)


def getBHv_bad_input():
    """ different input lengths
    """
    x=np.array([(1,2,3)])
    x2=np.array([(1,2,3)]*2)
    getBHv_level2(bh=True, src_type='Box', pos_obs=x, magnetization=x2, dimension=x)


# def base_geo_bad_pos():
#     """ bad position input shape
#     """
#     bg = BaseGeo((0,0,0), R.from_quat((0,0,0,1)))
#     poss = [[(1,2,3),(1,2,3)],[(1,2,3),(1,2,3)]]
#     bg.pos = poss


# def base_geo_bad_rot_axis():
#     """ bad rotation axis input
#     """
#     bg = BaseGeo((0,0,0), R.from_quat((0,0,0,1)))
#     bg.rotate_from_angax(15,'u')


def utility_format_obj_input():
    """ bad input object
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    format_obj_input([pm1,pm2,333])


def utility_format_src_inputs():
    """ bar src input
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    format_src_inputs([pm1,pm2,1])


def utility_format_obs_inputs():
    """ bar src input
    """
    sens1 = mag3.Sensor()
    sens2 = mag3.Sensor()
    possis = [1,2,3]
    format_obs_inputs([sens1,sens2,possis,'whatever'])


def utility_test_path_format():
    """ bad path format input
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm1.position = [(1,2,3),(1,2,3)]
    tpf(pm1)


def box_no_mag():
    """ Box with no mag input
    """
    mag3.magnet.Box(dimension=(1,2,3))


def box_no_dim():
    """ Box with no dim input
    """
    mag3.magnet.Box(magnetization=(1,2,3))


def cyl_no_mag():
    """ Cylinder with no mag input
    """
    mag3.magnet.Cylinder(dimension=(1,2))


def cyl_no_dim():
    """ Cylinder with no dim input
    """
    mag3.magnet.Cylinder(magnetization=(1,2,3))


def sphere_no_mag():
    """ Cylinder with no mag input
    """
    mag3.magnet.Sphere(dimension=1)


def sphere_no_dim():
    """ Cylinder with no dim input
    """
    mag3.magnet.Sphere(magnetization=(1,2,3))


def dipole_no_mom():
    """ Cylinder with no mag input
    """
    mag3.misc.Dipole()


def circular_no_current():
    """ Circular with no current input
    """
    mag3.current.Circular(dimension=1)


def circular_no_dim():
    """ Circular with no dim input
    """
    mag3.current.Circular(current=1)

########################################################################
# BAD INPUT SHAPE EXCEPTIONS
def bad_input_shape_basegeo_pos():
    """ bad position input shape
    """
    vec3 = (1,2,3)
    vec4 = (1,2,3,4)
    mag3.magnet.Box(vec3, vec3, vec4)


# def bad_input_shape_basegeo_move():
#     """ bad displacement input shape
#     """
#     vec3 = (1,2,3)
#     vec4 = (1,2,3,4)
#     src = mag3.magnet.Box(vec3, vec3)
#     src.move(vec4)


# def bad_input_shape_basegeo_rotate_from_aa_axis():
#     """ bad rotation axis input shape
#     """
#     vec3 = (1,2,3)
#     vec4 = (1,2,3,4)
#     src = mag3.magnet.Box(vec3, vec3)
#     src.rotate_from_angax(123,vec4)

# def bad_input_shape_basegeo_rotate_from_aa_anchor():
#     """ bad rotation anchor input shape
#     """
#     vec3 = (1,2,3)
#     vec4 = (1,2,3,4)
#     src = mag3.magnet.Box(vec3, vec3)
#     src.rotate_from_angax(123,vec3,vec4)


def bad_input_shape_box_dim():
    """ bad box dimension shape
    """
    vec3 = (1,2,3)
    vec4 = (1,2,3,4)
    mag3.magnet.Box(vec3, vec4)


def bad_input_shape_box_mag():
    """ bad box magnetization shape
    """
    vec3 = (1,2,3)
    vec4 = (1,2,3,4)
    mag3.magnet.Box(vec4, vec3)


def bad_input_shape_cyl_dim():
    """ bad cylinder dimension shape
    """
    vec3 = (1,2,3)
    vec4 = (1,2,3,4)
    mag3.magnet.Cylinder(vec3, vec4)


def bad_input_shape_cyl_mag():
    """ bad box magnetization shape
    """
    vec3 = (1,2,3)
    vec4 = (1,2,3,4)
    mag3.magnet.Cylinder(vec4, vec3)


def bad_input_shape_sphere_mag():
    """ bad sphere magnetization shape
    """
    vec4 = (1,2,3,4)
    mag3.magnet.Sphere(vec4, 1)


def bad_input_shape_sensor_pix_pos():
    """ bad sensor pix_pos input shape
    """
    vec4 = (1,2,3,4)
    vec3 = (1,2,3)
    mag3.Sensor(vec3, vec4)


def bad_input_shape_dipole_mom():
    """ bad sphere magnetization shape
    """
    vec4 = (1,2,3,4)
    mag3.misc.Dipole(moment=vec4)


########################################################################
class TestExceptions(unittest.TestCase):
    """ test class for exception testing
    """
    def test_except_class_Box(self):
        """ class_Box
        """
        self.assertRaises(MagpylibBadUserInput, box_no_mag)
        self.assertRaises(MagpylibBadUserInput, box_no_dim)

    def test_except_class_Cylinder(self):
        """ class_Cylinder
        """
        self.assertRaises(MagpylibBadUserInput, cyl_no_mag)
        self.assertRaises(MagpylibBadUserInput, cyl_no_dim)

    def test_except_class_Sphere(self):
        """ class_Sphere
        """
        self.assertRaises(MagpylibBadUserInput, sphere_no_mag)
        self.assertRaises(MagpylibBadUserInput, sphere_no_dim)

    def test_except_class_Dipole(self):
        """ class_Dipole
        """
        self.assertRaises(MagpylibBadUserInput, dipole_no_mom)

    def test_except_class_Circular(self):
        """ class_Circular
        """
        self.assertRaises(MagpylibBadUserInput, circular_no_current)
        self.assertRaises(MagpylibBadUserInput, circular_no_dim)

    def test_except_utility(self):
        """ utility
        """
        self.assertRaises(MagpylibBadUserInput, utility_test_path_format)
        self.assertRaises(MagpylibBadUserInput, utility_format_obj_input)
        self.assertRaises(MagpylibBadUserInput, utility_format_src_inputs)
        self.assertRaises(MagpylibBadUserInput, utility_format_obs_inputs)

    #def test_except_class_BaseGeo(self):
    #    """ BaseGeo
    #    """
    #    #self.assertRaises(MagpylibBadUserInput, base_geo_bad_pos)
    #    #self.assertRaises(MagpylibBadUserInput, base_geo_bad_rot_axis)

    def test_except_getBHv(self):
        """ getBHv
        """
        self.assertRaises(KeyError, getBHv_missing_input1)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input2)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input3)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_box)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_cyl)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input4_sphere)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_box)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_cyl)
        self.assertRaises(MagpylibBadUserInput, getBHv_missing_input5_sphere)
        self.assertRaises(MagpylibBadUserInput, getBHv_bad_input)

    def test_except_getBH_lev1(self):
        """ getBH_level1 exception testing
        """
        self.assertRaises(MagpylibInternalError, getBH_level1_internal_error)

    def test_except_getBH_lev2(self):
        """ getBH_level2 exception testing
        """
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input1)
        self.assertRaises(MagpylibBadUserInput, getBH_level2_bad_input2)
        self.assertRaises(MagpylibInternalError, getBH_level2_internal_error1)

    def test_except_bad_input_shape_basegeo(self):
        """ BaseGeo bad input shapes
        """
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_basegeo_pos)
        #self.assertRaises(MagpylibBadInputShape, bad_input_shape_basegeo_move)
        #self.assertRaises(MagpylibBadInputShape, bad_input_shape_basegeo_rotate_from_aa_axis)
        #self.assertRaises(MagpylibBadInputShape, bad_input_shape_basegeo_rotate_from_aa_anchor)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_box_dim)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_box_mag)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_cyl_dim)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_cyl_mag)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_sphere_mag)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_sensor_pix_pos)
        self.assertRaises(MagpylibBadInputShape, bad_input_shape_dipole_mom)
