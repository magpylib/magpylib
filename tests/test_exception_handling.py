import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import magpylib as mag3
from magpylib._lib.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._lib.fields.field_wrap_getBHv import getBHv_level2
from magpylib._lib.exceptions import MagpylibInternalError, MagpylibBadUserInput
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.math_utility.utility import format_obj_input, get_good_path_length
from magpylib._lib.math_utility import test_path_format as tpf

def test_level1_internal_error():
    """ test internal error at getBH_level1
    """
    x = np.array([(1,2,3)])
    rot = R.from_quat((0,0,0,1))

    flag = True
    try:
        getBH_level1(bh=True,src_type='Box', mag=x, dim=x, pos_obs=x, pos=x,rot=rot)
    except MagpylibInternalError:
        flag = False
    assert flag, 'exception badly raised'

    try:
        getBH_level1(bh=True,src_type='woot', mag=x, dim=x, pos_obs=x, pos=x,rot=rot)
    except MagpylibInternalError:
        flag = False
    assert not flag, 'exception not raised'


def test_level2_user_input_error():
    """ test BadUserInput error at getBH_level2
    """
    src = mag3.magnet.Box((1,1,2),(1,1,1))
    sens = mag3.Sensor()

    flag = True
    try:
        getBH_level2(True, [src,src],(0,0,0),False)
    except MagpylibBadUserInput:
        flag = False
    assert flag, 'exception badly raised'

    try:
        getBH_level2(True, [src,sens],(0,0,0),False)
    except MagpylibBadUserInput:
        flag = False
    assert not flag, 'exception not raised'


def test_level2v_errors():
    """ test internal error at getBHv_level2
    """
    x=np.array([(1,2,3)])
    x2=np.array([(1,2,3),(1,2,3)])

    # test bh, src_type, pos_obs
    flag = False
    try:
        getBHv_level2(bh=True)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'exception not raised1'

    # test source specific
    flag = False
    try:
        getBHv_level2(bh=True, src_type='Box', pos_obs=x)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'exception not raised2'

    flag = False
    try:
        getBHv_level2(bh=True, src_type='Cylinder', pos_obs=x)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'exception not raised3'

    # test vector length
    flag = False
    try:
        getBHv_level2(bh=True, src_type='Box', pos_obs=x, mag=x2, dim=x)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'exception not raised4'


def test_baseGeo_errors():
    """ test base Geo BadUserInputErrors
    """
    # test bad input shape
    bg = BaseGeo((0,0,0), R.from_quat((0,0,0,1)))
    poss = [[(1,2,3),(1,2,3)],[(1,2,3),(1,2,3)]]
    flag = False
    try:
        bg.pos = poss
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'bad pos input shape not caught'

    # test bad rotation axis string input
    bg = BaseGeo((0,0,0), R.from_quat((0,0,0,1)))
    flag = False
    try:
        bg.rotate_from_angax(15,'u')
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'bad rotation axis str input not caught'


def test_format_obj_input_error():
    """ special case testing
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))

    flag = False
    try:
        format_obj_input([pm1,pm2,333])
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'bad input not raised'


def test_test_path_format():
    """ test with single source input
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm1.pos = [(1,2,3),(1,2,3)]
    flag = False
    try:
        tpf(pm1)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'test_path_format_fail'


def test_get_good_path_length():
    """ test error raise
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm1.move_by((1,2,3),steps=11)
    pm2.move_by((1,2,3),steps=12)
    src_list = [pm1,pm2]

    flag = False
    try:
        get_good_path_length(src_list)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'get_good_path_length fail'
