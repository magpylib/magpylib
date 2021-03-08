import numpy as np
from magpylib._lib.math_utility.utility import format_obj_input





def test_rotobj_from_angax():
    a = 1
    aa = (0,0,0)
    x = rotobj_from_angax(a,aa)
    assert np.all(x.as_quat()==(0,0,0,1)), 'axis = 0 fail'

