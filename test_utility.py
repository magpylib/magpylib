import numpy as np
import magpylib as mag3
from magpylib._lib.math_utility.utility import rotobj_from_angax
from magpylib._lib.math_utility.utility import check_duplicates


def test_rotobj_from_angax():
    """ test special case axis=0
    """
    a = 1
    aa = (0,0,0)
    x = rotobj_from_angax(a,aa)
    assert np.all(x.as_quat()==(0,0,0,1)), 'axis = 0 fail'


def test_duplicates():
    """ test duplicate elimination and sorting
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(1,2))
    src_list = [pm1,pm2,pm1]
    src_list_new = check_duplicates(src_list)
    assert src_list_new == [pm1,pm2], 'duplicate elimination failed'
