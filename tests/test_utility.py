import numpy as np
import magpylib as mag3
from magpylib._lib.utility import (check_duplicates,
    only_allowed_src_types)


def test_duplicates():
    """ test duplicate elimination and sorting
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(1,2))
    src_list = [pm1,pm2,pm1]
    src_list_new = check_duplicates(src_list)
    assert src_list_new == [pm1,pm2], 'duplicate elimination failed'

def test_only_allowed_src_types():
    """ tests elimination of unwanted types
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(1,2))
    sens = mag3.Sensor()
    src_list = [pm1,pm2,sens]
    list_new = only_allowed_src_types(src_list)
    assert list_new == [pm1,pm2], 'Failed to eliminate sensor'


def test_format_getBH_class_inputs():
    """ special case testing of different input formats
    """
    possis = [3,3,3]
    sens = mag3.Sensor(pos=(3,3,3))
    pm1 = mag3.magnet.Box((11,22,33),(1,2,3))
    pm2 = mag3.magnet.Box((11,22,33),(1,2,3))
    col = pm1 + pm2

    B1 = pm1.getB(possis)
    B2 = pm1.getB(sens)
    assert np.allclose(B1,B2), 'pos_obs shold give same as sens'

    B3 = pm1.getB(sens,sens)
    B4 = pm1.getB([sens,sens])
    B44 = pm1.getB((sens,sens))
    assert np.allclose(B3, B4), 'sens,sens should give same as [sens,sens]'
    assert np.allclose(B3, B44), 'sens,sens should give same as (sens,sens)'

    B1 = sens.getH(pm1)*4
    B2 = sens.getH(pm1,pm2,col, sumup=True)
    B3 = sens.getH([col])*2
    B4 = sens.getH([col,pm1,pm2], sumup=True)

    assert np.allclose(B1,B2), 'src,src should give same as [src,src]'
    assert np.allclose(B1,B3), 'src should give same as [src]'
    assert np.allclose(B1,B4), 'src,src should give same as [src,src]'
