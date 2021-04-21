import numpy as np
from magpylib._lib.fields.field_BH_dipole import field_BH_dipole


def test_field_dipole1():
    """ Test standard dipole field output computed with mathematica
    """
    poso = np.array([(1,2,3),(-1,2,3)])
    mom = np.array([(2,3,4),(0,-3,-2)])
    B = field_BH_dipole(True, mom, poso)*np.pi
    Btest = np.array([
        (0.01090862,0.02658977,0.04227091),
        (0.0122722,-0.01022683,-0.02727156),
        ])

    assert np.allclose(B,Btest)


def test_field_dipole2():
    """ test nan return when pos_obs=0
    """
    poso = np.array([(0,0,0)])
    mom = np.array([(-1,2,3)])
    np.seterr(all='ignore')
    B = field_BH_dipole(True, mom, poso)*np.pi
    np.seterr(all='warn')

    assert all(np.isnan(B[0]))
