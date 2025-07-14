import numpy as np

import magpylib as magpy


def test_Dipole_basicB():
    """Basic dipole class test"""
    src = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    sens = magpy.Sensor()

    B = src.getB(sens)
    Btest = np.array([3.81801774e-09, 7.63603548e-09, 1.14540532e-08])
    np.testing.assert_allclose(B, Btest)


def test_Dipole_basicH():
    """Basic dipole class test"""
    src = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    sens = magpy.Sensor()
    H = src.getH(sens)
    Htest = np.array([0.00303828, 0.00607656, 0.00911485])
    np.testing.assert_allclose(H, Htest, rtol=1e-05, atol=1e-08)


def test_Dipole_zero_position():
    """Basic dipole class test"""
    src = magpy.misc.Dipole(moment=(1, 2, 3))
    sens = magpy.Sensor()
    np.seterr(all="ignore")
    B = magpy.getB(src, sens)
    np.seterr(all="print")
    assert all(np.isnan(B))


def test_repr():
    """test __repr__"""
    dip = magpy.misc.Dipole(moment=(1, 2, 3))
    assert repr(dip)[:6] == "Dipole", "Dipole repr failed"
