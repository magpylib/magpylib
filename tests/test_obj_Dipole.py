import numpy as np

import magpylib as magpy


def test_Dipole_basicB():
    """Basic dipole class test"""
    src = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    sens = magpy.Sensor()

    B = src.getB(sens)
    Btest = np.array([0.00303828, 0.00607656, 0.00911485])
    assert np.allclose(B, Btest)


def test_Dipole_basicH():
    """Basic dipole class test"""
    src = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    sens = magpy.Sensor()
    H = src.getH(sens)
    Htest = np.array([0.00241779, 0.00483558, 0.00725336])
    assert np.allclose(H, Htest)


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
