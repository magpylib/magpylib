import os
import pickle

import numpy as np

import magpylib as magpy

# GENERATE TESTDATA ---------------------------------------
# import pickle
# import magpylib as magpy

# # linear motion from (0,0,0) to (3,-3,3) in 100 steps
# pm = magpy.source.magnet.Cuboid(polarization=(111,222,333), dimension=(1,2,3))
# B1 = np.array([pm.getB((i,-i,i)) for i in np.linspace(0,3,100)])

# # rotation (pos_obs around magnet) from 0 to 444 deg, starting pos_obs at (0,3,0) about 'z'
# pm = magpy.source.magnet.Cuboid(polarization=(111,222,333), dimension=(1,2,3))
# possis = [(3*np.sin(t/180*np.pi),3*np.cos(t/180*np.pi),0) for t in np.linspace(0,444,100)]
# B2 = np.array([pm.getB(p) for p in possis])

# # spiral (magnet around pos_obs=0) from 0 to 297 deg, about 'z' in 100 steps
# pm = magpy.source.magnet.Cuboid(polarization=(111,222,333), dimension=(1,2,3), pos=(3,0,0))
# B = []
# for i in range(100):
#     B += [pm.getB((0,0,0))]
#     pm.rotate(3,(0,0,1),anchor=(0,0,0))
#     pm.move((0,0,.1))
# B3 = np.array(B)

# B = np.array([B1,B2,B3])
# pickle.dump(B, open('testdata_vs_mag2.p', 'wb'))
# -------------------------------------------------------------


def test_vs_mag2_linear():
    """test against magpylib v2"""
    with open(os.path.abspath("tests/testdata/testdata_vs_mag2.p"), "rb") as f:
        data = pickle.load(f)[0]
    poso = [(t, -t, t) for t in np.linspace(0, 3, 100)]
    pm = magpy.magnet.Cuboid(polarization=(111, 222, 333), dimension=(1, 2, 3))

    B = magpy.getB(pm, poso)
    np.testing.assert_allclose(B, data, err_msg="vs mag2 - linear")


def test_vs_mag2_rotation():
    """test against magpylib v2"""
    with open(os.path.abspath("tests/testdata/testdata_vs_mag2.p"), "rb") as f:
        data = pickle.load(f)[1]
    pm = magpy.magnet.Cuboid(polarization=(111, 222, 333), dimension=(1, 2, 3))
    possis = [
        (3 * np.sin(t / 180 * np.pi), 3 * np.cos(t / 180 * np.pi), 0)
        for t in np.linspace(0, 444, 100)
    ]
    B = pm.getB(possis)
    np.testing.assert_allclose(B, data, err_msg="vs mag2 - rot")


def test_vs_mag2_spiral():
    """test against magpylib v2"""
    with open(os.path.abspath("tests/testdata/testdata_vs_mag2.p"), "rb") as f:
        data = pickle.load(f)[2]
    pm = magpy.magnet.Cuboid(
        polarization=(111, 222, 333), dimension=(1, 2, 3), position=(3, 0, 0)
    )

    angs = np.linspace(0, 297, 100)
    pm.rotate_from_angax(angs, "z", anchor=0, start=0)
    possis = np.linspace((0, 0, 0.1), (0, 0, 9.9), 99)
    pm.move(possis, start=1)
    B = pm.getB((0, 0, 0))
    np.testing.assert_allclose(B, data, err_msg="vs mag2 - rot")


def test_vs_mag2_line():
    """test line current vs mag2 results"""
    Btest = np.array([1.47881931, -1.99789688, 0.2093811]) * 1e-6

    src = magpy.current.Polyline(
        current=10,
        vertices=[(0, -5, 0), (0, 5, 0), (3, 3, 3), (-1, -2, -3), (1, 1, 1), (2, 3, 4)],
    )
    B = src.getB([1, 2, 3])

    np.testing.assert_allclose(Btest, B)
