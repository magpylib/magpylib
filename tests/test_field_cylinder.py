"""
Testing all cases against a large set of pre-computed values
"""

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.fields.field_BH_cylinder import BHJM_magnet_cylinder
from magpylib._src.fields.field_BH_cylinder_segment import BHJM_cylinder_segment
from magpylib._src.fields.field_BH_cylinder_segment import (
    magnet_cylinder_segment_Hfield,
)

# pylint: disable="pointless-string-statement"
# creating test data
""" import os
import numpy as np
from magpylib._src.fields.field_BH_cylinder_tile import magnet_cylinder_segment_Hfield

N = 1111
null = np.zeros(N)
R = np.random.rand(N) * 10
R1, R2 = np.random.rand(2, N) * 5
R2 = R1 + R2
PHI, PHI1, PHI2 = (np.random.rand(3, N) - 0.5) * 10
PHI2 = PHI1 + PHI2
Z, Z1, Z2 = (np.random.rand(3, N) - 0.5) * 10
Z2 = Z1 + Z2

DIM_CYLSEG = np.array([R1, R2, PHI1, PHI2, Z1, Z2])
POS_OBS = np.array([R, PHI, Z])
MAG = np.random.rand(N, 3)

DATA = {}

# cases [112, 212, 132, 232]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z1 = z
phi1 = phi
r = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(
    magnetizations=MAG,
    dimensions=dim,
    observers=obs_pos,
)
DATA["cases [112, 212, 132, 232]"] = {
    "inputs": {"magnetizations": MAG, "dimensions": dim, "observers": obs_pos},
    "H_expected": H1,
}

# cases [122, 222, 132, 232]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z1 = z
phi1 = phi + np.pi
r = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [122, 222, 132, 232]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [113, 213, 133, 233, 115, 215, 135, 235]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z1 = z
phi1 = phi
r1 = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [113, 213, 133, 233, 115, 215, 135, 235]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}


# cases [123, 223, 133, 233, 125, 225, 135, 235]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z1 = z
phi1 = phi + np.pi
r1 = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [123, 223, 133, 233, 125, 225, 135, 235]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [125, 225, 135, 235, 124, 224, 134, 234]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z1 = z
phi1 = phi + np.pi
r = r2
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [125, 225, 135, 235, 124, 224, 134, 234]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [211, 221, 212, 222]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
phi1 = phi
phi2 = phi + np.pi
r = null
r1 = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [211, 221, 212, 222]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [214, 224, 215, 225]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
phi1 = phi
phi2 = phi + np.pi
r = r1
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [214, 224, 215, 225]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [111, 211, 121, 221, 112, 212, 122, 222]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z = z1
phi1 = phi
phi2 = phi + np.pi
r = null
r1 = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [111, 211, 121, 221, 112, 212, 122, 222]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [111, 211, 131, 231, 112, 212, 132, 232]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z = z1
phi1 = phi
r = null
r1 = null
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [111, 211, 131, 231, 112, 212, 132, 232]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

# cases [115, 215, 135, 235, 114, 214, 134, 234]
r1, r2, phi1, phi2, z1, z2 = DIM_CYLSEG
r, phi, z = POS_OBS
z = z1
phi1 = phi
r = r2
obs_pos = np.array([r, phi, z]).T
dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
H1 = magnet_cylinder_segment_Hfield(mag=MAG, dim=dim, obs_pos=obs_pos)
DATA["cases [115, 215, 135, 235, 114, 214, 134, 234]"] = {
    "inputs": {"mag": MAG, "dim": dim, "obs_pos": obs_pos},
    "H_expected": H1,
}

folder = r"magpylib"

np.save(os.path.join(folder, "tests/testdata/testdata_cy_cases"), DATA) """


# data is actually pickled, and a dictionary is stored inside of a numpy array
DATA = np.load("tests/testdata/testdata_cy_cases.npy", allow_pickle=True).item()


@pytest.mark.parametrize(
    "inputs, H_expected",
    [[v["inputs"], v["H_expected"]] for v in DATA.values()],
    ids=list(DATA.keys()),
)
def test_cylinder_tile_slanovc(inputs, H_expected):
    "testing precomputed cylinder test cases"
    inputs_mod = {
        "magnetizations": inputs["mag"],
        "observers": inputs["obs_pos"],
        "dimensions": inputs["dim"],
    }
    H = (
        magnet_cylinder_segment_Hfield(**inputs_mod) / 4 / np.pi * 1e7
    )  # factors come from B <->H change
    np.testing.assert_allclose(H, H_expected)


def test_cylinder_field1():
    """test the new cylinder field against old, full-cylinder
    implementations
    """
    N = 100
    magg, dim, poso, B0 = np.load("tests/testdata/testdata_full_cyl.npy")

    nulll = np.zeros(N)
    eins = np.ones(N)
    d, h, _ = dim.T  # pylint: disable=no-member
    dim5 = np.array([nulll, d / 2, h, nulll, eins * 360]).T
    B1 = BHJM_cylinder_segment(
        field="B", observers=poso, polarization=magg, dimension=dim5
    )

    np.testing.assert_allclose(B1, B0)


def test_cylinder_slanovc_field2():
    """testing B for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment(
        polarization=(22, 33, 44), dimension=(0.5, 1, 2, 0, 90)
    )

    r_in = (0.5, 0.6, 0.3)
    r_out = (1, 2, 3)
    r_corn = (1, 0, 0)

    b_in = (5.52525937, 13.04561569, 40.11111556)
    b_out = (0.0177018, 0.1277188, 0.27323195)
    b_corn = (0, 0, 0)

    # only inside
    btest = np.array([b_in] * 3)
    B = src.getB([r_in] * 3)
    np.testing.assert_allclose(B, btest)

    # only edge
    btest = np.array([b_corn] * 3)
    B = src.getB([r_corn] * 3)
    np.testing.assert_allclose(B, btest)

    # only outside
    btest = np.array([b_out] * 3)
    B = src.getB([r_out] * 3)
    np.testing.assert_allclose(B, btest, rtol=1e-05, atol=1e-08)

    # edge + out
    btest = np.array([b_corn, b_corn, b_out])
    B = src.getB([r_corn, r_corn, r_out])
    np.testing.assert_allclose(B, btest, rtol=1e-05, atol=1e-08)

    # surf + in
    btest = np.array([b_corn, b_corn, b_in])
    B = src.getB(r_corn, r_corn, r_in)
    np.testing.assert_allclose(B, btest)

    # in + out
    btest = np.array([b_out, b_in])
    B = src.getB(r_out, r_in)
    np.testing.assert_allclose(B, btest, rtol=1e-05, atol=1e-08)

    # in + out + surf
    btest = np.array([b_corn, b_corn, b_in, b_out, b_corn, b_corn])
    B = src.getB([r_corn, r_corn, r_in, r_out, r_corn, r_corn])
    np.testing.assert_allclose(B, btest, rtol=1e-05, atol=1e-08)


def test_cylinder_slanovc_field3():
    """testing H for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment(
        polarization=(22, 33, 44), dimension=(0.5, 1, 2, 0, 90)
    )

    hinn = np.array((-13.11018204, -15.87919449, -3.09467591)) * 1e6
    hout = np.array((0.01408664, 0.1016354, 0.21743108)) * 1e6
    nulll = (0, 0, 0)

    # only inside
    htest = np.array([hinn] * 3)
    H = src.getH([[0.5, 0.6, 0.3]] * 3)
    np.testing.assert_allclose(H, htest)

    # only surf
    htest = np.array([nulll] * 3)
    H = src.getH([[1, 0, 0]] * 3)
    np.testing.assert_allclose(H, htest)

    # only outside
    htest = np.array([hout] * 3)
    H = src.getH([[1, 2, 3]] * 3)
    np.testing.assert_allclose(H, htest)

    # surf + out
    htest = np.array([nulll, nulll, hout])
    H = src.getH([0.6, 0, 1], [1, 0, 0.5], [1, 2, 3])
    np.testing.assert_allclose(H, htest)

    # surf + in
    htest = np.array([nulll, nulll, hinn])
    H = src.getH([0, 0.5, 1], [1, 0, 0.5], [0.5, 0.6, 0.3])
    np.testing.assert_allclose(H, htest)

    # in + out
    htest = np.array([hout, hinn])
    H = src.getH([1, 2, 3], [0.5, 0.6, 0.3])
    np.testing.assert_allclose(H, htest)

    # in + out + surf
    htest = np.array([nulll, nulll, hinn, hout, nulll, nulll])
    H = src.getH(
        [0.5, 0.5, 1],
        [0, 1, 0.5],
        [0.5, 0.6, 0.3],
        [1, 2, 3],
        [0.5, 0.6, -1],
        [0, 1, -0.3],
    )
    np.testing.assert_allclose(H, htest)


def test_cylinder_rauber_field4():
    """
    test continuity across indefinite form in cylinder_rauber field when observer at r=r0
    """
    src = magpy.magnet.Cylinder(polarization=(22, 33, 0), dimension=(2, 2))
    es = list(10 ** -np.linspace(11, 15, 50))
    xs = np.r_[1 - np.array(es), 1, 1 + np.array(es)[::-1]]
    possis = [(x, 0, 1.5) for x in xs]
    B = src.getB(possis)
    B = B / B[25]
    assert np.all(abs(1 - B) < 1e-8)


def test_cylinder_tile_negative_phi():
    """same result for phi>0 and phi<0 inputs"""
    src1 = magpy.magnet.CylinderSegment(
        polarization=(11, 22, 33), dimension=(2, 4, 4, 0, 45)
    )
    src2 = magpy.magnet.CylinderSegment(
        polarization=(11, 22, 33), dimension=(2, 4, 4, -360, -315)
    )
    B1 = src1.getB((1, 0.5, 0.1))
    B2 = src2.getB((1, 0.5, 0.1))
    np.testing.assert_allclose(B1, B2)


def test_cylinder_tile_vs_fem():
    """test against fem results"""
    fd1, fd2, fd3, fd4 = np.load("tests/testdata/testdata_femDat_cylinder_tile2.npy")

    # chosen magnetization vectors
    mag1 = np.array((1, -1, 0)) / np.sqrt(2) * 1000
    mag2 = np.array((0, 0, 1)) * 1000
    mag3 = np.array((0, 1, -1)) / np.sqrt(2) * 1000

    # Magpylib magnet collection
    m1 = magpy.magnet.CylinderSegment(
        polarization=mag1,
        dimension=(1, 2, 1, -90, 0),
    )
    m2 = magpy.magnet.CylinderSegment(
        polarization=mag2,
        dimension=(1, 2.5, 1.5, 200, 250),
    )
    m3 = magpy.magnet.CylinderSegment(
        polarization=mag3,
        dimension=(0.75, 3, 0.5, 70, 180),
    )
    col = m1 + m2 + m3

    # create observer circles (see FEM screen shot)
    n = 101
    ts = np.linspace(0, 359.999, n) * np.pi / 180
    poso1 = np.array([0.5 * np.cos(ts), 0.5 * np.sin(ts), np.zeros(n)]).T
    poso2 = np.array([1.5 * np.cos(ts), 1.5 * np.sin(ts), np.zeros(n)]).T
    poso3 = np.array([1.5 * np.cos(ts), 1.5 * np.sin(ts), np.ones(n)]).T
    poso4 = np.array([3.5 * np.cos(ts), 3.5 * np.sin(ts), np.zeros(n)]).T

    # compute and plot fields
    B1 = col.getB(poso1)
    B2 = col.getB(poso2)
    B3 = col.getB(poso3)
    B4 = col.getB(poso4)

    amp1 = np.linalg.norm(B1, axis=1)
    amp2 = np.linalg.norm(B2, axis=1)
    amp3 = np.linalg.norm(B3, axis=1)
    amp4 = np.linalg.norm(B4, axis=1)

    # pylint: disable=unsubscriptable-object
    assert np.amax((fd1[:, 1:] * 1000 - B1).T / amp1) < 0.05
    assert np.amax((fd2[5:-5, 1:] * 1000 - B2[5:-5]).T / amp2[5:-5]) < 0.05
    assert np.amax((fd3[:, 1:] * 1000 - B3).T / amp3) < 0.05
    assert np.amax((fd4[:, 1:] * 1000 - B4).T / amp4) < 0.05


def test_cylinder_corner():
    """test corner =0 behavior"""
    a = 1
    s = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    B = s.getB(
        [
            [0, a, a],
            [0, a, -a],
            [0, -a, -a],
            [0, -a, a],
            [a, 0, a],
            [a, 0, -a],
            [-a, 0, -a],
            [-a, 0, a],
        ]
    )
    np.testing.assert_allclose(B, np.zeros((8, 3)))


def test_cylinder_corner_scaling():
    """test corner=0 scaling"""
    a = 1
    obs = [[a, 0, a + 1e-14], [a + 1e-14, 0, a]]
    s = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    Btest = [
        [5.12553286e03, -2.26623480e00, 2.59910242e02],
        [5.12803286e03, -2.26623480e00, 9.91024238e00],
    ]
    np.testing.assert_allclose(s.getB(obs), Btest)

    a = 1000
    obs = [[a, 0, a + 1e-14], [a + 1e-14, 0, a]]
    s = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    np.testing.assert_allclose(s.getB(obs), np.zeros((2, 3)))


def test_cylinder_scaling_invariance():
    """test scaling invariance"""
    obs = np.array(
        [
            [-0.12788963, 0.14872334, -0.35838915],
            [-0.17319799, 0.39177646, 0.22413971],
            [-0.15831916, -0.39768996, 0.41800279],
            [-0.05762575, 0.19985373, 0.02645361],
            [0.19120126, -0.13021813, -0.21615004],
            [0.39272212, 0.36457661, -0.09758084],
            [-0.39270581, -0.19805643, 0.36988649],
            [0.28942161, 0.31003054, -0.29558298],
            [0.13083584, 0.31396182, -0.11231319],
            [-0.04097917, 0.43394138, -0.14109254],
        ]
    )

    a = 1e-6
    s1 = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    Btest1 = s1.getB(obs * a)

    a = 1
    s2 = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    Btest2 = s2.getB(obs)

    a = 1e7
    s3 = magpy.magnet.Cylinder(polarization=(10, 10, 1000), dimension=(2 * a, 2 * a))
    Btest3 = s3.getB(obs * a)

    np.testing.assert_allclose(Btest1, Btest2)
    np.testing.assert_allclose(Btest1, Btest3)


def test_cylinder_diametral_small_r():
    """
    test if the transition from Taylor series to general case is smooth
    test if the gneral case fluctuations are small
    """
    B = BHJM_magnet_cylinder(
        field="B",
        observers=np.array([(x, 0, 3) for x in np.logspace(-1.4, -1.2, 1000)]),
        polarization=np.array([(1, 1, 0)] * 1000),
        dimension=np.array([(2, 2)] * 1000),
    )

    dB = np.log(abs(B[1:] - B[:-1]))
    ddB = abs(dB[1:] - dB[:-1])
    ddB = abs(ddB - np.mean(ddB, axis=0))

    assert np.all(ddB < 0.001)


def test_cyl_vs_cylseg_axial_H_inside_mask():
    """see https://github.com/magpylib/magpylib/issues/703"""
    field = "H"
    obs = np.array([(0.1, 0.2, 0.3)])
    pols = np.array([(0, 0, 1)])
    dims = np.array([(1, 1)])
    dims_cs = np.array([(0, 0.5, 1, 0, 360)])

    Bc = BHJM_magnet_cylinder(
        field=field,
        observers=obs,
        dimension=dims,
        polarization=pols,
    )
    Bcs = BHJM_cylinder_segment(
        field=field,
        observers=obs,
        dimension=dims_cs,
        polarization=pols,
    )
    np.testing.assert_allclose(Bc, Bcs)
