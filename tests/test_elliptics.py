import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.fields.special_cel import _cel, _cel_scalar, _cel_vector
from magpylib._src.fields.special_el3 import _el3, _el3_angle, _el3v, _el30

# def test_except_cel_scalar():
#    """bad _cel_scalar input"""
#    with pytest.raises(RuntimeError):
#        _cel_scalar(0, 0.1, 0.2, 0.3)


def test_except_el30():
    """bad el3"""
    with pytest.raises(RuntimeError):
        _el30(1, 1, -1)


def test_el3_inputs():
    """xxx"""
    assert _el30(0.1, 0.1, 1) == 0.09983241728793554
    assert _el30(1, 0.5, 0.25) == 1.0155300257327204


def test_el3_vs_original():
    """
    test new and vectorized el3 implementation vs original one
    """
    # store computations from original implementation
    # from florian_ell3_paper import el3 as _el30
    # N = 10000
    # x11 = np.random.rand(N)*5
    # kc11 = (np.random.rand(N)-.5)*10
    # p11 = (np.random.rand(N)-.5)*10
    # result0 = np.array([_el30(x, kc, p) for x,kc,p in zip(x11, kc11, p11)])
    # np.save('data_test_el3', np.array([result0,x11,kc11,p11]))

    # load data from original implementation
    data = np.load("tests/testdata/testdata_el3.npy")
    res0, x11, kc11, p11 = data

    # compare to vectorized
    resv = _el3v(x11, kc11, p11)
    np.testing.assert_allclose(res0, resv)

    # compare to modified original
    res1 = np.array([_el30(x, kc, p) for x, kc, p in zip(x11, kc11, p11, strict=False)])
    np.testing.assert_allclose(res0, res1)


def test_el3_angle_vs_original():
    """
    test vectorized _el3_angle implementation vs original one
    """
    # # store computations from original implementation of _el3_angle
    # N = 1000
    # phis = np.random.rand(N) * np.pi/2
    # ms = (np.random.rand(N)-.9)*10
    # ns = (np.random.rand(N)-.5)*5
    # result0 = np.array([_el3_angle0(phi, n, m) for phi,n,m in zip(phis, ns, ms)])
    # np.save('data_test_el3_angle', np.array([result0,phis,ns,ms]))

    # load data from original implementation
    data = np.load("tests/testdata/testdata_el3_angle.npy")
    res0, phis, ns, ms = data

    # compare to vectorized
    resv = _el3_angle(phis, ns, ms)
    np.testing.assert_allclose(res0, resv)


def test_el3s():
    """
    test _el30, _el3v, el3 vs each other
    """
    N = 999
    rng = np.random.default_rng()
    xs = (rng.random(N)) * 5
    kcs = (rng.random(N) - 0.5) * 10
    ps = (rng.random(N) - 0.5) * 10

    res0 = [_el30(x, kc, p) for x, kc, p in zip(xs, kcs, ps, strict=False)]
    res1 = _el3v(xs, kcs, ps)
    res2 = _el3(xs, kcs, ps)

    np.testing.assert_allclose(res0, res1)
    np.testing.assert_allclose(res1, res2)


def test_cels():
    """
    test cel, _cel_scalar (from florian) vs _cel_vector (from magpylib original)
    against each other
    """
    N = 999
    rng = np.random.default_rng()
    kcc = (rng.random(N) - 0.5) * 10
    pp = (rng.random(N) - 0.5) * 10
    cc = (rng.random(N) - 0.5) * 10
    ss = (rng.random(N) - 0.5) * 10

    res0 = [
        _cel_scalar(kc, p, c, s) for kc, p, c, s in zip(kcc, pp, cc, ss, strict=False)
    ]
    res1 = _cel_vector(kcc, pp, cc, ss)
    res2 = _cel(kcc, pp, cc, ss)

    np.testing.assert_allclose(res0, res1)
    np.testing.assert_allclose(res1, res2)


def test_cel_Bulirsch():
    """
    check _cel() results vs. values published in R. Bulirsch, Numerische Mathematik 13, 305-315 (1969), section 4.2
    """
    np.testing.assert_array_almost_equal_nulp(
        _cel_scalar(0.1, 4.1, 1.2, 1.1), 1.5464442694017956, nulp=3
    )
    np.testing.assert_array_almost_equal_nulp(
        _cel_scalar(0.1, -4.1, 1.2, 1.1), -0.67687378198360556, nulp=2
    )
    np.testing.assert_array_almost_equal_nulp(
        _cel_vector(np.array([0.1]), np.array([4.1]), np.array([1.2]), np.array([1.1])),
        [1.5464442694017956],
        nulp=3,
    )
    np.testing.assert_array_almost_equal_nulp(
        _cel_vector(
            np.array([0.1]), np.array([-4.1]), np.array([1.2]), np.array([1.1])
        ),
        [-0.67687378198360556],
        nulp=2,
    )


CYL_AX = magpy.magnet.Cylinder(dimension=(2, 1), polarization=(0, 0, 1))
CYL_DIA = magpy.magnet.Cylinder(dimension=(2, 1), polarization=(1, 0, 0))
SEG = magpy.magnet.CylinderSegment(
    dimension=(1, 2, 1, 0, 90), polarization=(0.1, -0.2, 1)
)
SEG_CORNER = magpy.magnet.CylinderSegment(
    dimension=(0.5, 1, 2, 0, 90), polarization=(0.6, 0, 0.8)
)
FINITE = np.array([(1.5, 0.3, 0.2), (0.4, -0.7, 0.9), (2, 2, -1)] * 4)  # 12 points


@pytest.mark.parametrize("n", [1, 12], ids=["1 point", "12 points"])
@pytest.mark.parametrize(
    ("src", "obs", "expect_nan"),
    [
        (CYL_AX, (np.nan, 0, 0), True),
        (CYL_AX, (np.inf, 0, 0), True),
        (CYL_AX, (0, 0, np.inf), True),
        (CYL_DIA, (np.inf, 0, 0), True),
        (SEG, (np.nan, 1, 0), True),
        # finite observers: only check that the call returns; the value is the
        # subject of #1073 and #1082
        (CYL_AX, (1e300, 0, 0), False),
        (SEG, (-1e300, -1e300, 1e300), False),
        (SEG_CORNER, (1 + 1.7e-9, 3.4e-9, 1 + 5.1e-9), False),
    ],
    ids=[
        "Cylinder axial nan",
        "Cylinder axial x inf",
        "Cylinder axial z inf",
        "Cylinder diametral inf",
        "CylinderSegment nan",
        "Cylinder 1e300",
        "CylinderSegment 1e300",
        "CylinderSegment near corner",
    ],
)
def test_elliptic_nonfinite_returns(src, obs, expect_nan, n):
    """Non-finite values in cel or el3 do not hang or raise (#1092, #1082)"""
    with np.errstate(all="ignore"):
        B = src.getB(np.tile(obs, (n, 1)))
    if expect_nan:
        assert np.isnan(B).all()


@pytest.mark.parametrize(
    "src",
    [CYL_AX, CYL_DIA, SEG],
    ids=["Cylinder axial", "Cylinder diametral", "CylinderSegment"],
)
def test_elliptic_nan_does_not_disturb_batch(src):
    """A NaN observer in a batch leaves the other observers' fields unchanged"""
    with np.errstate(all="ignore"):
        B = src.getB(np.vstack([FINITE, (np.nan, 0, 0)]))
    np.testing.assert_array_equal(B[:-1], src.getB(FINITE))
    assert np.isnan(B[-1]).all()


def test_el3_odd_in_x():
    """el3 is odd in x; negative x raised ValueError in the scalar version"""
    x = np.array([-1.0, -0.5, -3.0, -40.0] * 3)
    kc = np.array([0.5, 2.0, 0.01, 0.07] * 3)
    p = np.array([0.5, 3.0, -2.0, 0.02] * 3)
    v = _el3v(x, kc, p)
    assert np.isfinite(v).all()
    np.testing.assert_array_equal(v, -_el3v(-x, kc, p))
    for args in zip(x[:4], kc[:4], p[:4], strict=True):
        assert _el30(*args) == -_el30(-args[0], *args[1:])
