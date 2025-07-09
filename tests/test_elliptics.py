from __future__ import annotations

import numpy as np
import pytest

from magpylib._src.fields.special_cel import cel, cel0, celv
from magpylib._src.fields.special_el3 import el3, el3_angle, el3v, el30


def test_except_cel0():
    """bad cel0 input"""
    with pytest.raises(RuntimeError):
        cel0(0, 0.1, 0.2, 0.3)


def test_except_el30():
    """bad el3"""
    with pytest.raises(RuntimeError):
        el30(1, 1, -1)


def test_el3_inputs():
    """xxx"""
    assert el30(0.1, 0.1, 1) == 0.09983241728793554
    assert el30(1, 0.5, 0.25) == 1.0155300257327204


def test_el3_vs_original():
    """
    test new and vectroized el3 implemtnation vs original one
    """
    # store computations from original implementation
    # from florian_ell3_paper import el3 as el30
    # N = 10000
    # x11 = np.random.rand(N)*5
    # kc11 = (np.random.rand(N)-.5)*10
    # p11 = (np.random.rand(N)-.5)*10
    # result0 = np.array([el30(x, kc, p) for x,kc,p in zip(x11,kc11,p11)])
    # np.save('data_test_el3', np.array([result0,x11,kc11,p11]))

    # load data from original implementation
    data = np.load("tests/testdata/testdata_el3.npy")
    res0, x11, kc11, p11 = data

    # compare to vectorized
    resv = el3v(x11, kc11, p11)
    np.testing.assert_allclose(res0, resv)

    # compare to modified original
    res1 = np.array([el30(x, kc, p) for x, kc, p in zip(x11, kc11, p11, strict=False)])
    np.testing.assert_allclose(res0, res1)


def test_el3_angle_vs_original():
    """
    test vectroized el3_angle implemtnation vs original one
    """
    # # store computations from original implementation of el3_angle
    # N = 1000
    # phis = np.random.rand(N) * np.pi/2
    # ms = (np.random.rand(N)-.9)*10
    # ns = (np.random.rand(N)-.5)*5
    # result0 = np.array([el3_angle0(phi, n, m) for phi,n,m in zip(phis,ns,ms)])
    # np.save('data_test_el3_angle', np.array([result0,phis,ns,ms]))

    # load data from original implementation
    data = np.load("tests/testdata/testdata_el3_angle.npy")
    res0, phis, ns, ms = data

    # compare to vectorized
    resv = el3_angle(phis, ns, ms)
    np.testing.assert_allclose(res0, resv)


def test_el3s():
    """
    test el30, el3v, el3 vs each other
    """
    N = 999
    rng = np.random.default_rng()
    xs = (rng.random(N)) * 5
    kcs = (rng.random(N) - 0.5) * 10
    ps = (rng.random(N) - 0.5) * 10

    res0 = [el30(x, kc, p) for x, kc, p in zip(xs, kcs, ps, strict=False)]
    res1 = el3v(xs, kcs, ps)
    res2 = el3(xs, kcs, ps)

    np.testing.assert_allclose(res0, res1)
    np.testing.assert_allclose(res1, res2)


def test_cels():
    """
    test cel, cel0 (from florian) vs celv (from magpylib original)
    against each other
    """
    N = 999
    rng = np.random.default_rng()
    kcc = (rng.random(N) - 0.5) * 10
    pp = (rng.random(N) - 0.5) * 10
    cc = (rng.random(N) - 0.5) * 10
    ss = (rng.random(N) - 0.5) * 10

    res0 = [cel0(kc, p, c, s) for kc, p, c, s in zip(kcc, pp, cc, ss, strict=False)]
    res1 = celv(kcc, pp, cc, ss)
    res2 = cel(kcc, pp, cc, ss)

    np.testing.assert_allclose(res0, res1)
    np.testing.assert_allclose(res1, res2)
