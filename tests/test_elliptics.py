import numpy as np
from magpylib._lib.fields.special_el3 import el30, el3v, el3, el3_angle
from magpylib._lib.fields.special_cel import cel0, celv, cel

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

    # load data from orginal implementation
    data = np.load('tests/testdata/testdata_el3.npy')
    res0, x11, kc11, p11 = data

    # compare to vectorized
    resv = el3v(x11, kc11, p11)
    assert np.allclose(res0, resv)

    # compare to modified original
    res1 = np.array([el30(x, kc, p) for x,kc,p in zip(x11,kc11,p11)])
    assert np.allclose(res0, res1)


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

    # load data from orginal implementation
    data = np.load('tests/testdata/testdata_el3_angle.npy')
    res0, phis, ns, ms = data

    # compare to vectorized
    resv = el3_angle(phis, ns, ms)
    assert np.allclose(res0, resv)


def test_el3s():
    """
    test el30, el3v, el3 vs each other
    """
    N = 999
    xs = (np.random.rand(N))*5
    kcs = (np.random.rand(N)-.5)*10
    ps = (np.random.rand(N)-.5)*10

    res0 = [el30(x, kc, p) for x,kc,p in zip(xs, kcs, ps)]
    res1 = el3v(xs, kcs, ps)
    res2 = el3(xs, kcs, ps)

    assert np.allclose(res0, res1)
    assert np.allclose(res1, res2)


def test_cels():
    """
    test cel, cel0 (from florian) vs celv (from magpylib original)
    against each other
    """
    N = 999
    kcc = (np.random.rand(N)-.5)*10
    pp = (np.random.rand(N)-.5)*10
    cc = (np.random.rand(N)-.5)*10
    ss = (np.random.rand(N)-.5)*10

    res0 = [cel0(kc, p, c, s) for kc,p,c,s in zip(kcc,pp,cc,ss)]
    res1 = celv(kcc, pp, cc, ss)
    res2 = cel(kcc, pp, cc, ss)

    assert np.allclose(res0, res1)
    assert np.allclose(res1, res2)
