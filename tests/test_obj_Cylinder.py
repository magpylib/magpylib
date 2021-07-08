import numpy as np
import magpylib as mag3
from magpylib.magnet import Cylinder


def test_Cylinder_add():
    """ testing __add__
    """
    src1 = Cylinder((1,2,3),(1,2))
    src2 = Cylinder((1,2,3),(1,2))
    col = src1 + src2
    assert isinstance(col,mag3.Collection), 'adding cylinder fail'


def test_Cylinder_squeeze():
    """ testing squeeze output
    """
    src1 = Cylinder((1,1,1),(1,1))
    sensor = mag3.Sensor(pixel=[(1,2,3),(1,2,3)])
    B = src1.getB(sensor)
    assert B.shape==(2,3)
    H = src1.getH(sensor)
    assert H.shape==(2,3)

    B = src1.getB(sensor,squeeze=False)
    assert B.shape==(1,1,1,2,3)
    H = src1.getH(sensor,squeeze=False)
    assert H.shape==(1,1,1,2,3)


def test_repr():
    """ test __repr__
    """
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    assert pm2.__repr__()[:8] == 'Cylinder', 'Cylinder repr failed'


def test_Cylinder_getBH():
    """
    test Cylinder geB and getH with diffeent inputs
    vs the vectorized form
    """
    mag = (22,33,44)
    poso = (np.random.rand(100, 3)-.5)*5

    dim2 = (1,2)
    dim2_6 = (0,0.5,0,360,-1,1)

    dim5 = (1.1,2.2,30,145,5)
    dim5_6 = (1.1,2.2,30,145,-2.5,2.5)

    for dim,dim6 in zip([dim2, dim5], [dim2_6, dim5_6]):
        src = mag3.magnet.Cylinder(mag, dim)
        B1 = src.getB(poso)
        H1 = src.getH(poso)

        B2 = mag3.getBv(
            source_type='Cylinder',
            magnetization=mag,
            dimension=dim,
            observer=poso)
        H2 = mag3.getHv(
            source_type='Cylinder',
            magnetization=mag,
            dimension=dim,
            observer=poso)

        B3 = mag3.getBv(
            source_type='Cylinder',
            magnetization=mag,
            dimension=dim6,
            observer=poso)
        H3 = mag3.getHv(
            source_type='Cylinder',
            magnetization=mag,
            dimension=dim6,
            observer=poso)

        assert np.allclose(B1, B2)
        assert np.allclose(B1, B3)

        assert np.allclose(H1, H2)
        assert np.allclose(H1, H3)


def test_Cylinder_vs_old_inside():
    """
    test Cylinder vs old version
    """
    # inside
    mag = (np.random.rand(10,3)-.5)*1000
    dim = np.random.rand(10,2)+1
    poso = (np.random.rand(10, 3)-.5)

    mag3.Config.ITER_CYLINDER = 1000
    H_new = mag3.getHv(
        source_type='Cylinder',
        magnetization=mag,
        dimension=dim,
        observer=poso)
    H_old = mag3.getHv(
        source_type='Cylinder_old',
        magnetization=mag,
        dimension=dim,
        observer=poso)

    err = np.linalg.norm(H_new-H_old)/np.linalg.norm(H_new)
    assert err<1e-1
