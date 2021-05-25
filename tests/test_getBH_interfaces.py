import numpy as np
import magpylib as mag3


def test_getB_interfaces1():
    """ self-consitent test of different possibilities for computing the field
    """
    src = mag3.magnet.Box((1,2,3), (1,2,3))
    src.move([(.1,.2,.3)]*10, increment=True)
    poso = [[(-1,-1,-1)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)
    B = mag3.getBv(
        src_type='Box',
        pos=src.pos,
        mag=(1,2,3),
        dim=(1,2,3),
        pos_obs=(-1,-1,-1))
    B1 = np.tile(B,(2,2,1,1))
    B1 = np.swapaxes(B1,0,2)

    B_test = mag3.getB(src, sens)
    assert B_test.shape == B1.shape, 'mag3.getB1 shape failed'
    assert np.allclose(B1, B_test), 'mag3.getB1 value failed'

    B_test = src.getB(poso)
    assert B_test.shape == B1.shape, 'src.getB shape failed'
    assert np.allclose(B1, B_test), 'src.getB value failed'

    B_test = src.getB(sens)
    assert B_test.shape == B1.shape, 'src.getB3 shape failed'
    assert np.allclose(B1, B_test), 'src.getB3 value failed'

    B_test = sens.getB(src)
    assert B_test.shape == B1.shape, 'sens.getB1 shape failed'
    assert np.allclose(B1, B_test), 'sens.getB1 value failed'


def test_getB_interfaces2():
    """ self-consitent test of different possibilities for computing the field
    """
    src = mag3.magnet.Box((1,2,3), (1,2,3))
    src.move([(.1,.2,.3)]*10, increment=True)
    poso = [[(-1,-1,-1)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)
    B = mag3.getBv(
        src_type='Box',
        pos=src.pos,
        mag=(1,2,3),
        dim=(1,2,3),
        pos_obs=(-1,-1,-1))

    B2 = np.tile(B,(2,2,2,1,1))
    B2 = np.swapaxes(B2,1,3)

    B_test = mag3.getB([src,src], sens)
    assert B_test.shape == B2.shape, 'mag3.getB2 shape failed'
    assert np.allclose(B2, B_test), 'mag3.getB2 value failed'

    B_test = sens.getB([src,src])
    assert B_test.shape == B2.shape, 'sens.getB2 shape failed'
    assert np.allclose(B2, B_test), 'sens.getB2 value failed'


def test_getB_interfaces3():
    """ self-consitent test of different possibilities for computing the field
    """
    src = mag3.magnet.Box((1,2,3), (1,2,3))
    src.move([(.1,.2,.3)]*10, increment=True)
    poso = [[(-1,-1,-1)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)
    B = mag3.getBv(
        src_type='Box',
        pos=src.pos,
        mag=(1,2,3),
        dim=(1,2,3),
        pos_obs=(-1,-1,-1))

    B3 = np.tile(B,(2,2,2,1,1))
    B3 = np.swapaxes(B3,0,3)

    B_test = mag3.getB(src, [sens,sens])
    assert B_test.shape == B3.shape, 'mag3.getB3 shape failed'
    assert np.allclose(B3, B_test), 'mag3.getB3 value failed'

    B_test = src.getB([poso,poso])
    assert B_test.shape == B3.shape, 'src.getB2 shape failed'
    assert np.allclose(B3, B_test), 'src.getB2 value failed'

    B_test = src.getB([sens,sens])
    assert B_test.shape == B3.shape, 'src.getB4 shape failed'
    assert np.allclose(B3, B_test), 'src.getB4 value failed'


def test_getH_interfaces1():
    """ self-consitent test of different possibilities for computing the field
    """
    mag=(22,-33,44)
    dim=(3,2,3)
    src = mag3.magnet.Box(mag,dim)
    src.move([(.1,.2,.3)]*10, increment=True)

    poso = [[(-1,-2,-3)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)

    H = mag3.getHv(
        src_type='Box',
        pos=src.pos,
        mag=mag,
        dim=dim,
        pos_obs=(-1,-2,-3))
    H1 = np.tile(H,(2,2,1,1))
    H1 = np.swapaxes(H1,0,2)

    H_test = mag3.getH(src, sens)
    assert H_test.shape == H1.shape, 'mag3.getH1 shape failed'
    assert np.allclose(H1, H_test), 'mag3.getH1 value failed'

    H_test = src.getH(poso)
    assert H_test.shape == H1.shape, 'src.getH shape failed'
    assert np.allclose(H1, H_test), 'src.getH value failed'

    H_test = src.getH(sens)
    assert H_test.shape == H1.shape, 'src.getH3 shape failed'
    assert np.allclose(H1, H_test), 'src.getH3 value failed'

    H_test = sens.getH(src)
    assert H_test.shape == H1.shape, 'sens.getH1 shape failed'
    assert np.allclose(H1, H_test), 'sens.getH1 value failed'


def test_getH_interfaces2():
    """ self-consitent test of different possibilities for computing the field
    """
    mag=(22,-33,44)
    dim=(3,2,3)
    src = mag3.magnet.Box(mag,dim)
    src.move([(.1,.2,.3)]*10, increment=True)

    poso = [[(-1,-2,-3)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)

    H = mag3.getHv(
        src_type='Box',
        pos=src.pos,
        mag=mag,
        dim=dim,
        pos_obs=(-1,-2,-3))

    H2 = np.tile(H,(2,2,2,1,1))
    H2 = np.swapaxes(H2,1,3)

    H_test = mag3.getH([src,src], sens)
    assert H_test.shape == H2.shape, 'mag3.getH2 shape failed'
    assert np.allclose(H2, H_test), 'mag3.getH2 value failed'

    H_test = sens.getH([src,src])
    assert H_test.shape == H2.shape, 'sens.getH2 shape failed'
    assert np.allclose(H2, H_test), 'sens.getH2 value failed'


def test_getH_interfaces3():
    """ self-consitent test of different possibilities for computing the field
    """
    mag=(22,-33,44)
    dim=(3,2,3)
    src = mag3.magnet.Box(mag,dim)
    src.move([(.1,.2,.3)]*10, increment=True)

    poso = [[(-1,-2,-3)]*2]*2
    sens = mag3.Sensor(pos_pix=poso)

    H = mag3.getHv(
        src_type='Box',
        pos=src.pos,
        mag=mag,
        dim=dim,
        pos_obs=(-1,-2,-3))

    H3 = np.tile(H,(2,2,2,1,1))
    H3 = np.swapaxes(H3,0,3)

    H_test = mag3.getH(src, [sens,sens])
    assert H_test.shape == H3.shape, 'mag3.getH3 shape failed'
    assert np.allclose(H3, H_test), 'mag3.getH3 value failed'

    H_test = src.getH([poso,poso])
    assert H_test.shape == H3.shape, 'src.getH2 shape failed'
    assert np.allclose(H3, H_test), 'src.getH2 value failed'

    H_test = src.getH([sens,sens])
    assert H_test.shape == H3.shape, 'src.getH4 shape failed'
    assert np.allclose(H3, H_test), 'src.getH4 value failed'
