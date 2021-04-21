import magpylib as mag3


def test_config():
    """ test setting and resetting the config
    """
    mag3.Config.ITER_CYLINDER = 15
    assert mag3.Config.ITER_CYLINDER == 15, 'setting config failed'
    mag3.Config.reset()
    assert mag3.Config.ITER_CYLINDER == 50, 'resetting config failed'


def test_repr():
    """ test __repr__
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    pm3 = mag3.magnet.Sphere((1,2,3),3)
    col = mag3.Collection(pm1,pm2)
    sens = mag3.Sensor()
    dip = mag3.misc.Dipole(moment=(1,2,3))

    assert pm1.__repr__()[:3] == 'Box', 'Box repr failed'
    assert pm2.__repr__()[:8] == 'Cylinder', 'Cylinder repr failed'
    assert pm3.__repr__()[:6] == 'Sphere', 'Sphere repr failed'
    assert col.__repr__()[:10]== 'Collection', 'Collection repr failed'
    assert sens.__repr__()[:6]== 'Sensor', 'Sensor repr failed'
    assert dip.__repr__()[:6] == 'Dipole', 'Dipole repr failed'
