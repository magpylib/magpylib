import magpylib as mag3

def test_repr():
    """ test __repr__
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    col = mag3.Collection(pm1,pm2)
    sens = mag3.Sensor()

    assert pm1.__repr__()[:3]=='Box', 'Box repr failed'
    assert pm2.__repr__()[:8]=='Cylinder', 'Cylinder repr failed'
    assert col.__repr__()[:10]=='Collection', 'Collection repr failed'
    assert sens.__repr__()[:6]=='Sensor', 'Sensor repr failed'
