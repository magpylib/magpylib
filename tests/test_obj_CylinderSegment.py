import magpylib as magpy

def test_repr():
    """ test __repr__
    """
    pm2 = magpy.magnet.CylinderSegment((1,2,3),(1,2,3,0,90))
    assert pm2.__repr__()[:15] == 'CylinderSegment', 'CylinderSegment repr failed'
