import magpylib as magpy


def test_defaults():
    """ test setting and resetting the config
    """
    magpy.defaults.itercylinder = 15
    assert magpy.defaults.itercylinder == 15, 'setting config failed'
    magpy.defaults.reset()
    assert magpy.defaults.itercylinder == 50, 'resetting config failed'
