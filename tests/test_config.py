import magpylib as magpy


def test_config():
    """ test setting and resetting the config
    """
    magpy.Config.ITER_CYLINDER = 15
    assert magpy.Config.ITER_CYLINDER == 15, 'setting config failed'
    magpy.Config.reset()
    assert magpy.Config.ITER_CYLINDER == 50, 'resetting config failed'
