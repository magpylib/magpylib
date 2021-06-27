import magpylib as mag3


def test_config():
    """ test setting and resetting the config
    """
    mag3.Config.ITER_CYLINDER = 15
    assert mag3.Config.ITER_CYLINDER == 15, 'setting config failed'
    mag3.Config.reset()
    assert mag3.Config.ITER_CYLINDER == 50, 'resetting config failed'
