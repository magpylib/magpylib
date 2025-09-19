import importlib.metadata

import magpylib as m


def test_version():
    assert importlib.metadata.version("magpylib") == m.__version__
