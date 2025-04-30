from __future__ import annotations

import importlib.metadata

import magpylib as m


def test_version():
    assert importlib.metadata.version("magpylib") == m.__version__
