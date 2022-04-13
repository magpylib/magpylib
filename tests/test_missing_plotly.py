import sys
from unittest import mock

import pytest

import magpylib as magpy


def test_with_plotly_present():
    """no docstring"""
    # pylint: disable=assignment-from-no-return
    src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    x = src.show(renderer="json", backend="plotly")
    assert x is None, "display with plotly backend failed"


def test_with_plotly_is_missing():
    """no docstring"""
    with mock.patch.dict(sys.modules, {"plotly": None}):
        src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
        with pytest.raises(ModuleNotFoundError):
            src.show(backend="plotly")
