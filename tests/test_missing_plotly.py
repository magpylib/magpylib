import sys
from importlib import reload
from unittest import mock
import pytest
import magpylib as magpy


def test_with_plotly_present():
    src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    x = src.display(renderer="png", backend="plotly")
    assert x is None, "display with plotly backend failed"


def test_with_plotly_is_missing():
    with mock.patch.dict(sys.modules, {"plotly": None}):
        src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
        with pytest.raises(ModuleNotFoundError):
            src.display(backend="plotly")
