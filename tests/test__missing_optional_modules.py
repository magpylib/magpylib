import sys
from unittest.mock import patch

import pytest

import magpylib as magpy


def test_show_with_missing_pyvista():
    """Should raise if pyvista is not installed"""
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ModuleNotFoundError):
            src.show(return_fig=True, backend="pyvista")


def test_dataframe_output_missing_pandas():
    """test if pandas is installed when using dataframe output in `getBH`"""
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with patch.dict(sys.modules, {"pandas": None}):
        with pytest.raises(ModuleNotFoundError):
            src.getB((0, 0, 0), output="dataframe")
