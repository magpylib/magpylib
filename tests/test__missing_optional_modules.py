import subprocess
import sys
from unittest.mock import patch

import pytest

import magpylib as magpy


def test_show_with_missing_pyvista():
    """Should raise if pyvista is not installed"""
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with patch.dict(sys.modules, {"pyvista": None}), pytest.raises(ModuleNotFoundError):
        src.show(return_fig=True, backend="pyvista")


def test_dataframe_output_missing_pandas():
    """test if pandas is installed when using dataframe output in getBH"""
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with patch.dict(sys.modules, {"pandas": None}), pytest.raises(ModuleNotFoundError):
        src.getB((0, 0, 0), output="dataframe")


def test_import_loads_no_plotting_library():
    """A script that only computes fields does not pay for matplotlib, plotly
    or pyvista: `import magpylib` loads none of them."""
    code = (
        "import sys, magpylib; "
        "print([m for m in ('matplotlib', 'plotly', 'pyvista') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
