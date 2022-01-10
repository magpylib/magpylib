import pytest
import numpy as np
import magpylib as magpy
from magpylib._src.exceptions import MagpylibInternalError

# pylint: disable=assignment-from-no-return
# pylint: disable=unused-argument

def constant_Bfield(position=((0, 0, 0))):
    """ constant field"""
    return np.array([[1, 2, 3]] * len(position))

def constant_Hfield(position=((0, 0, 0))):
    """ constant field - no idea why we need this """
    return np.array([[4, 5, 6]] * len(position))

def bad_Bfield_func(position):
    """ another constant function without docstring"""
    return np.array([[1, 2, 3]])

def test_CustomSource_basicB():
    """Basic custom source class test"""
    external_field = magpy.misc.CustomSource(field_B_lambda=constant_Bfield)

    B = external_field.getB([[1, 2, 3], [4, 5, 6]])
    Btest = np.array([[1, 2, 3]] * 2)
    assert np.allclose(B, Btest)

    external_field.rotate_from_angax(45, "z")
    B = external_field.getB([[1, 2, 3], [4, 5, 6]])
    Btest = np.array([[-0.70710678, 2.12132034, 3.0]] * 2)
    assert np.allclose(B, Btest)


def test_CustomSource_basicH():
    """Basic custom source class test"""
    external_field = magpy.misc.CustomSource(field_H_lambda=constant_Hfield)

    H = external_field.getH([[1, 2, 3], [4, 5, 6]])
    Htest = np.array([[4, 5, 6]] * 2)
    assert np.allclose(H, Htest)

    external_field.rotate_from_angax(35, "x")
    H = external_field.getH([[1, 2, 3], [4, 5, 6]])
    Htest = np.array([[4.0, 0.6543016, 7.78279445]] * 2)
    assert np.allclose(H, Htest)

def test_CustomSource_bad_inputs():
    with pytest.raises(AssertionError):
        magpy.misc.CustomSource(field_H_lambda='not a callable')

    with pytest.raises(AssertionError):
        magpy.misc.CustomSource(field_H_lambda=bad_Bfield_func)

    src = magpy.misc.CustomSource()
    with pytest.raises(MagpylibInternalError):
        src.getB([0,0,0])

def test_repr():
    """test __repr__"""
    dip = magpy.misc.CustomSource()
    assert dip.__repr__()[:12] == "CustomSource", "Custom_Source repr failed"
