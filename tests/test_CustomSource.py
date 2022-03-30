import pytest
import numpy as np
import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput, MagpylibInternalError

# pylint: disable=assignment-from-no-return
# pylint: disable=unused-argument

def constant_field(field, observer=(0,0,0)):
    """ constant field"""
    position = np.array(observer)
    length = 1 if position.ndim==1 else len(position)
    return np.array([[1, 2, 3]] * length)

def bad_Bfield_func(field, observer):
    """ another constant function without docstring"""
    return np.array([[1, 2, 3]])


def test_CustomSource_basicB():
    """Basic custom source class test"""
    external_field = magpy.misc.CustomSource(field_func=constant_field)

    B = external_field.getB((1, 2, 3))
    Btest = np.array((1, 2, 3))
    np.testing.assert_allclose(B, Btest)

    B = external_field.getB([[1, 2, 3], [4, 5, 6]])
    Btest = np.array([[1, 2, 3]] * 2)
    np.testing.assert_allclose(B, Btest)

    external_field.rotate_from_angax(45, "z")
    B = external_field.getB([[1, 2, 3], [4, 5, 6]])
    Btest = np.array([[-0.70710678, 2.12132034, 3.0]] * 2)
    np.testing.assert_allclose(B, Btest)


def test_CustomSource_basicH():
    """Basic custom source class test"""
    external_field = magpy.misc.CustomSource(field_func=constant_field)

    H = external_field.getH((1, 2, 3))
    Htest = np.array((1, 2, 3))
    np.testing.assert_allclose(H, Htest)

    H = external_field.getH([[1, 2, 3], [4, 5, 6]])
    Htest = np.array([[1, 2, 3]] * 2)
    np.testing.assert_allclose(H, Htest)

    external_field.rotate_from_angax(45, "z")
    H = external_field.getH([[1, 2, 3], [4, 5, 6]])
    Htest = np.array([[-0.70710678, 2.12132034, 3.0]] * 2)
    np.testing.assert_allclose(H, Htest)


def test_CustomSource_bad_inputs():
    """missing docstring"""
    with pytest.raises(MagpylibBadUserInput):
        magpy.misc.CustomSource(field_func='not a callable')

    with pytest.raises(MagpylibBadUserInput):
        magpy.misc.CustomSource(field_func=bad_Bfield_func)

    src = magpy.misc.CustomSource()
    with pytest.raises(MagpylibInternalError):
        src.getB([0,0,0])


def test_repr():
    """test __repr__"""
    dip = magpy.misc.CustomSource()
    assert dip.__repr__()[:12] == "CustomSource", "Custom_Source repr failed"
