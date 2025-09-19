import numpy as np
import pytest

import magpylib as magpy

# pylint: disable=assignment-from-no-return
# pylint: disable=unused-argument


def constant_field(field, observers=(0, 0, 0)):  # noqa: ARG001
    """constant field"""
    position = np.array(observers)
    length = 1 if position.ndim == 1 else len(position)
    return np.array([[1, 2, 3]] * length)


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


def test_CustomSource_None():
    "Set source field_func to None"
    # pylint: disable=protected-access
    external_field = magpy.misc.CustomSource(field_func=constant_field)
    external_field.field_func = None
    external_field._editable_field_func = False
    with pytest.raises(AttributeError):
        external_field.field_func = constant_field


def test_repr():
    """test __repr__"""
    dip = magpy.misc.CustomSource()
    assert repr(dip)[:12] == "CustomSource", "Custom_Source repr failed"


def test_CustomSource_volume():
    """Test CustomSource volume calculation (should be 0)."""

    def my_field(field, observers):
        if field == "B":
            return np.zeros_like(observers)
        return observers

    custom = magpy.misc.CustomSource(field_func=my_field)
    calculated = custom.volume
    expected = 0
    assert calculated == expected


def test_CustomSource_centroid():
    """Test CustomSource centroid - should return position"""
    expected = (11, 12, 13)

    def custom_field(field, observers):
        assert isinstance(field, str)
        return np.array([[0.01, 0, 0]] * len(observers))

    custom_source = magpy.misc.CustomSource(field_func=custom_field, position=expected)
    assert np.allclose(custom_source.centroid, expected)
