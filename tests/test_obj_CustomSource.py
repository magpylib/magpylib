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


def test_CustomSource_centroid():
    """Test CustomSource centroid - should return position"""
    expected = (11, 12, 13)

    def custom_field(field, observers):
        assert isinstance(field, str)
        return np.array([[0.01, 0, 0]] * len(observers))

    custom_source = magpy.misc.CustomSource(field_func=custom_field, position=expected)
    assert np.allclose(custom_source.centroid, expected)


@pytest.mark.parametrize("in_collection", [False, True])
def test_CustomSource_bound_method_copy(in_collection):
    """A copied source's bound field method reads the copy's own state."""

    class UniformSource(magpy.misc.CustomSource):
        """Uniform field with mutable strength."""

        def __init__(self, strength):
            self.strength = strength
            super().__init__(field_func=self.compute_field)

        def compute_field(self, field, observers):
            """Return a uniform local x-directed field."""
            result = np.zeros(observers.shape)
            result[:, 0] = self.strength
            return result if field == "B" else result / magpy.mu_0

    source = UniformSource(1)
    original = magpy.Collection(source) if in_collection else source
    duplicate = original.copy()
    copied_source = duplicate[0] if in_collection else duplicate
    source.strength = 2
    np.testing.assert_allclose(original.getB((1, 0, 0)), (2, 0, 0))
    np.testing.assert_allclose(duplicate.getB((1, 0, 0)), (1, 0, 0))

    copied_source.strength = 3
    duplicate.rotate_from_angax(90, "z")
    np.testing.assert_allclose(duplicate.getB((1, 0, 0)), (0, 3, 0), atol=1e-14)
    np.testing.assert_allclose(
        duplicate.getH((1, 0, 0)) * magpy.mu_0, (0, 3, 0), atol=1e-14
    )
    np.testing.assert_allclose(original.getB((1, 0, 0)), (2, 0, 0))


def test_CustomSource_callable_object_copy():
    """A callable object can supply an independently copied field."""

    class UniformField:
        """Uniform field callback."""

        def __init__(self, strength):
            self.strength = strength

        def __call__(self, field, observers):
            result = np.full(observers.shape, self.strength, dtype=float)
            return result if field == "B" else result / magpy.mu_0

    callback = UniformField(1)
    source = magpy.misc.CustomSource(field_func=callback)
    duplicate = source.copy()
    callback.strength = 2
    np.testing.assert_allclose(source.getB((1, 0, 0)), (2, 2, 2))
    np.testing.assert_allclose(duplicate.getB((1, 0, 0)), (1, 1, 1))
