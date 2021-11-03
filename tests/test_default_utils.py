from copy import deepcopy
import pytest
from magpylib._lib.default_utils import (
    color_validator,
    update_nested_dict,
    magic_to_dict,
    linearize_dict,
    COLORS_MATPLOTLIB_TO_PLOTLY,
)


def test_update_nested_dict():
    """test all argument combinations of `update_nested_dicts`"""
    # `d` gets updated, that's why we deepcopy it
    d = {"a": 1, "b": {"c": 2, "d": None}, "f": None, "g": {"c": None, "d": 2}}
    u = {"a": 2, "b": 3, "e": 5, "g": {"c": 7, "d": 5}}
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=False, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "e": 5,
        "f": None,
        "g": {"c": 7, "d": 5},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "f": None,
        "g": {"c": 7, "d": 5},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=True
    )
    assert res == {
        "a": 1,
        "b": {"c": 2, "d": None},
        "f": None,
        "g": {"c": 7, "d": 2},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=False, replace_None_only=True
    )
    assert res == {
        "a": 1,
        "b": {"c": 2, "d": None},
        "f": None,
        "g": {"c": 7, "d": 2},
        "e": 5,
    }, "failed updating nested dict"


def test_magic_to_dict():
    """test all argument combinations of `magic_to_dict`"""
    d = {"a_b": 1, "c_d_e": 2, "a": 3, "c_d": {"e": 6}}
    res = magic_to_dict(d, separator="_")
    assert res == {"a": 3, "c": {"d": {"e": 6}}}
    d = {"a.b": 1, "c": 2, "a": 3, "c.d": {"e": 6}}
    res = magic_to_dict(d, separator=".")
    assert res == {"a": 3, "c": {"d": {"e": 6}}}
    with pytest.raises(AssertionError):
        magic_to_dict(0, separator=".")
        magic_to_dict(d, separator=0)


def test_linearize_dict():
    """test all argument combinations of `magic_to_dict`"""
    mydict = {
        "line": {"width": 1, "style": "solid", "color": None},
        "marker": {"size": 1, "symbol": "o", "color": None},
    }
    res = linearize_dict(mydict, separator=".")
    assert res == {
        "line.width": 1,
        "line.style": "solid",
        "line.color": None,
        "marker.size": 1,
        "marker.symbol": "o",
        "marker.color": None,
    }, "linearization of dict failed"
    with pytest.raises(AssertionError):
        magic_to_dict(0, separator=".")
        magic_to_dict(mydict, separator=0)


def test_color_validator():
    """test color validator based on matploblib validation"""

    assert color_validator("blue") == "blue", "should return `'blue'`"
    assert color_validator("r") == "red", "should return `'r'`"
    for shortC, longC in COLORS_MATPLOTLIB_TO_PLOTLY.items():
        assert color_validator(shortC) == longC, f"should return `'{longC}'`"
    assert color_validator(None) is None, "should return `'None'`"
    with pytest.raises(ValueError):
        color_validator(None, allow_None=False)
        color_validator("asdf")
        # does not support rgb values at the moment
        color_validator("rgb(255,255,255)")
