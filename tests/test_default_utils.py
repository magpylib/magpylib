from copy import deepcopy
import pytest
from magpylib._lib.default_utils import update_nested_dict, magic_to_dict


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
