from copy import deepcopy
import pytest
from magpylib._lib.default_utils import (
    MagicProperties,
    color_validator,
    get_defaults_dict,
    update_nested_dict,
    magic_to_dict,
    linearize_dict,
    COLORS_MATPLOTLIB_TO_PLOTLY,
)


def test_update_nested_dict():
    """test all argument combinations of `update_nested_dicts`"""
    # `d` gets updated, that's why we deepcopy it
    d = {"a": 1, "b": {"c": 2, "d": None}, "f": None, "g": {"c": None, "d": 2}, "h": 1}
    u = {"a": 2, "b": 3, "e": 5, "g": {"c": 7, "d": 5}, "h": {"i": 3}}
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=False, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "e": 5,
        "f": None,
        "g": {"c": 7, "d": 5},
        "h": {"i": 3},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "f": None,
        "g": {"c": 7, "d": 5},
        "h": {"i": 3},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=True
    )
    assert res == {
        "a": 1,
        "b": {"c": 2, "d": None},
        "f": None,
        "g": {"c": 7, "d": 2},
        "h": 1,
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
        "h": 1,
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
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
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
    with pytest.raises(ValueError):
        color_validator("asdf")
    with pytest.raises(ValueError):
        # does not support rgb values at the moment
        color_validator("rgb(255,255,255)")


def test_MagicProperties():
    """test MagicProperties class"""

    class BPsub1(MagicProperties):
        "MagicProperties class"

        @property
        def prop1(self):
            """prop1"""
            return self._prop1

        @prop1.setter
        def prop1(self, val):
            self._prop1 = val

    class BPsub2(MagicProperties):
        "MagicProperties class"

        @property
        def prop2(self):
            """prop2"""
            return self._prop2

        @prop2.setter
        def prop2(self, val):
            self._prop2 = val

    bp1 = BPsub1(prop1=1)

    # check setting attribute/property
    assert bp1.prop1 == 1, "`bp1.prop1` should be `1`"
    with pytest.raises(AttributeError):
        getattr(bp1, "prop1e")  # only properties are allowed to be set

    assert bp1.as_dict() == {"prop1": 1}, "`as_dict` method failed"

    bp2 = BPsub2(prop2=2)
    bp1.prop1 = bp2  # assigning class to subproperty

    # check as_dict method
    assert bp1.as_dict() == {"prop1": {"prop2": 2}}, "`as_dict` method failed"

    # check update method with different parameters
    assert bp1.update(prop1_prop2=10).as_dict() == {
        "prop1": {"prop2": 10}
    }, "magic property setting failed"

    with pytest.raises(AttributeError):
        bp1.update(prop1_prop2=10, prop3=4)
    assert bp1.update(prop1_prop2=10, prop3=4, _match_properties=False).as_dict() == {
        "prop1": {"prop2": 10}
    }, "magic property setting failed, should ignore `'prop3'`"

    assert bp1.update(prop1_prop2=20, _replace_None_only=True).as_dict() == {
        "prop1": {"prop2": 10}
    }, "magic property setting failed, `prop2` should be remained unchanged `10`"

    # check copy method

    bp3 = bp2.copy()
    assert bp3 is not bp2, "failed copying, should return a different id"
    assert (
        bp3.as_dict() == bp2.as_dict()
    ), "failed copying, should return the same property values"

    # check failing init
    with pytest.raises(AttributeError):
        BPsub1(a=0)  # `a` is not a property in the class

    # check repr
    assert repr(MagicProperties()) == "MagicProperties()", "repr failed"


def test_get_defaults_dict():
    """test get_defaults_dict"""
    s0 = get_defaults_dict("display.style")
    s1 = get_defaults_dict()["display"]["style"]
    assert s0 == s1, "dicts don't match"
