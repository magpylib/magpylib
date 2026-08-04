import numpy as np
import pytest

from magpylib._src.defaults.defaults_utility import (
    COLORS_SHORT_TO_LONG,
    color_validator,
    get_defaults_dict,
    linearize_dict,
    magic_to_dict,
)


def test_magic_to_dict():
    """test all argument combinations of magic_to_dict"""
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
    """test all argument combinations of magic_to_dict"""
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


@pytest.mark.parametrize(
    ("color", "allow_None", "color_expected"),
    [
        (None, True, None),
        ("blue", True, "blue"),
        (0, True, "#000000"),
        (0.5, True, "#7f7f7f"),
        pytest.param("0.5", True, "#7f7f7f", id="str0.5-True-#7f7f7f"),
        ((127, 127, 127), True, "#7f7f7f"),
        ("rgb(127, 127, 127)", True, "#7f7f7f"),
        ((0, 0, 0, 0), False, "#000000"),
        ((0.1, 0.2, 0.3), False, "#19334c"),
        ([127, 127, 127], True, "#7f7f7f"),
        ([0.1, 0.2, 0.3], False, "#19334c"),
        pytest.param(
            np.array([0.1, 0.2, 0.3]), False, "#19334c", id="ndarray-uncached"
        ),
    ]
    + [(shortC, True, longC) for shortC, longC in COLORS_SHORT_TO_LONG.items()],
)
def test_good_colors(color, allow_None, color_expected):
    """test color validator based on matploblib validation"""

    assert color_validator(color, allow_None=allow_None) == color_expected


@pytest.mark.parametrize(
    ("color", "allow_None", "expected_exception"),
    [
        (None, False, ValueError),
        (-1, False, ValueError),
        ((-1, 0, 0), False, ValueError),
        ((1, 2), False, ValueError),
        ((0, 0, 260), False, ValueError),
        ((0, "0", 200), False, ValueError),
        ("rgb(a, 0, 260)", False, ValueError),
        ("2", False, ValueError),
        ("mybadcolor", False, ValueError),
    ],
)
def test_bad_colors(color, allow_None, expected_exception):
    """test color validator based on Matplotlib validation"""

    with pytest.raises(expected_exception):
        color_validator(color, allow_None=allow_None)


def test_get_defaults_dict():
    """test get_defaults_dict"""
    s0 = get_defaults_dict("display.style")
    s1 = get_defaults_dict()["display"]["style"]
    assert s0 == s1, "dicts don't match"
