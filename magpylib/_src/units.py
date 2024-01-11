# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
from functools import wraps
from math import log10

import numpy as np

from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.exceptions import MagpylibBadUserInput

# Set pint unit registry. This needs to be unique through the library."
try:
    from pint import UnitRegistry as _UnitRegistry

    ureg = _UnitRegistry()
    unit_lib_found = True
except ImportError as missing_module:
    # error only raised when ureg becomes necessary in the code
    ureg = None
    unit_lib_found = False


MU0 = 4 * np.pi * 1e-7

_UNIT_PREFIX = {
    -24: "y",  # yocto
    -21: "z",  # zepto
    -18: "a",  # atto
    -15: "f",  # femto
    -12: "p",  # pico
    -9: "n",  # nano
    -6: "µ",  # micro
    -3: "m",  # milli
    0: "",
    3: "k",  # kilo
    6: "M",  # mega
    9: "G",  # giga
    12: "T",  # tera
    15: "P",  # peta
    18: "E",  # exa
    21: "Z",  # zetta
    24: "Y",  # yotta
}


def unit_prefix(number, unit="", precision=3, char_between="") -> str:
    """
    displays a number with given unit and precision and uses unit prefixes for the exponents from
    yotta (y) to Yocto (Y). If the exponent is smaller or bigger, falls back to scientific notation.
    Parameters
    ----------
    number : int, float
        can be any number
    unit : str, optional
        unit symbol can be any string, by default ""
    precision : int, optional
        gives the number of significant digits, by default 3
    char_between : str, optional
        character to insert between number of prefix. Can be " " or any string, if a space is wanted
        before the unit symbol , by default ""
    Returns
    -------
    str
        returns formatted number as string
    """
    number = downcast(number, unit)
    digits = int(log10(abs(number))) // 3 * 3 if number != 0 else 0
    prefix = _UNIT_PREFIX.get(digits, "")

    if prefix == "":
        digits = 0
    new_number_str = f"{number / 10 ** digits:.{precision}g}"
    return f"{new_number_str}{char_between}{prefix}{unit}"


def downcast(inp, unit):
    """convert to SI units if obj is a Quantity"""
    if isinstance(inp, (list, tuple)):
        return type(inp)([downcast(i, unit) for i in inp])
    if is_Quantity(inp):
        downcast.units_used = True
        inp = inp.to(unit)
        inp = inp.magnitude
    return inp


def is_Quantity(inp):
    """Return True if value as a pint Quantity else False"""
    return unit_lib_found and isinstance(inp, ureg.Quantity)


def to_Quantity(inp, unit):
    """Convert to quantity"""
    if not unit_lib_found:
        raise_missing_unit_package("pint")
    return ureg.Quantity(inp, unit)


def to_unit_from_target(inp, *, target, default_unit):
    """Transform to target unit if any otherwise SI"""
    if is_Quantity(target):
        if not is_Quantity(inp):
            inp = ureg.Quantity(inp, default_unit)
        return inp.to(target.units)
    return downcast(inp, default_unit)


def raise_missing_unit_package(lib):
    """Raise ModuleNotFoundError if no unit package is found"""
    units_mode = default_settings.units.mode
    if lib == "pint":
        raise ModuleNotFoundError(
            f"In order to use units in Magpylib with {units_mode!r} units mode, "
            "you need to install the `pint` package, "
            "see https://pint.readthedocs.io/en/stable/getting/index.html#installation"
        )


def unit_checker():
    """Decorator to add unit checks via pint"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            units_mode = default_settings.units.mode
            inp = args[0]
            sig_name = kwargs.get("sig_name", "")
            unit = kwargs.pop("unit", None)
            inp_unit = None
            is_unit_like_as_list = (
                isinstance(inp, (list, tuple))
                and len(inp) == 2
                and isinstance(inp[-1], str)
            )
            is_unit_like = isinstance(inp, str) or is_unit_like_as_list
            if not unit_lib_found and (
                is_unit_like or units_mode in ("upcast", "coerce")
            ):
                raise_missing_unit_package("pint")
            is_quantity = is_Quantity(inp)
            out_to_units = is_quantity or is_unit_like
            if out_to_units:
                if units_mode == "forbid":
                    raise MagpylibBadUserInput(
                        f"while the units mode is set to {units_mode!r},"
                        f" input parameter {sig_name!r} is unit-like ({inp!r}) "
                    )
                if is_unit_like_as_list:
                    inp, inp_unit = inp
                from pint.errors import PintError

                sig_str = f" `{sig_name!r}`" if sig_name else ""
                if isinstance(inp, ureg.Quantity):
                    inp_wu = inp
                else:
                    try:
                        inp_wu = ureg.Quantity(inp, inp_unit)
                    except PintError as msg:
                        raise MagpylibBadUserInput(
                            f"{msg}\nInput parameter{sig_str} must be in compatible units "
                            f"of {unit!r}."
                        ) from msg
                if unit is not None:
                    # pint unit check returns False even within context for A/m <-> T conversion
                    unit_check = inp_wu.check(unit)
                    if not unit_check:
                        try:  # if in Gaussian context, this will work
                            inp_wu.to(unit)
                            unit_check = True
                        except PintError:
                            # reraised after if unit_check is False
                            pass
                    if not unit_check:
                        raise MagpylibBadUserInput(
                            f"Input parameter{sig_str} must be in compatible units of {unit!r}."
                            f" Instead received {inp_wu.units!r}."
                        )
                args = (inp_wu.m, *args[1:])
            res = func(*args, **kwargs)
            if out_to_units:
                res = ureg.Quantity(res, inp_wu.units)
                if units_mode == "base" and unit is not None:
                    res = res.to(unit)
                elif units_mode == "downcast":
                    res = res.to(unit).m
            elif units_mode == "upcast":
                res = ureg.Quantity(res, unit)
            return res

        return wrapper

    return decorator
