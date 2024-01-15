# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
from functools import wraps
from math import log10

import numpy as np

from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.exceptions import MagpylibBadUserInput


MU0 = 4 * np.pi * 1e-7


class Units:
    def __init__(self):
        self._registry = None

    @property
    def registry(self):
        return self._registry


units_global = Units()

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


class UnitHandler:
    pgk_link = ""

    def __init__(self):
        units_global._registry = self.ureg = self.get_registry()


class PintHandler(UnitHandler):
    pgk_link = "https://pint.readthedocs.io/en/stable/getting/index.html#installation"

    def get_registry(self):
        import pint

        # Set pint unit registry. This needs to be unique through the library."
        return pint.UnitRegistry()

    def is_unit(self, inp):
        return isinstance(inp, self.ureg.Quantity)

    def to_quantity(self, inp, unit):
        return self.ureg.Quantity(inp, unit)

    def to_unit(self, inp, unit):
        return inp.to(unit)

    def get_unit(self, inp):
        return inp.units

    def check_unit(self, inp, unit):
        unit_check = inp.check(unit)
        # pint unit check returns False even within context for A/m <-> T conversion
        if not unit_check:
            from pint.errors import PintError

            try:  # if in Gaussian context, this will work
                inp.to(unit)
                unit_check = True
            except PintError:
                pass
        return unit_check

    def get_magnitude(self, inp):
        return inp.magnitude


class UnytHandler(UnitHandler):
    pgk_link = "https://unyt.readthedocs.io/en/stable/installation.html"

    def get_registry():
        import unyt

    def is_unit(self, inp):
        ...

    def to_unit(self, inp, unit):
        ...

    def get_unit(self, inp):
        ...

    def check_unit(self, inp, unit):
        ...

    def get_magnitude(self, inp):
        ...


def get_units_handler(error="ignore"):
    pkg = default_settings.units.package
    handler = handlers[pkg]
    if not isinstance(handler, UnitHandler):
        try:
            handler = handlers[pkg]()
            handlers[pkg] = handler
        except ImportError as msg:
            # error only raised when it becomes necessary in the code
            handler = None
            if error == "raise":
                raise_missing_unit_package(pkg)
    return handler


handlers = {"pint": PintHandler, "unyt": UnytHandler}


def downcast(inp, unit, units_handler=None):
    """convert to SI units if obj is a Quantity"""
    if isinstance(inp, (list, tuple)):
        return type(inp)([downcast(i, unit, units_handler=units_handler) for i in inp])
    if units_handler is None:
        units_handler = get_units_handler()
    if is_Quantity(inp, units_handler=units_handler):
        downcast.units_used = True
        inp = units_handler.to_unit(inp, unit)
        inp = units_handler.get_magnitude(inp)
    return inp


def is_Quantity(inp, units_handler=None):
    """Return True if value as a pint Quantity else False"""
    if units_handler is None:
        units_handler = get_units_handler()
    return units_handler is not None and units_handler.is_unit(inp)


def to_Quantity(inp, unit, sig_name="", units_handler=None):
    """Convert to quantity"""
    if units_handler is None:
        units_handler = get_units_handler(error="raise")
    try:
        if units_handler.is_unit(inp):
            inp = units_handler.to_unit(inp, unit)
        else:
            inp = units_handler.to_quantity(inp, unit)
    except Exception as msg:
        sig_str = f" `{sig_name!r}`" if sig_name else ""
        raise MagpylibBadUserInput(
            f"{msg}\nInput parameter{sig_str} must be in compatible units "
            f"of {unit!r}."
        ) from msg
    return inp


def to_unit_from_target(inp, *, target, default_unit, units_handler=None):
    """Transform to target unit if any otherwise SI"""
    if units_handler is None:
        units_handler = get_units_handler()
    if is_Quantity(target, units_handler=units_handler):
        if not is_Quantity(inp, units_handler=units_handler):
            inp = to_Quantity(inp, default_unit, units_handler=units_handler)
        return to_Quantity(inp, units_handler.get_unit(target))
    return downcast(inp, default_unit, units_handler=units_handler)


def raise_missing_unit_package(pkg):
    """Raise ModuleNotFoundError if no unit package is found"""
    units_mode = default_settings.units.mode
    msg = (
        f"In order to use units in Magpylib with {units_mode!r} units mode, "
        "you need to install the `pint` package."
    )
    link = handlers[pkg].pgk_link
    if link is not None:
        msg += f"see {link}"
    raise ModuleNotFoundError(msg)


def unit_checker():
    """Decorator to add unit checks"""

    def decorator(func):
        units_mode = default_settings.units.mode
        units_package_default = default_settings.units.package
        units_handler = get_units_handler()

        @wraps(func)
        def wrapper(*args, **kwargs):
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
            if units_handler is None and (
                is_unit_like or units_mode in ("upcast", "coerce")
            ):
                raise_missing_unit_package(units_package_default)
            out_to_units = is_Quantity(inp) or is_unit_like
            if out_to_units:
                if units_mode == "forbid":
                    raise MagpylibBadUserInput(
                        f"while the units mode is set to {units_mode!r},"
                        f" input parameter {sig_name!r} is unit-like ({inp!r}) "
                    )
                if is_unit_like_as_list:
                    inp, inp_unit = inp
                if is_Quantity(inp):
                    inp_wu = inp
                else:
                    inp_wu = to_Quantity(inp, inp_unit, sig_name=sig_name)
                if unit is not None and not units_handler.check_unit(inp_wu, unit):
                    sig_str = f" `{sig_name!r}`" if sig_name else ""
                    raise MagpylibBadUserInput(
                        f"Input parameter{sig_str} must be in compatible units of {unit!r}."
                        f" Instead received {inp_wu.units!r}."
                    )
                args = (units_handler.get_magnitude(inp_wu), *args[1:])
            res = func(*args, **kwargs)
            if out_to_units:
                res = to_Quantity(res, units_handler.get_unit(inp_wu))
                if units_mode == "base" and unit is not None:
                    res = to_Quantity(unit)
                elif units_mode == "downcast":
                    res = units_handler.get_magnitude(to_Quantity(unit))
            elif units_mode == "upcast":
                res = to_Quantity(res, unit)
            return res

        return wrapper

    return decorator
