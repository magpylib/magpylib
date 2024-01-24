# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
import abc
from functools import wraps
from math import log10

import numpy as np

from magpylib._src.defaults.defaults_utility import ALLOWED_UNITS_MODES
from magpylib._src.exceptions import MagpylibBadUserInput


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


# ---------------------------------------classes----------------------------------------------------


def check_call(func, *args, expected, verbose=False):
    """Call a function with the specified arguments and check against an expected value"""
    result = func(*args)
    eq = result == expected
    if isinstance(eq, np.ndarray):
        eq = eq.all()
    if not eq or verbose:
        args_repr = ", ".join(repr(arg) for arg in args)
        call_str = f"{func.__self__.__class__.__name__}.{func.__name__}({args_repr})"
        res_str = f"{result!r}"
        if not eq:
            raise ValueError(f"{call_str} expected {expected!r}, got {res_str}.")
        if verbose:
            print(f"{call_str} -> {res_str}")


class UnitHandler(metaclass=abc.ABCMeta):
    """
    An abstract base class to create a consistent interface for unit conversion.

    This base class ensures that all subclasses provide specific methods for
    unit conversion processes. It cannot be instantiated directly and requires
    subclassing with all abstract methods overridden.


    Attributes
    ----------
    pgk_link : str
        An URL pointing to the package documentation.

    Methods
    -------
    is_quantity(inp)
        Abstract method to check if the input is a unit-qualified quantity.

    to_quantity(inp, unit)
        Abstract method to convert input to a quantity with the specified units.

    to_unit(inp, unit)
        Abstract method to convert a quantity to the specified units.

    get_unit(inp)
        Abstract method to retrieve the unit of the input quantity.

    get_magnitude(inp)
        Abstract method to extract the magnitude from the input quantity.

    See Also
    --------
    PintHandler : A concrete implementation using the pint library.
    UnytHandler : A concrete implementation using the unyt library.

    Notes
    -----
    Subclasses must provide concrete implementations of all abstract methods and
    may require additional methods to handle specialized cases of unit conversions.
    """

    # pylint: disable=missing-function-docstring

    handlers = {}
    pkg_name = ""
    pgk_link = ""
    __validated = False

    def __init_subclass__(
        cls, *, pkg_name, validate_on_declaration=True, override=False, **kwargs
    ):
        # cannot use `name` parameter as it conflicts with ABCmeta
        super().__init_subclass__(**kwargs)
        pkg_name = str(pkg_name)
        if pkg_name in cls.handlers and not override:
            left_names = set(cls.handlers) - {pkg_name}
            raise ValueError(
                f"The UnitHandler name {pkg_name!r} is already in use, as well as {left_names}."
                " To replace an existing handler use `override=True` in the class declaration."
            )
        if validate_on_declaration:
            # avoid validation on handlers that use packages that may not be installed
            # wait for instantiation
            cls().validate()
        cls.pkg_name = pkg_name
        cls.handlers[pkg_name] = cls

    def __init__(self):
        if not self.__validated:
            self.validate()

    @abc.abstractmethod
    def is_quantity(self, inp):
        pass

    @abc.abstractmethod
    def to_quantity(self, inp, unit=None):
        pass

    @abc.abstractmethod
    def to_unit(self, inp, unit):
        pass

    @abc.abstractmethod
    def get_unit(self, inp):
        pass

    @abc.abstractmethod
    def get_magnitude(self, inp):
        pass

    def validate(self):
        """Validate new UnitHandler"""
        for inp in (1.23, [1, 2, 3], np.array([1.0, 1.2, 1.23])):
            q_cm = self.to_quantity(inp, "cm")
            q_mm = self.to_unit(q_cm, "mm")
            q_cm_unit = self.get_unit(q_cm)
            check_call(self.is_quantity, inp, expected=False)
            check_call(self.is_quantity, q_cm, expected=True)
            check_call(
                self.get_unit, self.to_quantity(inp, q_cm_unit), expected=q_cm_unit
            )
            check_call(self.get_magnitude, q_cm, expected=inp)
            check_call(self.get_magnitude, q_mm / 10, expected=inp)

        tq = self.to_quantity
        check_call(tq, "1.23mm", expected=tq(1.23 * 1e-3, "m"))
        check_call(tq, "1.23mm", "m", expected=tq("0.00123m"))
        check_call(tq, 1.23, "A*m**2", expected=tq("1.23A*m**2"))


class PintHandler(UnitHandler, pkg_name="pint", validate_on_declaration=False):
    """A concrete implementation of `UnitHandler` using the `pint` library.

    Attributes
    ----------
    pgk_link : str
        A URL link to the `pint` documentation for getting started and installation instructions.
    ureg : `pint.UnitRegistry`
        An instance of the `pint` UnitRegistry that manages definitions and conversions.
    """

    # pylint: disable=missing-function-docstring

    pgk_link = "https://pint.readthedocs.io/en/stable/getting/index.html#installation"

    def __init__(self):
        # pylint: disable=wrong-import-position
        # pint may not be installed in the user environment
        # should only trigger an ImportError when called for
        from pint import UnitRegistry

        # Set pint unit registry. This needs to be unique through the library."
        units_global._registry = self.ureg = UnitRegistry()
        super().__init__()

    def is_quantity(self, inp):
        return isinstance(inp, self.ureg.Quantity)

    def to_quantity(self, inp, unit=None):
        if isinstance(inp, str):
            res = self.ureg.Quantity(inp)
            if unit is not None:
                return res.to(unit)
            return res
        return self.ureg.Quantity(inp, unit)

    def to_unit(self, inp, unit):
        return inp.to(unit)

    def get_unit(self, inp):
        return inp.units

    def get_magnitude(self, inp):
        return inp.magnitude


class UnytHandler(UnitHandler, pkg_name="unyt", validate_on_declaration=False):
    """A concrete implementation of `UnitHandler` using the `unyt` library.

    Attributes
    ----------
    pgk_link : str
        A URL link to the `unyt` documentation for getting started and installation instructions.
    ureg : `unyt.UnitRegistry`
        An instance of the `unyt` UnitRegistry that manages definitions and conversions.
    unyt : `unyt.unyt_array`
        The `unyt` module imported into the handler.
    """

    # pylint: disable=missing-function-docstring

    pgk_link = "https://unyt.readthedocs.io/en/stable/installation.html"

    def __init__(self):
        # pylint: disable=wrong-import-position
        # unyt may not be installed in the user environment
        # should only trigger an ImportError when called for
        from unyt import UnitRegistry, unyt_quantity, unyt_array, Unit

        units_global._registry = self.ureg = UnitRegistry()
        self.unyt_quantity = unyt_quantity
        self.unyt_array = unyt_array
        self.Unit = Unit
        super().__init__()

    def is_quantity(self, inp):
        return isinstance(inp, self.unyt_array)

    def to_quantity(self, inp, unit=None):
        if isinstance(inp, str):
            res = self.unyt_quantity.from_string(inp)
            if unit is not None:
                return res.to(unit)
            return res
        return inp * self.Unit(unit)

    def to_unit(self, inp, unit):
        return inp.to(unit)

    def get_unit(self, inp):
        return inp.units

    def get_magnitude(self, inp):
        val = inp.value
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return val.tolist()
        return val


class AstropyHandler(UnitHandler, pkg_name="astropy", validate_on_declaration=False):
    """
    A concrete implementation of `UnitHandler` using the `astropy` library for handling units.

    This class provides methods to perform unit conversions and checks using `astropy.units`.
    It is designed to interact with `astropy.units.Quantity` objects.

    Attributes
    ----------
    pgk_link : str
        A URL link to the `astropy` documentation for getting started and installation.
    """

    # pylint: disable=missing-function-docstring

    pgk_link = "https://docs.astropy.org/en/stable/install.html"

    def __init__(self):
        # pylint: disable=wrong-import-position
        # astropy may not be installed in the user environment
        # should only trigger an ImportError when called for
        from astropy.units import Quantity, Unit

        self.Quantity = Quantity
        self.Unit = Unit
        super().__init__()

    def is_quantity(self, inp):
        return isinstance(inp, self.Quantity)

    def to_quantity(self, inp, unit=None):
        return self.Quantity(inp, unit=unit)

    def to_unit(self, inp, unit):
        return inp.to(unit)

    def get_unit(self, inp):
        return inp.unit

    def get_magnitude(self, inp):
        return inp.value


class Units:
    """
    A simple container for define units handling options.

    Attributes
    ----------
    package: {'pint', 'unyt', 'astropy'}
        Set Magpylib'S default unit package.

    mode: str, default='keep'
        Set Magpylib's units mode. Classes input parameters can be:
          - arrays or scalars
          - quantity object from a `UnitHandler`
          - unit-like inputs of the form (<value>, <units>) (e.g. [[1,2,3], 'meter'])
        When inputs are quantity objects, a dimensionality check is always performed. If it is
        only array-like or a scalar, it is assumed to be of base SI units. If it is unit-like it
        gets transformed into a quantity object.
        The following `units_mode` are implemented to cover a wide range of possible behaviors
        when dealing with units:
          - "consistent": either only units or none sould be used (first input determines the case).
          - "keep" : keep input object type.  Allows and stores derived units.
          - "downcast" : allow unit-like inputs but converts to base SI units, stores the magnitude
              only.
          - "upcast" : converts to units quantity object if input is unit-like, keep
              otherwise.
          - "base" : converts to base SI units quantity object (e.g. 'mm' -> 'm').
          - "coerce": forces inputs to be unit-like (raise if not).
          - "forbid": forbids unit-like inputs.

    registry : `pint.UnitRegistry` or `unyt.UnitRegistry` or None
        The unit registry used for unit conversions and definitions.

    in_use: bool or None, read only
        Tells if units are in use or not. If it is None, it means that it is undetermined.
        In mode='consistent', the first input is used to determine if units should be used
        throughout or not.
    """

    UnitHandler = UnitHandler

    def __init__(self):
        self._mode = "consistent"
        self._package = "pint"
        self._registry = None
        self._in_use = None
        self._first_param = None

    @property
    def in_use(self):
        """Boolean or None. Tells if units are in use or not. If it is None,
        it means that it is undetermined and yet to be set by first input."""
        return self._in_use

    @property
    def registry(self):
        """
        Accessor for the unit registry.

        Returns
        -------
        `pint.UnitRegistry` or `unyt.UnitRegistry` or None
            The current unit registry of the `Units` instance.
        """
        if self._registry is None:
            get_units_handler(error="raise")
        return self._registry

    @property
    def package(self):
        """Set Magpylib's default units package."""
        return self._package

    @package.setter
    def package(self, val):
        supported = tuple(units_global.UnitHandler.handlers)
        assert val in supported, (
            f"the `package` property of {type(self).__name__} must be one of"
            f" {supported}"
            f" but received {repr(val)} instead"
        )
        self._package = val

    @property
    def mode(self):
        """Set Magpylib's units mode"""
        return self._mode

    @mode.setter
    def mode(self, val):
        assert val is None or val in ALLOWED_UNITS_MODES, (
            f"the `mode` property of {type(self).__name__} must be one of"
            f" {ALLOWED_UNITS_MODES}"
            f" but received {repr(val)} instead"
        )
        self._reset_mode_params()
        self._mode = val

    def _reset_mode_params(self):
        """Reset mode parameters"""
        self._in_use = None
        self._first_param = None


units_global = Units()

# ---------------------------------------functions--------------------------------------------------


def get_units_handler(error="ignore"):
    """Retrieve the appropriate unit handler based on the default settings.

    Parameters
    ----------
    error : str, optional
        The error handling strategy when a unit package is missing. Defaults to 'ignore'.
        If 'raise', it will raise an exception if the package is missing.

    Returns
    -------
    UnitHandler
        An instance of the subclass of `UnitHandler` corresponding to the unit package.

    Raises
    ------
    ModuleNotFoundError
        If error is 'raise' and the required unit package is not installed.
    """

    pkg = units_global.package
    handlers = UnitHandler.handlers
    handler = handlers[pkg]
    if not isinstance(handler, UnitHandler):
        try:
            # instantiate UnitHandler subclass on first use
            handler = handlers[pkg]()
            handlers[pkg] = handler
        except ImportError:
            # error only raised when it becomes necessary in the code
            handler = None
            if error == "raise":
                raise_missing_unit_package(pkg)
    return handler


def raise_missing_unit_package(pkg):
    """Raise ModuleNotFoundError if no unit package is found"""
    units_mode = units_global.mode
    msg = (
        f"In order to use units in Magpylib with {units_mode!r} units mode, "
        "you need to install the `pint` package."
    )
    link = UnitHandler.handlers[pkg].pgk_link
    if link is not None:
        msg += f" (see {link})"
    raise ModuleNotFoundError(msg)


def is_Quantity(inp, units_handler=None):
    """Return True if value as a pint Quantity else False"""
    if units_handler is None:
        units_handler = get_units_handler()
    return units_handler is not None and units_handler.is_quantity(inp)


def to_Quantity(inp, unit, *, sig_name="", default_unit=None, units_handler=None):
    """Convert to quantity"""
    if units_handler is None:
        units_handler = get_units_handler(error="raise")
    try:
        if units_handler.is_quantity(inp):
            inp = units_handler.to_unit(inp, unit)
        else:
            inp = units_handler.to_quantity(inp, unit)
    except Exception as msg:
        sig_str = f" `{sig_name!r}`" if sig_name else ""
        expected = unit if default_unit is None else default_unit
        raise MagpylibBadUserInput(
            f"{msg}\nInput parameter{sig_str} cannot be converted to "
            f"units of {expected!r}."
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


def downcast(inp, unit, units_handler=None):
    """convert to SI units if obj is a Quantity"""
    if isinstance(inp, (list, tuple)):
        return type(inp)([downcast(i, unit, units_handler=units_handler) for i in inp])
    if units_handler is None:
        units_handler = get_units_handler()
    if is_Quantity(inp, units_handler=units_handler):
        inp = units_handler.to_unit(inp, unit)
        inp = units_handler.get_magnitude(inp)
    return inp


def unit_checker():
    """Decorator to add unit checks"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # pylint: disable=protected-access
            inp = args[0]
            if kwargs.get("allow_None", False) and inp is None:
                return None
            units_mode = units_global.mode
            units_package_default = units_global.package
            units_handler = get_units_handler()
            sig_name = kwargs.get("sig_name", "")
            unit = kwargs.pop("unit", None)
            inp_unit = unit
            is_quantity_like_as_list = (
                isinstance(inp, (list, tuple))
                and len(inp) == 2
                and isinstance(inp[-1], str)
            )
            is_quantity_like = isinstance(inp, str) or is_quantity_like_as_list
            out_to_units = is_Quantity(inp) or is_quantity_like
            if (
                units_mode in ("consistent", "coerce")
                and units_global.in_use
                and not out_to_units
                and isinstance(inp, (list, tuple, np.ndarray))
            ):
                # in consistent mode position may come as default=(0,0,0)
                # it has to adapt to the current expected mode, since it is not set
                # explicitly by the user
                inp = np.asarray(inp)
                if np.all(inp == 0):
                    out_to_units = True
            if units_handler is None and (
                is_quantity_like or units_mode in ("upcast", "coerce", "base")
            ):
                raise_missing_unit_package(units_package_default)
            if units_global.in_use is None:
                units_global._in_use = out_to_units
                units_global._first_param = (sig_name, inp)
            if units_mode == "consistent":
                if units_global.in_use != out_to_units:
                    s = (" not", "") if out_to_units else ("", " not")
                    f = units_global._first_param
                    raise MagpylibBadUserInput(
                        f"while magpylib.units.mode is set to {units_mode!r},"
                        f" input parameter {f[0]} is{s[0]} unit-like ({f[1]})"
                        f" but input parameter {sig_name!r} is{s[1]} ({inp!r}) "
                    )
            if out_to_units:
                if units_mode == "forbid":
                    raise MagpylibBadUserInput(
                        f"while magpylib.units.mode is set to {units_mode!r},"
                        f" input parameter {sig_name!r} is unit-like ({inp!r}) "
                    )
                if is_quantity_like_as_list:
                    inp, inp_unit = inp
                elif isinstance(inp, str):
                    inp_unit = None
                if is_Quantity(inp):
                    inp_wu = inp
                else:
                    inp_wu = to_Quantity(
                        inp, inp_unit, default_unit=unit, sig_name=sig_name
                    )
                if unit is not None:
                    try:
                        units_handler.to_unit(inp_wu, unit)
                    except Exception as msg:
                        sig_str = f" `{sig_name!r}`" if sig_name else ""
                        raise MagpylibBadUserInput(
                            f"Input parameter{sig_str} must be in compatible units of {unit!r}."
                            f" Instead received {units_handler.get_unit(inp_wu)!r}."
                        ) from msg
                args = (units_handler.get_magnitude(inp_wu), *args[1:])
            elif units_mode == "coerce":
                raise MagpylibBadUserInput(
                    f"while magpylib.units.mode is set to {units_mode!r},"
                    f" input parameter {sig_name!r} is not unit-like ({inp!r}) "
                )
            res = func(*args, **kwargs)
            if out_to_units:
                res = to_Quantity(
                    res, units_handler.get_unit(inp_wu), units_handler=units_handler
                )
                if units_mode == "base":
                    res = to_Quantity(res, unit, units_handler=units_handler)
                elif units_mode == "downcast":
                    res = to_Quantity(res, unit, units_handler=units_handler)
                    res = units_handler.get_magnitude(res)
            elif units_mode in ("upcast", "coerce", "base"):
                units_global._in_use = True
                res = to_Quantity(res, unit, units_handler=units_handler)
            return res

        return wrapper

    return decorator


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
