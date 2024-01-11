"""BaseGeo class code"""
# pylint: disable=cyclic-import
# pylint: disable=too-many-branches
import numpy as np

from magpylib._src.display.display import show
from magpylib._src.display.traces_core import make_DefaultTrace
from magpylib._src.units import is_Quantity

UNITS = {
    "parent": None,
    "position": "m",
    "orientation": "deg",
    "dimension": "m",
    "diameter": "m",
    "current": "A",
    "magnetization": "A/m",
    "polarization": "T",
    "moment": "A·m²",
}


class BaseDisplayRepr:
    """Provides the show and repr methods for all objects"""

    show = show
    get_trace = make_DefaultTrace

    def _property_names_generator(self):
        """returns a generator with class properties only"""
        return (
            attr
            for attr in dir(self)
            if isinstance(getattr(type(self), attr, None), property)
        )

    def _get_description(self, exclude=None):
        """Returns list of string describing the object properties.

        Parameters
        ----------
        exclude: bool, default=("style",)
            properties to be excluded in the description view.
        """
        if exclude is None:
            exclude = ()
        params = list(self._property_names_generator())
        lines = [f"{self!r}"]
        for key in list(dict.fromkeys(list(UNITS) + list(params))):
            if not key.startswith("_") and key in params and key not in exclude:
                val, val_str = getattr(self, key), None
                unit = UNITS.get(key, None)
                if is_Quantity(val):
                    # let pint take care of displaying units if available
                    unit = None
                if val is None:
                    val_str = None
                elif key == "position":
                    v = getattr(self, "_position")
                    if v.shape[0] != 1:
                        lines.append(f"  • path length: {v.shape[0]}")
                        key = f"{key} (last)"
                    val_str = f"{v[-1]}"
                elif key == "orientation":
                    v = getattr(self, "_orientation")
                    v = v.as_rotvec(degrees=True)
                    if len(v) != 1:
                        key = f"{key} (last)"
                    val_str = f"{v[-1]}"
                elif key == "pixel":
                    px_shape = val.shape[:-1]
                    val_str = f"{int(np.prod(px_shape))}"
                    if val.ndim > 2:
                        val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                elif key == "status_disconnected_data":
                    val_str = f"{len(val)} part{'s'[:len(val)^1]}"
                elif isinstance(val, (list, tuple, np.ndarray)):
                    v = np.array(val)
                    if np.prod(v.shape) > 4:
                        val_str = f"shape{v.shape}"
                unit_str = f" {unit}" if unit else ""
                val_str = val if val_str is None else val_str
                lines.append(f"  • {key}: {val_str}{unit_str}")
        return lines

    def describe(self, *, exclude=("style", "field_func"), return_string=False):
        """Returns a view of the object properties.

        Parameters
        ----------
        exclude: bool, default=("style",)
            Properties to be excluded in the description view.

        return_string: bool, default=`False`
            If `False` print description with stdout, if `True` return as string.
        """
        lines = self._get_description(exclude=exclude)
        output = "\n".join(lines)

        if return_string:
            return output

        print(output)
        return None

    def _repr_html_(self):
        lines = self._get_description(exclude=("style", "field_func"))
        return f"""<pre>{'<br>'.join(lines)}</pre>"""

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None and hasattr(self, "style"):
            name = getattr(getattr(self, "style"), "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"
