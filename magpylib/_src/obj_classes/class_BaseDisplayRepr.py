"""BaseGeo class code"""
# pylint: disable=cyclic-import
# pylint: disable=too-many-branches
import numpy as np

from magpylib._src.display.display import show
from magpylib._src.display.traces_core import make_DefaultTrace

UNITS = {
    "parent": None,
    "position": "mm",
    "orientation": "degrees",
    "dimension": "mm",
    "diameter": "mm",
    "current": "A",
    "magnetization": "mT",
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
        for k in list(dict.fromkeys(list(UNITS) + list(params))):
            if not k.startswith("_") and k in params and k not in exclude:
                unit = UNITS.get(k, None)
                unit_str = f"{unit}" if unit else ""
                if k == "position":
                    val = getattr(self, "_position")
                    if val.shape[0] != 1:
                        lines.append(f"  • path length: {val.shape[0]}")
                        k = f"{k} (last)"
                    val = f"{val[-1]}"
                elif k == "orientation":
                    val = getattr(self, "_orientation")
                    val = val.as_rotvec(degrees=True)
                    if len(val) != 1:
                        k = f"{k} (last)"
                    val = f"{val[-1]}"
                elif k == "pixel":
                    val = getattr(self, "pixel")
                    px_shape = val.shape[:-1]
                    val_str = f"{int(np.prod(px_shape))}"
                    if val.ndim > 2:
                        val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                    val = val_str
                elif k == "status_disconnected_data":
                    val = getattr(self, k)
                    if val is not None:
                        val = f"{len(val)} part{'s'[:len(val)^1]}"
                elif isinstance(getattr(self, k), (list, tuple, np.ndarray)):
                    val = np.array(getattr(self, k))
                    if np.prod(val.shape) > 4:
                        val = f"shape{val.shape}"
                else:
                    val = getattr(self, k)
                lines.append(f"  • {k}: {val} {unit_str}")
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
