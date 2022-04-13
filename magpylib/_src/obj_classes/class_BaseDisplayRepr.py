"""BaseGeo class code
READY FOR V4
"""
import numpy as np

from magpylib._src.display.display import show

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
    _object_type = None

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
            if k in params and k not in exclude:
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
                    val_str = f"{int(np.product(px_shape))}"
                    if val.ndim > 2:
                        val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                    val = val_str
                else:
                    val = getattr(self, k)
                lines.append(f"  • {k}: {val} {unit_str}")
        return lines

    def describe(self, *, exclude=("style",), return_string=False):
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
        lines = self._get_description(exclude=("style",))
        return f"""<pre>{'<br>'.join(lines)}</pre>"""

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None and hasattr(self, "style"):
            name = getattr(getattr(self, "style"), "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"
