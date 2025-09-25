"""Base class for all display representation code"""

# pylint: disable=cyclic-import
# pylint: disable=too-many-branches

import numpy as np
from scipy.spatial.transform import Rotation

from magpylib._src.display.display import show
from magpylib._src.display.traces_core import make_DefaultTrace

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
    """Provide display (``show()``) and textual representation (``__repr__``) helpers."""

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
        """Return list of lines describing the object properties.

        Parameters
        ----------
        exclude : None | str | Sequence[str], default ('style',)
            Property names to omit from the description.

        Returns
        -------
        list of str
            One line per entry ready to be joined with newlines.
        """
        if exclude is None:
            exclude = ()
        params = list(self._property_names_generator())
        lines = [f"{self!r}"]
        for key in list(dict.fromkeys(list(UNITS) + list(params))):
            k = key
            if not k.startswith("_") and k in params and k not in exclude:
                unit = UNITS.get(k)
                unit_str = f" {unit}" if unit else ""
                val = ""
                if k == "position":
                    val = getattr(self, "_position", None)
                    if isinstance(val, np.ndarray):
                        if val.shape[0] != 1:
                            lines.append(f"  • path length: {val.shape[0]}")
                            k = f"{k} (last)"
                        val = f"{val[-1]}"
                elif k == "orientation":
                    val = getattr(self, "_orientation", None)
                    if isinstance(val, Rotation):
                        val = val.as_rotvec(degrees=True)  # pylint: disable=no-member
                        if len(val) != 1:
                            k = f"{k} (last)"
                        val = f"{val[-1]}"
                elif k == "pixel":
                    val = getattr(self, "pixel", None)
                    if isinstance(val, np.ndarray):
                        px_shape = val.shape[:-1]
                        val_str = f"{int(np.prod(px_shape))}"
                        if val.ndim > 2:
                            val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                        val = val_str
                elif k == "status_disconnected_data":
                    val = getattr(self, k)
                    if val is not None:
                        val = f"{len(val)} part{'s'[: len(val) ^ 1]}"
                elif isinstance(getattr(self, k), list | tuple | np.ndarray):
                    val = np.array(getattr(self, k))
                    if np.prod(val.shape) > 4:
                        val = f"shape{val.shape}"
                else:
                    val = getattr(self, k)
                lines.append(f"  • {k}: {val}{unit_str}")
        return lines

    def describe(self, *, exclude=("style", "field_func"), return_string=False):
        """Return or print a formatted description of object properties.

        Parameters
        ----------
        exclude : str | Sequence[str], default ('style', 'field_func')
            Property names to omit from the description.
        return_string : bool, default False
            If ``True`` return the description string; if ``False`` print it and
            return ``None``.

        Returns
        -------
        str | None
            Description string if ``return_string=True`` else ``None``.
        """
        lines = self._get_description(exclude=exclude)
        output = "\n".join(lines)

        if return_string:
            return output

        print(output)  # noqa: T201
        return None

    def _repr_html_(self):
        """Rich HTML representation for notebooks and other frontends."""
        lines = self._get_description(exclude=("style", "field_func"))
        return f"""<pre>{"<br>".join(lines)}</pre>"""

    def __repr__(self) -> str:
        """Return concise string representation for terminals and logs."""
        name = getattr(self, "name", None)
        if name is None:
            style = getattr(self, "style", None)
            name = getattr(style, "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"
