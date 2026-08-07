"""Library default settings classes, declared as a typed-property tree."""

from magpylib._src.defaults.defaults_utility import (
    get_defaults_dict,
    get_registered_backends,
)
from magpylib._src.defaults.property_tree import (
    Boolean,
    Choice,
    ColorSequence,
    Integer,
    Nested,
    Number,
    Property,
    PropertyNode,
)
from magpylib._src.style import DisplayStyle
from magpylib._src.utility import LENGTH_UNITS


class AnimationOutput(Property):
    """A string ending with 'mp4' or 'gif', coerced via str()."""

    def validate(self, obj, value):
        value = str(value)
        if not value.endswith(("mp4", "gif")):
            self._fail(obj, value, "either 'mp4' or 'gif'")
        return value


class Animation(PropertyNode):
    """Defines the animation properties used by the `plotly` plotting backend
    when `animation=True` in the `show` function.

    Parameters
    ----------
    fps: int, default=None
        Target number of frames to be displayed per second.
    maxfps: int, default=None
        Maximum number of frames to be displayed per second before downsampling
        kicks in.
    maxframes: int, default=None
        Maximum total number of frames to be displayed before downsampling
        kicks in.
    time: int, default=None
        Default animation time.
    slider: bool, default=None
        If True, an interactive slider will be displayed and stay in sync with
        the animation, will be hidden otherwise.
    output: str, default=None
        The path where to store the animation. Must end with `.mp4` or `.gif`.
        If only the suffix is used, the file is only stored in a temporary
        folder and deleted after the animation is done.
    """

    fps = Integer(
        exclusive_minimum=0, doc="Target number of frames displayed per second."
    )
    maxfps = Integer(
        exclusive_minimum=0,
        doc="Maximum number of frames per second before downsampling kicks in.",
    )
    maxframes = Integer(
        exclusive_minimum=0,
        doc="Maximum total number of frames before downsampling kicks in.",
    )
    time = Integer(exclusive_minimum=0, doc="Default animation time.")
    slider = Boolean(doc="Show/hide interactive animation slider.")
    output = AnimationOutput(doc="Animation output type ('mp4' or 'gif').")


class Units(PropertyNode):
    """Defines the units a scene is drawn in.

    Parameters
    ----------
    length: str, default=None
        Length unit the scene is drawn in (e.g. 'mm'), overridable per call
        with `show(units_length=...)`. With 'auto' the unit is inferred from
        the extent of the displayed system, which makes the coordinates handed
        to the backend depend on the scene as a whole; pin it to a fixed unit
        when they have to depend only on the objects themselves.
    """

    length = Choice(
        "auto",
        *LENGTH_UNITS,
        doc="Length unit the scene is drawn in ('auto' infers it from the scene).",
    )


class Display(PropertyNode):
    """Defines the properties for the plotting features.

    Parameters
    ----------
    backend: str, default=None
        Plotting backend to be used by default, if not explicitly set in the
        `display` function (e.g. 'matplotlib', 'plotly'). The built-in
        backends are listed in magpylib.SUPPORTED_PLOTTING_BACKENDS; any
        backend registered at runtime is accepted as well.
    colorsequence: iterable, default=None
        An iterable of color values used to cycle through for every object
        displayed. A color may be specified by a hex string, an rgb string,
        or a named CSS color.
    animation: dict or `Animation` object, default=None
        Animation properties used by the `plotly` plotting backend when
        `animation=True` in the `show` function.
    autosizefactor: float, default=None
        Defines at which scale objects like sensors and dipoles are displayed.
        Specifically `object_size` = `canvas_size` / `AUTOSIZE_FACTOR`.
    units: dict or `Units` object, default=None
        Units the scene is drawn in. Reachable as `units.length` or, via magic
        underscore, as `units_length` -- matching the `show()` argument.
    style: dict or `DisplayStyle` object, default=None
        Display styling properties for all object families.
    """

    backend = Choice(
        lambda: (*get_registered_backends(), "auto"),
        doc="Default plotting backend.",
    )
    colorsequence = ColorSequence(
        doc="Colors cycled through for every object displayed."
    )
    animation = Nested(Animation, doc="Animation properties (plotly backend).")
    autosizefactor = Number(
        exclusive_minimum=0, doc="Display scale of autosized objects."
    )
    units = Nested(Units, doc="Units the scene is drawn in.")
    style = Nested(DisplayStyle, doc="Display styling properties for all families.")


class DefaultSettings(PropertyNode):
    """Library default settings.

    Parameters
    ----------
    display: dict or `Display` object
        `Display` class containing display settings
        ('backend', 'animation', 'colorsequence', ...).
    """

    display = Nested(Display, doc="Display settings.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values."""
        self.update(get_defaults_dict(), _match_properties=False)
        return self


default_settings = DefaultSettings()
