import param

from magpylib._src.defaults.defaults_utility import color_validator
from magpylib._src.defaults.defaults_utility import get_defaults_dict
from magpylib._src.defaults.defaults_utility import MagicParameterized
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.style import DisplayStyle

# pylint: disable=missing-class-docstring


class Animation(MagicParameterized):
    fps = param.Integer(
        default=30,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Target number of frames to be displayed per second.""",
    )

    maxfps = param.Integer(
        default=50,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Maximum number of frames to be displayed per second before downsampling kicks in.""",
    )

    maxframes = param.Integer(
        default=200,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Maximum total number of frames to be displayed before downsampling kicks in.""",
    )

    time = param.Number(
        default=5,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Default animation time.""",
    )

    slider = param.Boolean(
        default=True,
        doc="""
        If True, an interactive slider will be displayed and stay in sync with the animation,
        will be hidden otherwise.""",
    )

    # either `mp4` or `gif` or ending with `.mp4` or `.gif`"
    output = param.String(
        doc="""Animation output type""",
        regex=r"^(mp4|gif|(.*\.(mp4|gif))?)$",
    )


class Display(MagicParameterized):
    def __setattr__(self, name, value):
        if name == "colorsequence":
            value = [
                color_validator(v, allow_None=False, parent_name="Colorsequence")
                for v in value
            ]
        return super().__setattr__(name, value)

    backend = param.Selector(
        default="matplotlib",
        objects=["auto", *SUPPORTED_PLOTTING_BACKENDS],
        doc="""
        Plotting backend to be used by default, if not explicitly set in the `display`
        function (e.g. 'matplotlib', 'plotly').
        Supported backends are defined in magpylib.SUPPORTED_PLOTTING_BACKENDS""",
    )

    colorsequence = param.List(
        doc="""
        An iterable of color values used to cycle trough for every object displayed.
        A color may be specified by
            - a hex string (e.g. '#ff0000')
            - an rgb/rgba string (e.g. 'rgb(255,0,0)')
            - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - a named CSS color""",
    )

    animation = param.ClassSelector(
        class_=Animation,
        default=Animation(),
        doc="""
        Animation properties used when `animation=True` in the `show` function,
        if applicaple to the chosen plotting backend.""",
    )

    autosizefactor = param.Number(
        default=10,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(5, 15),
        doc="""
        Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / AUTOSIZE_FACTOR""",
    )

    style = param.ClassSelector(
        class_=DisplayStyle,
        default=DisplayStyle(),
        doc="""class containing styling properties for any object family.""",
    )


class DefaultSettings(MagicParameterized):
    """Library default settings. All default values get reset at class instantiation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._declare_watchers()
        with param.parameterized.batch_call_watchers(self):
            self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict(), _match_properties=False)
        return self

    def _declare_watchers(self):
        props = get_defaults_dict(flatten=True, separator=".").keys()
        for prop in props:
            attrib_chain = prop.split(".")
            child = attrib_chain[-1]
            parent = self  # start with self to go through dot chain
            for attrib in attrib_chain[:-1]:
                parent = getattr(parent, attrib)
            parent.param.watch(self._set_to_defaults, parameter_names=[child])

    @staticmethod
    def _set_to_defaults(event):
        """Sets class defaults whenever magpylib defaults parameters instance are modifed."""
        event.obj.param[event.name].default = event.new

    display = param.ClassSelector(
        class_=Display,
        default=Display(),
        doc="""
        `Display` defaults-class containing display settings.
        `(e.g. 'backend', 'animation', 'colorsequence', ...)`""",
    )


default_settings = DefaultSettings()
