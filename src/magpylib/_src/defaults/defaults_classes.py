from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.defaults.defaults_utility import MagicProperties
from magpylib._src.defaults.defaults_utility import color_validator
from magpylib._src.defaults.defaults_utility import get_defaults_dict
from magpylib._src.defaults.defaults_utility import validate_property_class
from magpylib._src.style import DisplayStyle


class DefaultSettings(MagicProperties):
    """Library default settings.

    Parameters
    ----------
    display: dict or Display
        `Display` class containing display settings. `('backend', 'animation', 'colorsequence' ...)`
    """

    def __init__(
        self,
        display=None,
        **kwargs,
    ):
        super().__init__(
            display=display,
            **kwargs,
        )
        self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict(), _match_properties=False)
        return self

    @property
    def display(self):
        """`Display` class containing display settings.
        `('backend', 'animation', 'colorsequence')`"""
        return self._display

    @display.setter
    def display(self, val):
        self._display = validate_property_class(val, "display", Display, self)


class Display(MagicProperties):
    """
    Defines the properties for the plotting features.

    Properties
    ----------
    backend: str, default='matplotlib'
        Defines the plotting backend to be used by default, if not explicitly set in the `display`
        function (e.g. 'matplotlib', 'plotly').
        Supported backends are defined in magpylib.SUPPORTED_PLOTTING_BACKENDS

    colorsequence: iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color may be specified by
      - a hex string (e.g. '#ff0000')
      - an rgb/rgba string (e.g. 'rgb(255,0,0)')
      - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - a named CSS color

    animation: dict or Animation
        Defines the animation properties used by the `plotly` plotting backend when `animation=True`
        in the `show` function.

    autosizefactor: int, default=10
        Defines at which scale objects like sensors and dipoles are displayed.
        Specifically `object_size` = `canvas_size` / `AUTOSIZE_FACTOR`.

    styles: dict or DisplayStyle
        Base class containing display styling properties for all object families.
    """

    @property
    def backend(self):
        """plotting backend to be used by default, if not explicitly set in the `display`
        function (e.g. 'matplotlib', 'plotly').
        Supported backends are defined in magpylib.SUPPORTED_PLOTTING_BACKENDS"""
        return self._backend

    @backend.setter
    def backend(self, val):
        backends = [*SUPPORTED_PLOTTING_BACKENDS, "auto"]
        assert val is None or val in backends, (
            f"the `backend` property of {type(self).__name__} must be one of"
            f"{backends}"
            f" but received {repr(val)} instead"
        )
        self._backend = val

    @property
    def colorsequence(self):
        """An iterable of color values used to cycle trough for every object displayed.
          A color may be specified by
        - a hex string (e.g. '#ff0000')
        - an rgb/rgba string (e.g. 'rgb(255,0,0)')
        - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
        - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
        - a named CSS color"""
        return self._colorsequence

    @colorsequence.setter
    def colorsequence(self, val):
        if val is not None:
            name = type(self).__name__
            try:
                val = tuple(
                    color_validator(c, allow_None=False, parent_name=f"{name}")
                    for c in val
                )
            except TypeError as err:
                raise ValueError(
                    f"The `colorsequence` property of {name} must be an "
                    f"iterable of colors but received {val!r} instead"
                ) from err

        self._colorsequence = val

    @property
    def animation(self):
        """Animation properties used by the `plotly` plotting backend when `animation=True`
        in the `show` function."""
        return self._animation

    @animation.setter
    def animation(self, val):
        self._animation = validate_property_class(val, "animation", Animation, self)

    @property
    def autosizefactor(self):
        """Defines at which scale objects like sensors and dipoles are displayed.
        Specifically `object_size` = `canvas_size` / `AUTOSIZE_FACTOR`."""
        return self._autosizefactor

    @autosizefactor.setter
    def autosizefactor(self, val):
        assert val is None or isinstance(val, (int, float)) and val > 0, (
            f"the `autosizefactor` property of {type(self).__name__} must be a strictly positive"
            f" number but received {repr(val)} instead"
        )
        self._autosizefactor = val

    @property
    def style(self):
        """Base class containing display styling properties for all object families."""
        return self._style

    @style.setter
    def style(self, val):
        self._style = validate_property_class(val, "style", DisplayStyle, self)


class Animation(MagicProperties):
    """
    Defines the animation properties used by the `plotly` plotting backend when `animation=True`
    in the `display` function.

    Properties
    ----------
    fps: str, default=30
        Target number of frames to be displayed per second.

    maxfps: str, default=50
        Maximum number of frames to be displayed per second before downsampling kicks in.

    maxframes: int, default=200
        Maximum total number of frames to be displayed before downsampling kicks in.

    time: float, default=5
        Default animation time.

    slider: bool, default = True
        If True, an interactive slider will be displayed and stay in sync with the animation, will
        be hidden otherwise.

    output: str, default = None
        The path where to store the animation. Must end with `.mp4` or `.gif`. If only the suffix
        is used, the file is only store in a temporary folder and deleted after the animation is
        done.
    """

    @property
    def maxfps(self):
        """Maximum number of frames to be displayed per second before downsampling kicks in."""
        return self._maxfps

    @maxfps.setter
    def maxfps(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"The `maxfps` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead."
        )
        self._maxfps = val

    @property
    def fps(self):
        """Target number of frames to be displayed per second."""
        return self._fps

    @fps.setter
    def fps(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"The `fps` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead."
        )
        self._fps = val

    @property
    def maxframes(self):
        """Maximum total number of frames to be displayed before downsampling kicks in."""
        return self._maxframes

    @maxframes.setter
    def maxframes(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"The `maxframes` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead."
        )
        self._maxframes = val

    @property
    def time(self):
        """Default animation time."""
        return self._time

    @time.setter
    def time(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"The `time` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead."
        )
        self._time = val

    @property
    def slider(self):
        """show/hide slider"""
        return self._slider

    @slider.setter
    def slider(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `slider` property of {type(self).__name__} must be a either `True` or `False`"
            f" but received {repr(val)} instead."
        )
        self._slider = val

    @property
    def output(self):
        """Animation output type"""
        return self._output

    @output.setter
    def output(self, val):
        if val is not None:
            val = str(val)
            valid = val.endswith("mp4") or val.endswith("gif")
            assert val is None or valid, (
                f"The `output` property of {type(self).__name__} must be a either `mp4` or `gif` "
                "or a valid path ending with `.mp4` or `.gif`"
                f" but received {repr(val)} instead."
            )
        self._output = val


default_settings = DefaultSettings()
