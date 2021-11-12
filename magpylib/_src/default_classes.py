from magpylib._src.default_utils import (
    MagicProperties,
    validate_property_class,
    color_validator,
    get_defaults_dict,
)
from magpylib._src.style import DisplayStyle

SUPPORTED_PLOTTING_BACKENDS = ("matplotlib", "plotly")

# ON INTERFACE
class DefaultConfig(MagicProperties):
    """Library default settings. All default values get set at class instantiation.

    Parameters
    ----------
    checkinputs: bool, default=True
        Check user input types, shapes at various stages and raise errors
        when they are not within designated parameters.

    edgesize: float, default=1e-8
        getB and getH return 0 on edge, formulas often show singularities there and undefined forms.
        EDGESIZE defines how close to the edge 0 will be returned to avoid running into numerical
        instabilities.

    itercylinder: int, default=50
        Cylinder with diametral magnetization uses Simpsons iterative formula
        to compute the integral. More iterations increase precision but slow
        down the computation.

    display: dict or Display
        `Display` class containing display settings. `('backend', 'animation', 'colorsequence' ...)`
    """

    def __init__(
        self,
        checkinputs=None,
        edgesize=None,
        itercylinder=None,
        display=None,
        **kwargs,
    ):
        super().__init__(
            checkinputs=checkinputs,
            edgesize=edgesize,
            itercylinder=itercylinder,
            display=display,
            **kwargs,
        )
        self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict(), _match_properties=False)
        return self

    @property
    def checkinputs(self):
        """Check user input types, shapes at various stages and raise errors
        when they are not within designated parameters."""
        return self._checkinputs

    @checkinputs.setter
    def checkinputs(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `checkinputs` property of {type(self).__name__} must be either `True` or `False`"
            f" but received {repr(val)} instead"
        )
        self._checkinputs = val

    @property
    def edgesize(self):
        """getB and getH return 0 on edge, formulas often show singularities there and undefined
        forms. EDGESIZE defines how close to the edge 0 will be returned to avoid running into
        numerical instabilities."""
        return self._edgesize

    @edgesize.setter
    def edgesize(self, val):
        assert val is None or isinstance(val, (int, float)) and val > 0, (
            f"the `edgesize` property of {type(self).__name__} must be a strictly positive number"
            f" but received {repr(val)} instead"
        )
        self._edgesize = val

    @property
    def itercylinder(self):
        """Cylinder with diametral magnetization uses Simpsons iterative formula
        to compute the integral. More iterations increase precision but slow
        down the computation."""
        return self._itercylinder

    @itercylinder.setter
    def itercylinder(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"the `itercylinder` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead"
        )
        self._itercylinder = val

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
    Defines the display properties for the plotting features

    Properties
    ----------
    backend: str, default='matplotlib'
        defines the plotting backend to be used by default, if not explicitly set in the `display`
        function. Can be one of `['matplotlib', 'plotly']`

    colorsequence:, iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    animation: dict or Animation
        defines the animation properties used by the `plotly` plotting backend when `path='animate'`
        in the `display` function.

    autosizefactor: int, default=10
        Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / AUTOSIZE_FACTOR

    styles: dict or DisplayStyle
        Base class containing display styling properties for all object families.
    """

    @property
    def backend(self):
        """plotting backend to be used by default, if not explicitly set in the `display`
        function. Can be one of `['matplotlib', 'plotly']`"""
        return self._backend

    @backend.setter
    def backend(self, val):
        assert val is None or val in SUPPORTED_PLOTTING_BACKENDS, (
            f"the `backend` property of {type(self).__name__} must be one of"
            f"{SUPPORTED_PLOTTING_BACKENDS}"
            f" but received {repr(val)} instead"
        )
        self._backend = val

    @property
    def colorsequence(self):
        """An iterable of color values used to cycle trough for every object displayed.
            A color and may be specified as:
        - A hex string (e.g. '#ff0000')
        - An rgb/rgba string (e.g. 'rgb(255,0,0)')
        - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
        - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
        - A named CSS color"""
        return self._colorsequence

    @colorsequence.setter
    def colorsequence(self, val):
        assert val is None or all(color_validator(c, allow_None=False) for c in val), (
            f"the `colorsequence` property of {type(self).__name__} must be one an iterable of"
            f"color sequences"
            f" but received {repr(val)} instead"
        )
        self._colorsequence = val

    @property
    def animation(self):
        """animation properties used by the `plotly` plotting backend when `path='animate'`
        in the `display` function."""
        return self._animation

    @animation.setter
    def animation(self, val):
        self._animation = validate_property_class(val, "animation", Animation, self)

    @property
    def autosizefactor(self):
        """Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / AUTOSIZE_FACTOR"""
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
    Defines the animation properties used by the `plotly` plotting backend when `path='animate'`
    in the `display` function.

    Properties
    ----------
    maxfps: str, default='matplotlib'
        Maximum number of frames to be displayed per second before downsampling kicks in.

    maxframes: int, default=200
        Maximum total number of frames to be displayed before downsampling kicks in.

    slider: bool, default = True
        if True, an interactive slider will be displayed and stay in sync with the animation, will
        be hidden otherwise.
    """

    @property
    def maxfps(self):
        """Maximum number of frames to be displayed per second before downsampling kicks in."""
        return self._maxfps

    @maxfps.setter
    def maxfps(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"the `maxfps` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead"
        )
        self._maxfps = val

    @property
    def maxframes(self):
        """Maximum total number of frames to be displayed before downsampling kicks in."""
        return self._maxframes

    @maxframes.setter
    def maxframes(self, val):
        assert val is None or isinstance(val, int) and val > 0, (
            f"the `maxframes` property of {type(self).__name__} must be a strictly positive"
            f" integer but received {repr(val)} instead"
        )
        self._maxframes = val

    @property
    def slider(self):
        """show/hide slider"""
        return self._slider

    @slider.setter
    def slider(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `slider` property of {type(self).__name__} must be a either `True` or `False`"
            f" but received {repr(val)} instead"
        )
        self._slider = val


default_settings = DefaultConfig()
