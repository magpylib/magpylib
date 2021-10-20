"""Config class code"""


from typing import Any
import warnings

_DEFAULTS = dict(
    _SUPPORTED_PLOTTING_BACKENDS=("matplotlib", "plotly"),
    CHECK_INPUTS=True,
    EDGESIZE=1e-8,
    ITER_CYLINDER=50,
    PLOTTING_BACKEND="matplotlib",
    NORTH_COLOR="rgb(231,17,17)",  # 'red'
    MIDDLE_COLOR="rgb(221,221,221)",  # 'grey'
    SOUTH_COLOR="rgb(0,176,80)",  # 'green'
    COLOR_TRANSITION=0,
    PIXEL_COLOR="grey",
    COLOR_DISCRETE_SEQUENCE=(
        "#2E91E5",
        "#E15F99",
        "#1CA71C",
        "#FB0D0D",
        "#DA16FF",
        "#222A2A",
        "#B68100",
        "#750D86",
        "#EB663B",
        "#511CFB",
        "#00A08B",
        "#FB00D1",
        "#FC0080",
        "#B2828D",
        "#6C7C32",
        "#778AAE",
        "#862A16",
        "#A777F1",
        "#620042",
        "#1616A7",
        "#DA60CA",
        "#6C4516",
        "#0D2A63",
        "#AF0038",
    ),
)


# ON INTERFACE
class Config:
    """Library default settings

    Parameters
    ----------
    CHECK_INPUTS: bool, default=True
        Check user input types, shapes at various stages and raise errors
        when they are not within designated parameters.

    EDGESIZE: float, default=True
        getBand getH return 0 on edge, formulas often show singularities there and undefined forms.
        EDGESIZE defines how close to the edge 0 will be returned to avoid running into numerical
        instabilities.

    ITER_CYLINDER: int, default=True
        Cylinder with diametral magnetization uses Simpsons iterative formula
        to compute the integral. More iterations increase precision but slow
        down the computation.

    PLOTTING_BACKEND: str, default=True
        One of 'matplotlib', 'plotly'. Defines the default plotting backend to fall to when not
        set explicitly in the display function.

    COLOR_TRANSITION, float, default=0
        A value between 0 and 1. Sets the smoothness of the color transitions from north
        to south pole visualization.

    SOUTH_COLOR, str, default=rgb(0,176,80)
        The property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    MIDDLE_COLOR, str, default=rgb(221,221,221)
        The property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    NORTH_COLOR, str, default=rgb(231,17,17)
        The property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    PIXEL_COLOR, str, default=rgb(231,17,17)
        The property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    COLOR_DISCRETE_SEQUENCE, iterable, default=
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

    Examples
    --------
    Compute the field very close to a line current:

    >>> import magpylib as magpy
    >>> current = magpy.current.Line(current=1, vertices=[(-.1,0,0),(.1,0,0)])
    >>> B_close = current.getB((0,0,1e-11))
    >>> print(B_close)
    [ 0.e+00 -2.e+10  0.e+00]

    Change the edgesize setting so that the position is now inside of the cut-off region

    >>> magpy.Config.EDGESIZE=1e-10
    >>> B_close = current.getB((0,0,1e-11))
    >>> print(B_close)
    [0. 0. 0.]

    Reset the Config to original values:

    >>> magpy.Config.reset()
    >>> B_close = current.getB((0,0,1e-11))
    >>> print(B_close)
    [ 0.e+00 -2.e+10  0.e+00]
    """

    def __init__(self):
        self.reset()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "PLOTTING_BACKEND":
            backends = _DEFAULTS["_SUPPORTED_PLOTTING_BACKENDS"]
            assert (
                value in backends
            ), f"`{value}` is not a valid plotting backend, \n supported backends: {backends}"
        super().__setattr__(name, value)

    def __getattr__(self, name):
        ''' will only get called for undefined attributes '''
        warnings.warn(f'No member "{name}" contained in settings config.')

    @classmethod
    def reset(cls, args=None):
        """
        Reset Config to default values.
        Parameters
        ----------
        args: iterable of strings. If not set, all defaults will be reset.

        Returns
        -------
        None: NoneType
        """
        if args is None:
            args = _DEFAULTS.keys()
        for k in args:
            setattr(cls, k, _DEFAULTS[k])

Config.reset()
