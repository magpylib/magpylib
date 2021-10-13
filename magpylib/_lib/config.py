"""Config class code"""

# ON INTERFACE
class Config:
    """
    Library default settings

    Parameters
    ----------
    CHECK_INPUTS: bool, default=True
        Check user input types, shapes at various stages and raise errors
        when they are not within designated parameters.

    EDGESIZE: float, default=1e-14
        getBand getH return 0 on edge, formulas often show singularities there and undefined forms.
        EDGESIZE defines how close to the edge 0 will be returned to avoid running into numerical
        instabilities.

    ITER_CYLINDER: int, default=50
        Cylinder with diametral magnetization uses Simpsons iterative formula
        to compute the integral. More iterations increase precision but slow
        down the computation.
        
    PLOTTING_BACKEND: str, default='matplotlib'
        One of 'matplotlib', 'plotly'. Defines the default plotting backend to fall to when not 
        set explicitly in the display function.

    COLOR_TRANSITION, float, default=0
        value between 0 and 1 sets the smoothness of the color transition from north to south pole 
        visualization. If set to a negative value (e.g. `-1`), the magnet polarity will be hidden
        and the object color will be uniform and managed by the plotting library.
    
    SOUTH_COLOR: 
        The property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    NORTH_COLOR: 
        The property is a color and may be specified as:
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
    _SUPPORTED_PLOTTING_BACKENDS = ('matplotlib', 'plotly')

    CHECK_INPUTS = True
    EDGESIZE = 1e-8
    ITER_CYLINDER = 50
    PLOTTING_BACKEND = 'matplotlib'
    COLOR_TRANSITION = 0.
    NORTH_COLOR = 'red' # 'magenta'
    SOUTH_COLOR = 'blue' # 'turquoise'
    COLOR_TRANSITION = 0
    
    @classmethod
    def reset(cls):
        """
        Reset Config to default values.

        Returns
        -------
        None: NoneType
        """
        cls.CHECK_INPUTS = True
        cls.EDGESIZE = 1e-14
        cls.ITER_CYLINDER = 50
        cls.PLOTTING_BACKEND = 'matplotlib'
        cls.COLOR_TRANSITION = 0.
        cls.NORTH_COLOR = 'red' # 'magenta'
        cls.SOUTH_COLOR = 'blue' # 'turquoise'
