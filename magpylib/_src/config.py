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

    CHECK_INPUTS = True
    EDGESIZE = 1e-8
    ITER_CYLINDER = 50

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
