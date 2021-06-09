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
    """

    CHECK_INPUTS = True
    EDGESIZE = 1e-14
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
