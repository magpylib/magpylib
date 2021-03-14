"""Config class code"""

# ON INTERFACE
class Config:
    """
    Read and change library default settings:

    Settings
    --------
    EDGESIZE: float
        getB returns 0 on edge, formulas often show singularities there,
        EDGESIZE defines how close to the edge 0 will be returned

    ITER_CYLINDER: int
        Cylinder with diametral magnetization uses Simpsons iterative formula
        to compute the integral. More iterations increase precision but slow
        down the computation.

    Methods
    -------
    reset():
        Reset Config to default values.
    """

    EDGESIZE = 1e-14
    ITER_CYLINDER = 50
    @classmethod
    def reset(cls):
        """ Reset Config to default values.
        """
        cls.EDGESIZE = 1e-14
        cls.ITER_CYLINDER = 50
