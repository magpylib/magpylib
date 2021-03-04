"""config"""

class Config:
    """ configuration class for quick access to computation settings
    """
    EDGESIZE = 1e-14
    ITER_CYLINDER = 50

    @classmethod
    def reset(cls):
        """ reset configuration to default values
        """
        cls.EDGESIZE = 1e-14
        cls.ITER_CYLINDER = 50
