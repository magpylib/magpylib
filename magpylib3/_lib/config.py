"""config"""


class config:
    EDGESIZE = 1e-14
    ITER_CYLINDER = 50

    @classmethod
    def reset(cls):
        cls.EDGESIZE = 1e-14
        cls.ITER_CYLINDER = 50