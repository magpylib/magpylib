"""Custom class code"""

from magpylib._src.obj_classes.class_BaseExcitations import BaseSource


class CustomSource(BaseSource):
    """User-defined custom source.

    Can be used as ``sources`` input for magnetic field computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` global and local
    coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : None | Rotation, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    field_func : None | callable, default None
        Function for B- and H-field computation with the two positional arguments
        ``field`` and ``observers``. With ``field='B'`` or ``field='H'`` the function must
        return the B-field (T) or H-field (A/m) respectively. ``observers`` must accept
        an ``ndarray`` of shape ``(o, 3)`` in units (m) and the returned array must have
        shape ``(o, 3)``.
    style : None | dict, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    field_func : None | callable
        Same as constructor parameter ``field_func``.
    parent : Collection | None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Examples
    --------
    With version 4, ``CustomSource`` objects enable users to define their own source
    objects and embed them in the Magpylib object-oriented interface. In this example
    we create a source that generates a constant field and evaluate the field at the
    observer position ``(0.01, 0.01, 0.01)`` in units (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> def funcBH(field, observers):
    ...     return np.array([(0.01 if field == 'B' else 0.08, 0, 0)] * len(observers))
    >>> src = magpy.misc.CustomSource(field_func=funcBH)
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> print(H)
    [0.08 0.   0.  ]
    """

    _editable_field_func = True

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        field_func=None,
        style=None,
        **kwargs,
    ):
        # init inheritance
        super().__init__(position, orientation, field_func, style, **kwargs)

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.position
        return self._position
