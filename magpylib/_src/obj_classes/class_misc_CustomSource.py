"""Custom class code """
from magpylib._src.obj_classes.class_BaseExcitations import BaseSource


class CustomSource(BaseSource):
    """User-defined custom source.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` local object coordinates
    coincide with the global coordinate system.

    Parameters
    ----------
    field_func: callable, default=`None`
        The function for B- and H-field computation must have the two positional arguments
        `field` and `observers`. With `field='B'` or `field='H'` the B- or H-field in units
        of mT or kA/m must be returned respectively. The `observers` argument must
        accept numpy ndarray inputs of shape (n,3), in which case the returned fields must
        be numpy ndarrays of shape (n,3) themselves.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of mm. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    source: `CustomSource` object

    Examples
    --------
    With version 4 `CustomSource` objects enable users to define their own source
    objects, and to embedded them in the Magpylib object oriented interface. In this example
    we create a source that generates a constant field and evaluate the field at observer
    position (1,1,1) given in mm:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>>
    >>> funcBH = lambda field, observers: np.array([(100 if field=='B' else 80,0,0)]*len(observers))
    >>> src = magpy.misc.CustomSource(field_func=funcBH)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [80.  0.  0.]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'z')
    CustomSource(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    CustomSource(id=...)
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]]
    """

    _editable_field_func = True

    def __init__(
        self,
        field_func=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # init inheritance
        super().__init__(position, orientation, field_func, style, **kwargs)
