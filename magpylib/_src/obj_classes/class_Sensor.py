"""Sensor class code
DOCSTRINGS V4 READY
"""
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.utility import format_star_input


class Sensor(BaseGeo, BaseDisplayRepr):
    """Magnetic field sensor.

    Can be used as `observers` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` local object coordinates
    coincide with the global coordinate system.

    A sensor is made up of pixel (sensing elements) where the magnetic field is evaluated.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    pixel: array_like, shape (3,) or (n1,n2,...,3), default=`(0,0,0)`
        Sensor pixel (=sensing elements) positions in the local object coordinates
        (rotate with object), in units of [mm].

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
    observer: `Sensor` object

    Examples
    --------
    `Sensor` objects are observers for magnetic field computation. In this example we compute the
    B-field in units of [mT] as seen by the sensor in the center of a circular current loop:

    >>> import magpylib as magpy
    >>> sens = magpy.Sensor()
    >>> loop = magpy.current.Loop(current=1, diameter=1)
    >>> B = sens.getB(loop)
    >>> print(B)
    [0.         0.         1.25663706]

    We rotate the sensor by 45 degrees and compute the field again:

    >>> sens.rotate_from_rotvec((45,0,0))
    Sensor(id=...)
    >>> B = sens.getB(loop)
    >>> print(B)
    [0.         0.88857659 0.88857659]

    Finally we set some sensor pixels and compute the field again:

    >>> sens.pixel=((0,0,0), (.1,0,0), (.2,0,0))
    >>> B = sens.getB(loop)
    >>> print(B)
    [[0.         0.88857659 0.88857659]
     [0.         0.916274   0.916274  ]
     [0.         1.01415383 1.01415383]]
    """

    def __init__(
        self,
        position=(0, 0, 0),
        pixel=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.pixel = pixel
        self._object_type = "Sensor"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)

    # property getters and setters
    @property
    def pixel(self):
        """Sensor pixel (=sensing elements) positions in the local object coordinates
        (rotate with object), in units of [mm].
        """
        return self._pixel

    @pixel.setter
    def pixel(self, pix):
        """Set sensor pixel positions in the local sensor coordinates.
        Must be an array_like, float compatible with shape (..., 3)
        """
        self._pixel = check_format_input_vector(
            pix,
            dims=range(1, 20),
            shape_m1=3,
            sig_name="pixel",
            sig_type="array_like (list, tuple, ndarray) with shape (n1, n2, ..., 3)",
        )

    def getB(self, *sources, sumup=False, squeeze=True, pixel_agg=None):
        """Compute the B-field in units of [mT] as seen by the sensor.

        Parameters
        ----------
        sources: source and collection objects or 1D list thereof
            Sources that generate the magnetic field. Can be a single source (or collection)
            or a 1D list of l source and/or collection objects.

        sumup: bool, default=`False`
            If `True`, the fields of all sources are summed up.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        Returns
        -------
        B-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3)
            B-field of each source (l) at each path position (m) and each sensor pixel
            position (n1,n2,...) in units of [mT]. Paths of objects that are shorter than
            m will be considered as static beyond their end.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        B-field [mT] as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Loop(current=1, diameter=1)
        >>> B = sens.getB(loop)
        >>> print(B)
        [0.         0.         1.25663706]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45,0,0))
        Sensor(id=...)
        >>> B = sens.getB(loop)
        >>> print(B)
        [0.         0.88857659 0.88857659]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel=((0,0,0), (.1,0,0), (.2,0,0))
        >>> B = sens.getB(loop)
        >>> print(B)
        [[0.         0.88857659 0.88857659]
         [0.         0.916274   0.916274  ]
         [0.         1.01415383 1.01415383]]
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources, self, sumup=sumup, squeeze=squeeze, pixel_agg=pixel_agg, field="B"
        )

    def getH(self, *sources, sumup=False, squeeze=True, pixel_agg=None):
        """Compute the H-field in units of [kA/m] as seen by the sensor.

        Parameters
        ----------
        sources: source and collection objects or 1D list thereof
            Sources that generate the magnetic field. Can be a single source (or collection)
            or a 1D list of l source and/or collection objects.

        sumup: bool, default=`False`
            If `True`, the fields of all sources are summed up.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        Returns
        -------
        H-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3)
            H-field of each source (l) at each path position (m) and each sensor pixel
            position (n1,n2,...) in units of [kA/m]. Paths of objects that are shorter than
            m will be considered as static beyond their end.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        H-field [kA/m] as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Loop(current=1, diameter=1)
        >>> H = sens.getH(loop)
        >>> print(H)
        [0. 0. 1.]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45,0,0))
        Sensor(id=...)
        >>> H = sens.getH(loop)
        >>> print(H)
        [0.         0.70710678 0.70710678]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel=((0,0,0), (.1,0,0), (.2,0,0))
        >>> H = sens.getH(loop)
        >>> print(H)
        [[0.         0.70710678 0.70710678]
         [0.         0.72914768 0.72914768]
         [0.         0.80703798 0.80703798]]
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources, self, sumup=sumup, squeeze=squeeze, pixel_agg=pixel_agg, field="H"
        )
