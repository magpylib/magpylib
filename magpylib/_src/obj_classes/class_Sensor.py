# pylint: disable=too-many-positional-arguments

"""Sensor class code"""

import numpy as np

from magpylib._src.display.traces_core import make_Sensor
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_wrap_BH import getBH_level2
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.style import SensorStyle
from magpylib._src.utility import format_star_input


class Sensor(BaseGeo, BaseDisplayRepr):
    """Magnetic field sensor.

    Can be used as `observers` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` local object coordinates
    coincide with the global coordinate system.

    A sensor is made up of pixel (sensing elements / positions) where the magnetic
    field is evaluated.

    SI units are used for all inputs and outputs.

    Parameters
    ----------

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    pixel: array_like, shape (3,) or (n1,n2,...,3), default=`(0,0,0)`
        Sensor pixel (=sensing elements) positions in the local object coordinates
        (rotate with object), in units of m.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    handedness: {"right", "left"}
        Object local coordinate system handedness. If "left", the x-axis is flipped.

    Returns
    -------
    observer: `Sensor` object

    Examples
    --------
    `Sensor` objects are observers for magnetic field computation. In this example we compute the
    B-field in units of T as seen by the sensor in the center of a circular current loop:

    >>> import magpylib as magpy
    >>> sens = magpy.Sensor()
    >>> loop = magpy.current.Circle(current=1, diameter=0.01)
    >>> B = sens.getB(loop)
    >>> print(B)
    [0.         0.         0.00012566]

    We rotate the sensor by 45 degrees and compute the field again:

    >>> sens.rotate_from_rotvec((45,0,0))
    Sensor(id=...)
    >>> B = sens.getB(loop)
    >>> print(B)
    [0.00000000e+00 8.88576588e-05 8.88576588e-05]

    Finally we set some sensor pixels and compute the field again:

    >>> sens.pixel=((0,0,0), (.001,0,0), (.002,0,0))
    >>> B = sens.getB(loop)
    >>> print(B)
    [[0.00000000e+00 8.88576588e-05 8.88576588e-05]
     [0.00000000e+00 9.16274003e-05 9.16274003e-05]
     [0.00000000e+00 1.01415384e-04 1.01415384e-04]]
    """

    _style_class = SensorStyle
    _autosize = True
    get_trace = make_Sensor

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        pixel=None,
        handedness="right",
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.pixel = pixel
        self.handedness = handedness

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)

    # property getters and setters
    @property
    def pixel(self):
        """Sensor pixel (=sensing elements) positions in the local object coordinates
        (rotate with object), in units of m.
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
            sig_type="array_like (list, tuple, ndarray) with shape (n1, n2, ..., 3) or None",
            allow_None=True,
        )

    @property
    def handedness(self):
        """Sensor handedness in the local object coordinates."""
        return self._handedness

    @handedness.setter
    def handedness(self, val):
        """Set Sensor handedness in the local object coordinates."""
        if val not in {"right", "left"}:
            raise MagpylibBadUserInput(
                "Sensor `handedness` must be either `'right'` or `'left'`"
            )
        self._handedness = val

    def getB(
        self,
        *sources,
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        output="ndarray",
        in_out="auto",
    ):
        """Compute the B-field in units of T as seen by the sensor.

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

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        B-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3) or DataFrame
            B-field of each source (index l) at each path position (index m) and each sensor pixel
            position (indices n1,n2,...) in units of T. Paths of objects that are shorter than
            index m are considered as static beyond their end.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        B-field in T as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Circle(current=1, diameter=.01)
        >>> B = sens.getB(loop)
        >>> print(B)
        [0.         0.         0.00012566]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45,0,0))
        Sensor(id=...)
        >>> B = sens.getB(loop)
        >>> print(B)
        [0.00000000e+00 8.88576588e-05 8.88576588e-05]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel=((0,0,0), (.001,0,0), (.002,0,0))
        >>> B = sens.getB(loop)
        >>> print(B)
        [[0.00000000e+00 8.88576588e-05 8.88576588e-05]
         [0.00000000e+00 9.16274003e-05 9.16274003e-05]
         [0.00000000e+00 1.01415384e-04 1.01415384e-04]]
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources,
            self,
            field="B",
            sumup=sumup,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getH(
        self,
        *sources,
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        output="ndarray",
        in_out="auto",
    ):
        """Compute the H-field in units of A/m as seen by the sensor.

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

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        H-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3) or DataFrame
            H-field of each source (index l) at each path position (index m) and each sensor pixel
            position (indices n1,n2,...) in units of A/m. Paths of objects that are shorter than
            index m are considered as static beyond their end.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        B-field in T as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Circle(current=1, diameter=.01)
        >>> H = sens.getH(loop)
        >>> print(H)
        [  0.   0. 100.]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45,0,0))
        Sensor(id=...)
        >>> H = sens.getH(loop)
        >>> print(H)
        [ 0.         70.71067812 70.71067812]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel=((0,0,0), (.001,0,0), (.002,0,0))
        >>> H = sens.getH(loop)
        >>> print(H)
        [[ 0.         70.71067812 70.71067812]
         [ 0.         72.9147684  72.9147684 ]
         [ 0.         80.7037979  80.7037979 ]]
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources,
            self,
            field="H",
            sumup=sumup,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getM(
        self,
        *sources,
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        output="ndarray",
        in_out="auto",
    ):
        """Compute the M-field in units of A/m as seen by the sensor.

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

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        M-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3) or DataFrame
            M-field of each source (index l) at each path position (index m) and each sensor pixel
            position (indices n1,n2,...) in units of A/m. Paths of objects that are shorter than
            index m are considered as static beyond their end.
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources,
            self,
            field="M",
            sumup=sumup,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getJ(
        self,
        *sources,
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        output="ndarray",
        in_out="auto",
    ):
        """Compute the J-field in units of T as seen by the sensor.

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

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        J-field: ndarray, shape squeeze(l, m, n1, n2, ..., 3) or DataFrame
            J-field of each source (index l) at each path position (index m) and each sensor pixel
            position (indices n1,n2,...) in units of T. Paths of objects that are shorter than
            index m are considered as static beyond their end.
        """
        sources = format_star_input(sources)
        return getBH_level2(
            sources,
            self,
            field="J",
            sumup=sumup,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        pix = self.pixel
        desc = ""
        if pix is not None:
            px_shape = pix.shape[:-1]
            nop = int(np.prod(px_shape))
            if pix.ndim > 2:
                desc += f"{'x'.join(str(p) for p in px_shape)}="
            desc += f"{nop} pixel{'s'[:nop^1]}"
        return desc
