"""Sensor class code"""

# pylint: disable=too-many-positional-arguments
# pylint: disable=arguments-differ

import numpy as np

from magpylib._src.display.traces_core import make_Sensor
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_BH import _getBH_level2
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.style import SensorStyle
from magpylib._src.utility import format_star_input


class Sensor(BaseGeo, BaseDisplayRepr):
    """Magnetic field sensor.

    Can be used as ``observers`` input for magnetic field computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the local object coordinates
    coincide with the global coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : None | Rotation, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    pixel : None | array-like, shape (3,) or (o1, o2, ..., 3), default None
        Sensor pixel (= sensing element) positions in local object coordinates
        (rotate with object) in units (m).
    handedness : {'right', 'left'}, default 'right'
        Object local coordinate system handedness. If ``'left'``, the x-axis is flipped.
    style : None | dict, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    pixel : None | ndarray, shape (3,) or (o1, o2, ..., 3)
        Same as constructor parameter ``pixel``.
    handedness : str
        Same as constructor parameter ``handedness``.
    parent : Collection | None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Examples
    --------
    ``Sensor`` objects are observers for magnetic field computation. In this example we
    compute the B-field in units (T) as seen by the sensor in the center of a circular
    current loop:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> sens = magpy.Sensor()
    >>> loop = magpy.current.Circle(current=1, diameter=0.01)
    >>> B = sens.getB(loop)
    >>> with np.printoptions(precision=3):
    ...     print(B*1000)
    [0.    0.    0.126]

    We rotate the sensor by 45 degrees and compute the field again:

    >>> sens.rotate_from_rotvec((45, 0, 0))
    Sensor(id=...)
    >>> B = sens.getB(loop)
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [0.000e+00 8.886e-05 8.886e-05]

    Finally, we set some sensor pixels and compute the field again:

    >>> sens.pixel = ((0, 0, 0), (0.001, 0, 0), (0.002, 0, 0))
    >>> B = sens.getB(loop)
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[0.000e+00 8.886e-05 8.886e-05]
     [0.000e+00 9.163e-05 9.163e-05]
     [0.000e+00 1.014e-04 1.014e-04]]
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

    # Properties
    @property
    def pixel(self):
        """Sensor pixel positions in local object coordinates in units (m)."""
        return self._pixel

    @pixel.setter
    def pixel(self, pix):
        """Set sensor pixel positions in local object coordinates.

        Parameters
        ----------
        pix : None | array-like, shape (3,) or (o1, o2, ..., 3)
            Sensor pixel positions in local object coordinates in units (m).
        """
        self._pixel = check_format_input_vector(
            pix,
            dims=range(1, 20),
            shape_m1=3,
            sig_name="pixel",
            sig_type="array-like (list, tuple, ndarray) with shape (o1, o2, ..., 3) or None",
            allow_None=True,
        )

    @property
    def handedness(self):
        """Object local coordinate system handedness."""
        return self._handedness

    @handedness.setter
    def handedness(self, val):
        """Set object local coordinate system handedness.

        Parameters
        ----------
        val : {'right', 'left'}
            If ``'left'``, the x-axis is flipped.
        """
        if val not in {"right", "left"}:
            msg = (
                f"Input handedness of {self} must be either 'right' or 'left'; "
                f"instead received {val!r}."
            )
            raise MagpylibBadUserInput(msg)
        self._handedness = val

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
            desc += f"{nop} pixel{'s'[: nop ^ 1]}"
        return desc

    # Methods
    def getB(
        self,
        *sources,
        sumup=False,
        squeeze=True,
        pixel_agg=None,
        output="ndarray",
        in_out="auto",
    ):
        """Return B-field (T) from s sources as seen by the sensor.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        *sources : Source | list
            Sources that generate the magnetic field. Can be a single source
            or a 1D list of s source objects.
        sumup : bool, default False
            If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
        squeeze : bool, default True
            If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
        pixel_agg : str | None, default None
            Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
            pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
        output : {'ndarray', 'dataframe'}, default 'ndarray'
            Output container type. ``'dataframe'`` returns a pandas DataFrame.
        in_out : {'auto', 'inside', 'outside'}, default 'auto'
            Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
            observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
            all outside (faster).

        Returns
        -------
        ndarray | DataFrame
            B-field (T) with squeezed shape (s, p, 1, o1, o2, ..., 3) where s is the number
            of sources, p is the path length, and o1, o2, ... are sensor pixel dimensions.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        B-field in T as seen by the sensor in the center of a circular current loop:

        >>> import numpy as np
        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Circle(current=1, diameter=0.01)
        >>> B = sens.getB(loop)
        >>> with np.printoptions(precision=3):
        ...     print(B*1000)
        [0.    0.    0.126]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45, 0, 0))
        Sensor(id=...)
        >>> B = sens.getB(loop)
        >>> with np.printoptions(precision=3):
        ...     print(B)
        [0.000e+00 8.886e-05 8.886e-05]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel = ((0, 0, 0), (0.001, 0, 0), (0.002, 0, 0))
        >>> B = sens.getB(loop)
        >>> with np.printoptions(precision=3):
        ...     print(B)
        [[0.000e+00 8.886e-05 8.886e-05]
         [0.000e+00 9.163e-05 9.163e-05]
         [0.000e+00 1.014e-04 1.014e-04]]
        """
        sources = format_star_input(sources)
        return _getBH_level2(
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
        """Return H-field (A/m) from s sources as seen by the sensor.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        *sources : Source | list
            Sources that generate the magnetic field. Can be a single source
            or a 1D list of s source objects.
        sumup : bool, default False
            If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
        squeeze : bool, default True
            If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
        pixel_agg : str | None, default None
            Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
            pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
        output : {'ndarray', 'dataframe'}, default 'ndarray'
            Output container type. ``'dataframe'`` returns a pandas DataFrame.
        in_out : {'auto', 'inside', 'outside'}, default 'auto'
            Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
            observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
            all outside (faster).

        Returns
        -------
        ndarray | DataFrame
            H-field (A/m) with squeezed shape (s, p, 1, o1, o2, ..., 3) where s is the number
            of sources, p is the path length, and o1, o2, ... are sensor pixel dimensions.

        Examples
        --------
        Sensors are observers for magnetic field computation. In this example we compute the
        B-field in T as seen by the sensor in the center of a circular current loop:

        >>> import numpy as np
        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> loop = magpy.current.Circle(current=1, diameter=0.01)
        >>> H = sens.getH(loop)
        >>> with np.printoptions(precision=3):
        ...     print(H)
        [  0.   0. 100.]

        Then we rotate the sensor by 45 degrees and compute the field again:

        >>> sens.rotate_from_rotvec((45, 0, 0))
        Sensor(id=...)
        >>> H = sens.getH(loop)
        >>> with np.printoptions(precision=3):
        ...     print(H)
        [ 0.    70.711 70.711]

        Finally we set some sensor pixels and compute the field again:

        >>> sens.pixel = ((0, 0, 0), (0.001, 0, 0), (0.002, 0, 0))
        >>> H = sens.getH(loop)
        >>> with np.printoptions(precision=3):
        ...     print(H)
        [[ 0.    70.711 70.711]
         [ 0.    72.915 72.915]
         [ 0.    80.704 80.704]]
        """
        sources = format_star_input(sources)
        return _getBH_level2(
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
        """Return magnetization (A/m) from s sources as seen by the sensor.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        *sources : Source | list
            Sources that generate the magnetic field. Can be a single source
            or a 1D list of s source objects.
        sumup : bool, default False
            If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
        squeeze : bool, default True
            If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
        pixel_agg : str | None, default None
            Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
            pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
        output : {'ndarray', 'dataframe'}, default 'ndarray'
            Output container type. ``'dataframe'`` returns a pandas DataFrame.
        in_out : {'auto', 'inside', 'outside'}, default 'auto'
            Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
            observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
            all outside (faster).

        Returns
        -------
        ndarray | DataFrame
            Magnetization (A/m) with squeezed shape (s, p, 1, o1, o2, ..., 3) where s is the number
            of sources, p is the path length, and o1, o2, ... are sensor pixel dimensions.

        Examples
        --------
        Test if there is magnetization at the location of the sensor.

        >>> import numpy as np
        >>> import magpylib as magpy
        >>> cube = magpy.magnet.Cuboid(
        ...     dimension=(10, 1, 1),
        ...     polarization=(1, 0, 0)
        ... )
        >>> sens = magpy.Sensor()
        >>> M = sens.getM(cube)
        >>> with np.printoptions(precision=0):
        ...    print(M)
        [795775.      0.      0.]
        """
        sources = format_star_input(sources)
        return _getBH_level2(
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
        """Return magnetic polarization (T) from s sources as seen by the sensor.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        *sources : Source | list
            Sources that generate the magnetic field. Can be a single source
            or a 1D list of s source objects.
        sumup : bool, default False
            If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
        squeeze : bool, default True
            If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
        pixel_agg : str | None, default None
            Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
            pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
        output : {'ndarray', 'dataframe'}, default 'ndarray'
            Output container type. ``'dataframe'`` returns a pandas DataFrame.
        in_out : {'auto', 'inside', 'outside'}, default 'auto'
            Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
            observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
            all outside (faster).

        Returns
        -------
        ndarray | DataFrame
            Magnetic polarization (T) with squeezed shape (s, p, 1, o1, o2, ..., 3) where s is the number
            of sources, p is the path length, and o1, o2, ... are sensor pixel dimensions.

        Examples
        --------
        Test if there is polarization at the location of the sensor.

        >>> import numpy as np
        >>> import magpylib as magpy
        >>> cube = magpy.magnet.Cuboid(
        ...     dimension=(10, 1, 1),
        ...     polarization=(1, 0, 0)
        ... )
        >>> sens = magpy.Sensor()
        >>> J = sens.getJ(cube)
        >>> with np.printoptions(precision=0):
        ...    print(J)
        [1. 0. 0.]
        """
        sources = format_star_input(sources)
        return _getBH_level2(
            sources,
            self,
            field="J",
            sumup=sumup,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def _get_centroid(self):
        """Centroid of object in units (m)."""
        if self.pixel is not None:
            pixel_mean = np.mean(self.pixel.reshape(-1, 3), axis=0)
            return self.position + pixel_mean
        return self.position
