"""BaseRotation class code"""
# pylint: disable=too-many-instance-attributes
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
    check_start_type,
    check_increment_type)
from magpylib._src.utility import adjust_start


class BaseMove:
    """Move method for Magpylib objects"""


    def __init__(self):
        # _target_object explanation in move() below
        self._target_object = self


    def move(self, displacement, start=-1, increment=False):
        """
        Translates the object by the input displacement (can be a path).

        This method uses vector addition to merge the input path given by displacement and the
        existing old path of an object. It keeps the old orientation. If the input path extends
        beyond the old path, the old path will be padded by its last entry before paths are
        added up.

        Parameters
        ----------
        displacement: array_like, shape (3,) or (N,3)
            Displacement vector shape=(3,) or path shape=(N,3) in units of [mm].

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input displacements are absolute.
            If `increment=True`, input displacements are interpreted as increments of each other.
            For example, an incremental input displacement of `[(2,0,0), (2,0,0), (2,0,0)]`
            corresponds to an absolute input displacement of `[(2,0,0), (4,0,0), (6,0,0)]`.

        Returns
        -------
        self: Magpylib object

        Examples
        --------

        With the ``move`` method Magpylib objects can be repositioned in the global coordinate
        system:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> sensor.move((1,1,1))
        >>> print(sensor.position)
        [1. 1. 1.]

        It is also a powerful tool for creating paths:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.move((1,1,1), start='append')
        >>> print(sensor.position)
        [[0. 0. 0.]
         [1. 1. 1.]]
        >>> sensor.move([(.1,.1,.1)]*2, start='append')
        >>> print(sensor.position)
        [[0.  0.  0. ]
         [1.  1.  1. ]
         [1.1 1.1 1.1]
         [1.1 1.1 1.1]]

        Complex paths can be generated with ease, by making use of the ``increment`` keyword
        and superposition of subsequent paths:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.move([(1,1,1)]*4, start='append', increment=True)
        >>> print(sensor.position)
        [[0. 0. 0.]
         [1. 1. 1.]
         [2. 2. 2.]
         [3. 3. 3.]
         [4. 4. 4.]]
        >>> sensor.move([(.1,.1,.1)]*5, start=2)
        >>> print(sensor.position)
        [[0.  0.  0. ]
         [1.  1.  1. ]
         [2.1 2.1 2.1]
         [3.1 3.1 3.1]
         [4.1 4.1 4.1]
         [4.1 4.1 4.1]
         [4.1 4.1 4.1]]

        """
        # pylint: disable=protected-access

        # Code explanation:
        #  - For Magpylib objects that inherit BaseMove and BaseGeo, move() is applied
        #    only to the object itself.
        #  - Collections inherit only BaseMove. In this case move() is only applied to
        #    the children of the Collection object.
        #  - Compounds are Collections that also inherit BaseGeo. In this case move()
        #    is applied to the object itself, but also to its children.

        if getattr(self, "_object_type", None) == "Collection":
            for obj in self.objects:         # pylint: disable=no-member
                self._target_object = obj
                self._move(displacement, start, increment)
            return self
        return self._move(displacement, start, increment)


    def _move(self, displacement, start=-1, increment=False):
        """
        move method implementation
        """
        # pylint: disable=protected-access
        # pylint: disable=attribute-defined-outside-init

        # check input types
        if Config.checkinputs:
            check_vector_type(displacement, "displacement")
            check_start_type(start)
            check_increment_type(increment)

        # displacement vector -> ndarray
        inpath = np.array(displacement, dtype=float)

        # check input format
        if Config.checkinputs:
            check_path_format(inpath, "displacement")

        # expand if input is shape (3,)
        if inpath.ndim == 1:
            inpath = np.expand_dims(inpath, 0)

        # load old path
        old_ppath = self._target_object._position
        old_opath = self._target_object._orientation.as_quat()
        lenop = len(old_ppath)
        lenin = len(inpath)

        # change start to positive values in [0, lenop]
        start = adjust_start(start, lenop)

        # incremental input -> absolute input
        if increment:
            for i, d in enumerate(inpath[:-1]):
                inpath[i + 1] = inpath[i + 1] + d

        end = start + lenin  # end position of new_path

        til = end - lenop
        if til > 0:  # case inpos extends beyond old_path -> tile up old_path
            old_ppath = np.pad(old_ppath, ((0, til), (0, 0)), "edge")
            old_opath = np.pad(old_opath, ((0, til), (0, 0)), "edge")
            self._target_object.orientation = R.from_quat(old_opath)

        # add new_ppath to old_ppath
        old_ppath[start:end] += inpath
        self._target_object.position = old_ppath

        return self._target_object
