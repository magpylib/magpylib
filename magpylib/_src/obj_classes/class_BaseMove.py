"""BaseRotation class code"""
# pylint: disable=too-many-instance-attributes
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format)
from magpylib._src.exceptions import MagpylibBadUserInput


def check_start_type(start):
    """start input must be int or str"""
    if not (isinstance(start, int) or start == 'auto'):
        msg = 'start input must be int or str ("auto")'
        raise MagpylibBadUserInput(msg)


def check_absolute_type(inp):
    """absolute input must be bool"""
    if not isinstance(inp, bool):
        msg = 'absolute input must be boolean'
        raise MagpylibBadUserInput(msg)


def apply_move(target_object, displacement, start='auto', absolute=False):
    """
    Implementation of the move() functionality.

    target_object: object with position and orientation attributes
    displacement: displacement vector/path, array_like, shape (3,) or (n,3).
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    start: int, str, default='auto'
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the wole path is moved. With
        start='auto' and vector input the input is appended.
    absolute: bool, default=False
        If absolute=False then transformations are applied on to existing
        positions/orientations. If absolute=True position/orientation are
        set to input values.
    """
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-branches

    # check input types
    if Config.checkinputs:
        check_vector_type(displacement, "displacement")
        check_start_type(start)
        check_absolute_type(absolute)

    # displacement vector -> ndarray
    inpath = np.array(displacement, dtype=float)
    scalar_input = inpath.ndim==1

    # check input format
    if Config.checkinputs:
        check_path_format(inpath, "displacement")

    # load old path
    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    # path lengths
    lenop = len(ppath)
    lenip = 1 if scalar_input else len(inpath)

    # initialize paddings
    pad_before = 0
    pad_behind = 0

    # start='auto': apply to all if scalar, append if vector
    if start=='auto':
        if scalar_input:
            start=0
        else:
            start=lenop

    # numpy convention with negative start indices
    if start<0:
        start=lenop+start
        # if start smaller than -old_path_length: pad before
        if start<0:
            pad_before = -start # pylint: disable=invalid-unary-operand-type
            start=0

    # vector: if start+inpath extends beyond oldpath: pad behind and merge
    if start+lenip>lenop+pad_before:
        pad_behind = start+lenip - (lenop+pad_before)

    # avoid execution when there is no padding (cost~100ns)
    if pad_before+pad_behind:
        ppath = np.pad(ppath, ((pad_before, pad_behind), (0, 0)), "edge")
        opath = np.pad(opath, ((pad_before, pad_behind), (0, 0)), "edge")
        target_object.orientation = R.from_quat(opath)

    # set end-index
    if scalar_input:
        end = len(ppath)
    else:
        end = start+lenip

    # apply move operation
    if absolute:
        ppath[start:end] = inpath
    else:
        ppath[start:end] += inpath
    target_object.position = ppath

    return target_object


class BaseMove:
    """
    Inherit this class to provide move() methods.

    The apply_move function is applied to all target objects:
    - For Magpylib objects that inherit BaseMove and BaseGeo (e.g. Cuboid()),
      apply_move() is applied only to the object itself.
    - Collections inherit BaseGeo and have children with BaseGeo. In this case
    apply_move() is applied to the object itself, but also to the children.
    """

    def move(self, displacement, start='auto', absolute=False):
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

        # if Collection: apply to children
        if getattr(self, "_object_type", None) == "Collection":
            for obj in self.objects:
                apply_move(obj, displacement, start, absolute)
            return self

        # if BaseGeo apply to self
        return apply_move(self, displacement, start, absolute)
