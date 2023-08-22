"""BaseTransform class code"""
# pylint: disable=protected-access
import numbers

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.input_checks import check_degree_type
from magpylib._src.input_checks import check_format_input_anchor
from magpylib._src.input_checks import check_format_input_angle
from magpylib._src.input_checks import check_format_input_axis
from magpylib._src.input_checks import check_format_input_orientation
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.input_checks import check_start_type


def multi_anchor_behavior(anchor, inrotQ, rotation):
    """define behavior of rotation with given anchor
    if one is longer than the other pad up other
    """
    len_inrotQ = 0 if inrotQ.ndim == 1 else inrotQ.shape[0]
    len_anchor = 0 if anchor.ndim == 1 else anchor.shape[0]

    if len_inrotQ > len_anchor:
        if len_anchor == 0:
            anchor = np.reshape(anchor, (1, 3))
            len_anchor = 1
        anchor = np.pad(anchor, ((0, len_inrotQ - len_anchor), (0, 0)), "edge")
    elif len_inrotQ < len_anchor:
        if len_inrotQ == 0:
            inrotQ = np.reshape(inrotQ, (1, 4))
            len_inrotQ = 1
        inrotQ = np.pad(inrotQ, ((0, len_anchor - len_inrotQ), (0, 0)), "edge")
        rotation = R.from_quat(inrotQ)

    return anchor, inrotQ, rotation


def path_padding_param(scalar_input: bool, lenop: int, lenip: int, start: int):
    """compute path padding parameters

    Example: with start>0 and input path exceeds old_path
        old_path:            |abcdefg|
        input_path:              |xzyuvwrst|
        -> padded_old_path:  |abcdefggggggg|

    Parameters:
    -----------
    scalar_input: True if rotation input is scalar, else False
    lenop: length of old_path
    lenip: length of input_path
    start: start index

    Returns:
    --------
    padding: (pad_before, pad_behind)
        how much the old_path must be padded before
    start: modified start value
    """
    # initialize paddings
    pad_before = 0
    pad_behind = 0

    # start='auto': apply to all if scalar, append if vector
    if start == "auto":
        if scalar_input:
            start = 0
        else:
            start = lenop

    # numpy convention with negative start indices
    if start < 0:
        start = lenop + start
        # if start smaller than -old_path_length: pad before
        if start < 0:
            pad_before = -start  # pylint: disable=invalid-unary-operand-type
            start = 0

    # vector: if start+inpath extends beyond oldpath: pad behind
    if start + lenip > lenop + pad_before:
        pad_behind = start + lenip - (lenop + pad_before)

    if pad_before + pad_behind > 0:
        return (pad_before, pad_behind), start
    return [], start


def path_padding(inpath, start, target_object):
    """pad path of target_object and compute start- and end-index for apply_move()
    and apply_rotation() functions below so that ppath[start:end] = X... can be
    applied.

    Parameters
    ----------
    inpath: user input as np.ndarray
    start: start index
    target_object: Magpylib object with position and orientation attributes

    Returns
    -------
    ppath: padded target_object position path
    opath: padded target_object orientation path
    start: modified start idex
    end: end index
    padded: True if padding was necessary, else False
    """
    # scalar or vector input
    scalar_input = inpath.ndim == 1

    # load old path
    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    lenip = 1 if scalar_input else len(inpath)

    # pad old path depending on input
    padding, start = path_padding_param(scalar_input, len(ppath), lenip, start)
    if padding:
        ppath = np.pad(ppath, (padding, (0, 0)), "edge")
        opath = np.pad(opath, (padding, (0, 0)), "edge")

    # set end-index
    end = len(ppath) if scalar_input else start + lenip

    return ppath, opath, start, end, bool(padding)


def apply_move(target_object, displacement, start="auto"):
    """Implementation of the move() functionality.

    Parameters
    ----------
    target_object: object with position and orientation attributes
    displacement: displacement vector/path, array_like, shape (3,) or (n,3).
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    start: int, str, default=`'auto'`
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the whole path is moved. With
        start='auto' and vector input the input is appended.

    Returns
    -------
    target_object
    """
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-branches

    # check and format inputs
    inpath = check_format_input_vector(
        displacement,
        dims=(1, 2),
        shape_m1=3,
        sig_name="displacement",
        sig_type="array_like (list, tuple, ndarray) with shape (3,) or (n,3)",
    )
    check_start_type(start)

    # pad target_object path and compute start and end-index for rotation application
    ppath, opath, start, end, padded = path_padding(inpath, start, target_object)
    if padded:
        target_object._orientation = R.from_quat(opath)

    # apply move operation
    ppath[start:end] += inpath
    target_object._position = ppath

    return target_object


def apply_rotation(
    target_object, rotation: R, anchor=None, start="auto", parent_path=None
):
    """Implementation of the rotate() functionality.

    Parameters
    ----------
    target_object: object with position and orientation attributes
    rotation: a scipy Rotation object
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    anchor: array_like shape (3,)
        Rotation anchor
    start: int, str, default=`'auto'`
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the wole path is moved. With
        start='auto' and vector input the input is appended.
    parent_path=None if there is no parent else parent._position

    Returns
    -------
    target_object
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # check and format inputs
    rotation, inrotQ = check_format_input_orientation(rotation)
    anchor = check_format_input_anchor(anchor)
    check_start_type(start)

    # when an anchor is given
    if anchor is not None:
        # apply multi-anchor behavior
        anchor, inrotQ, rotation = multi_anchor_behavior(anchor, inrotQ, rotation)

    # pad target_object path and compute start and end-index for rotation application
    ppath, opath, newstart, end, _ = path_padding(inrotQ, start, target_object)

    # compute anchor when dealing with Compound rotation (target_object is a child
    #   that rotates about its parent). This happens when a rotation with anchor=None
    #   is applied to a child in a Collection. In this case the anchor must be set to
    #   the parent_path.
    if anchor is None and parent_path is not None:
        # target anchor length
        len_anchor = end - newstart
        # pad up parent_path if input requires it
        padding, start = path_padding_param(
            inrotQ.ndim == 1, parent_path.shape[0], len_anchor, start
        )
        if padding:
            parent_path = np.pad(parent_path, (padding, (0, 0)), "edge")
        # slice anchor from padded parent_path
        anchor = parent_path[start : start + len_anchor]

    # position change when there is an anchor
    if anchor is not None:
        ppath[newstart:end] -= anchor
        ppath[newstart:end] = rotation.apply(ppath[newstart:end])
        ppath[newstart:end] += anchor

    # set new rotation
    oldrot = R.from_quat(opath[newstart:end])
    opath[newstart:end] = (rotation * oldrot).as_quat()

    # store new position and orientation
    # pylint: disable=attribute-defined-outside-init
    target_object._orientation = R.from_quat(opath)
    target_object._position = ppath

    return target_object


class BaseTransform:
    """Inherit this class to provide rotation() and move() methods."""

    def move(self, displacement, start="auto"):
        """Move object by the displacement input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        displacement: array_like, shape (3,) or (n,3)
            Displacement vector in units of mm.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        Returns
        -------
        self: Magpylib object

        Examples
        --------

        Move objects around with scalar input:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,1,1))
        >>> print(sens.position)
        [1. 1. 1.]
        >>> sens.move((1,1,1))
        Sensor(id=...)
        >>> print(sens.position)
        [2. 2. 2.]

        Create len>1 object paths with vector input:

        >>> sens.move([(1,1,1),(2,2,2),(3,3,3)])
        Sensor(id=...)
        >>> print(sens.position)
        [[2. 2. 2.]
         [3. 3. 3.]
         [4. 4. 4.]
         [5. 5. 5.]]

        Apply operations starting with a designated path index:

        >>> sens.move((0,0,2), start=2)
        Sensor(id=...)
        >>> print(sens.position)
        [[2. 2. 2.]
         [3. 3. 3.]
         [4. 4. 6.]
         [5. 5. 7.]]
        """

        # Idea: An operation applied to a Collection is individually
        #    applied to its BaseGeo and to each child.

        for child in getattr(self, "children", []):
            child.move(displacement, start=start)

        apply_move(self, displacement, start=start)

        return self

    def _rotate(self, rotation: R, anchor=None, start="auto", parent_path=None):
        """Rotate object about a given anchor.

        See `rotate` docstring for other parameters.

        Parameters
        ----------
        parent_path: if there is no parent else parent._position
            needs to be transmitted from the top level for nested collections, hence using a
            private `_rotate` method to do so.

        """
        # Idea: An operation applied to a Collection is individually
        #    applied to its BaseGeo and to each child.
        #  -> this automatically generates the rotate-Compound behavior

        # pylint: disable=no-member
        for child in getattr(self, "children", []):
            ppth = self._position if parent_path is None else parent_path
            child._rotate(rotation, anchor=anchor, start=start, parent_path=ppth)

        apply_rotation(
            self, rotation, anchor=anchor, start=start, parent_path=parent_path
        )
        return self

    def rotate(self, rotation: R, anchor=None, start="auto"):
        """Rotate object about a given anchor.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        rotation: `None` or scipy `Rotation` object
            Rotation to be applied to the object. The scipy `Rotation` input can
            be scalar or vector type (see terminology above). `None` input is interpreted
            as unit rotation.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        Returns
        -------
        self: Magpylib object

        Examples
        --------

        Rotate an object about the origin:

        >>> from scipy.spatial.transform import Rotation as R
        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate(R.from_euler('z', 45, degrees=True), anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 45.]

        Rotate the object about itself:

        >>> sens.rotate(R.from_euler('z', 45, degrees=True))
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Create a rotation path by rotating in several steps about an anchor:

        >>> sens.rotate(R.from_euler('z', (15,30,45), degrees=True), anchor=(0,0,0))
        Sensor(id=...)
        >>> print(sens.position)
        [[ 7.07106781e-01  7.07106781e-01  0.00000000e+00]
         [ 5.00000000e-01  8.66025404e-01  0.00000000e+00]
         [ 2.58819045e-01  9.65925826e-01  0.00000000e+00]
         [-2.22044605e-16  1.00000000e+00  0.00000000e+00]]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.  90.]
         [  0.   0. 105.]
         [  0.   0. 120.]
         [  0.   0. 135.]]
        """

        return self._rotate(rotation=rotation, anchor=anchor, start=start)

    def rotate_from_angax(self, angle, axis, anchor=None, start="auto", degrees=True):
        """Rotates object using angle-axis input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        angle: int, float or array_like with shape (n,)
            Angle(s) of rotation in units of deg (by default).

        axis: str or array_like, shape (3,)
            The direction of the axis of rotation. Input can be a vector of shape (3,)
            or a string 'x', 'y' or 'z' to denote respective directions.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        degrees: bool, default=`True`
            Interpret input in units of deg or rad.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_angax(45, axis='z', anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 45.]

        Rotate the object about itself:

        >>> sens.rotate_from_angax(45, axis=(0,0,1))
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Create a rotation path by rotating in several steps about an anchor:

        >>> sens.rotate_from_angax((15,30,45), axis='z', anchor=(0,0,0))
        Sensor(id=...)
        >>> print(sens.position)
        [[ 7.07106781e-01  7.07106781e-01  0.00000000e+00]
         [ 5.00000000e-01  8.66025404e-01  0.00000000e+00]
         [ 2.58819045e-01  9.65925826e-01  0.00000000e+00]
         [-2.22044605e-16  1.00000000e+00  0.00000000e+00]]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.  90.]
         [  0.   0. 105.]
         [  0.   0. 120.]
         [  0.   0. 135.]]
        """
        # check/format inputs
        angle = check_format_input_angle(angle)
        axis = check_format_input_axis(axis)
        check_start_type(start)
        check_degree_type(degrees)

        # degree to rad
        if degrees:
            angle = angle / 180 * np.pi

        # create rotation vector from angle/axis input
        if isinstance(angle, numbers.Number):
            angle = np.ones(3) * angle
        else:
            angle = np.tile(angle, (3, 1)).T
        axis = axis / np.linalg.norm(axis) * angle

        # forwards rotation object to rotate method
        rot = R.from_rotvec(axis)
        return self.rotate(rot, anchor, start)

    def rotate_from_rotvec(self, rotvec, anchor=None, start="auto", degrees=True):
        """Rotates object using rotation vector input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        rotvec : array_like, shape (n,3) or (3,)
            Rotation input. Rotation vector direction is the rotation axis, vector length is
            the rotation angle in units of rad.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        degrees: bool, default=`True`
            Interpret input in units of deg or rad.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_rotvec((0,0,45), anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 45.]

        Rotate the object about itself:

        >>> sens.rotate_from_rotvec((0,0,45))
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Create a rotation path by rotating in several steps about an anchor:

        >>> sens.rotate_from_rotvec([(0,0,15), (0,0,30), (0,0,45)], anchor=(0,0,0))
        Sensor(id=...)
        >>> print(sens.position)
        [[ 7.07106781e-01  7.07106781e-01  0.00000000e+00]
         [ 5.00000000e-01  8.66025404e-01  0.00000000e+00]
         [ 2.58819045e-01  9.65925826e-01  0.00000000e+00]
         [-2.22044605e-16  1.00000000e+00  0.00000000e+00]]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.  90.]
         [  0.   0. 105.]
         [  0.   0. 120.]
         [  0.   0. 135.]]
        """
        rot = R.from_rotvec(rotvec, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_euler(self, angle, seq, anchor=None, start="auto", degrees=True):
        """Rotates object using Euler angle input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        angle: int, float or array_like with shape (n,)
            Angle(s) of rotation in units of deg (by default).

        seq : string
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        degrees: bool, default=`True`
            Interpret input in units of deg or rad.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_euler(45, 'z', anchor=0)
        Sensor...
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 45.]

        Rotate the object about itself:

        >>> sens.rotate_from_euler(45, 'z')
        Sensor(id=...)
        >>> print(sens.position)
        [0.70710678 0.70710678 0.        ]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Create a rotation path by rotating in several steps about an anchor:

        >>> sens.rotate_from_euler((15,30,45), 'z', anchor=(0,0,0))
        Sensor(id=...)
        >>> print(sens.position)
        [[ 7.07106781e-01  7.07106781e-01  0.00000000e+00]
         [ 5.00000000e-01  8.66025404e-01  0.00000000e+00]
         [ 2.58819045e-01  9.65925826e-01  0.00000000e+00]
         [-2.22044605e-16  1.00000000e+00  0.00000000e+00]]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.  90.]
         [  0.   0. 105.]
         [  0.   0. 120.]
         [  0.   0. 135.]]
        """
        rot = R.from_euler(seq, angle, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_matrix(self, matrix, anchor=None, start="auto"):
        """Rotates object using matrix input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].


        Parameters
        ----------
        matrix : array_like, shape (n,3,3) or (3,3)
            Rotation input. See scipy.spatial.transform.Rotation for details.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_matrix([(0,-1,0),(1,0,0),(0,0,1)], anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [0. 1. 0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Rotate the object about itself:

        >>> sens.rotate_from_matrix([(0,-1,0),(1,0,0),(0,0,1)])
        Sensor(id=...)
        >>> print(sens.position)
        [0. 1. 0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [  0.   0. 180.]
        """
        rot = R.from_matrix(matrix)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_mrp(self, mrp, anchor=None, start="auto"):
        """Rotates object using Modified Rodrigues Parameters (MRPs) input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        mrp : array_like, shape (n,3) or (3,)
            Rotation input. See scipy Rotation package for details on Modified Rodrigues
            Parameters (MRPs).

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_mrp((0,0,1), anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [-1.  0.  0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [  0.   0. 180.]

        Rotate the object about itself:

        >>> sens.rotate_from_matrix([(0,-1,0),(1,0,0),(0,0,1)])
        Sensor(id=...)
        >>> print(sens.position)
        [-1.  0.  0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [  0.   0. -90.]
        """
        rot = R.from_mrp(mrp)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_quat(self, quat, anchor=None, start="auto"):
        """Rotates object using quaternion input.

        Terminology for move/rotate methods:

        - `path` refers to `position` and `orientation` of an object.
        - When an input is just a single operation (e.g. one displacement vector or one angle)
          we call it 'scalar input'. When it is an array_like of multiple scalars, we refer to
          it as 'vector input'.

        General move/rotate behavior:

        - Scalar input is applied to the whole object path, starting with path index `start`.
        - Vector input of length n applies the individual n operations to n object path
          entries, starting with path index `start`.
        - When an input extends beyond the object path, the object path will be padded by its
          edge-entries before the operation is applied.
        - By default (`start='auto'`) the index is set to `start=0` for scalar input [=move
          whole object path], and to `start=len(object path)` for vector input [=append to
          existing object path].

        Parameters
        ----------
        quat : array_like, shape (n,4) or (4,)
            Rotation input in quaternion form.

        anchor: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
            The axis of rotation passes through the anchor point given in units of mm.
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin `(0,0,0)`.

        start: int or str, default=`'auto'`
            Starting index when applying operations. See 'General move/rotate behavior' above
            for details.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Rotate an object about the origin:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor(position=(1,0,0))
        >>> sens.rotate_from_quat((0,0,1,1), anchor=0)
        Sensor(id=...)
        >>> print(sens.position)
        [0. 1. 0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 90.]

        Rotate the object about itself:

        >>> sens.rotate_from_quat((0,0,1,1))
        Sensor(id=...)
        >>> print(sens.position)
        [0. 1. 0.]
        >>> print(sens.orientation.as_euler('xyz', degrees=True))
        [  0.   0. 180.]
        """
        rot = R.from_quat(quat)
        return self.rotate(rot, anchor=anchor, start=start)
