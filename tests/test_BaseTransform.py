import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
from magpylib._src.obj_classes.class_BaseRotate import apply_rotation
from magpylib._src.obj_classes.class_BaseMove import apply_move


def test_apply_move_v4_pt1():
    """ v4 path functionality tests """
    # pylint: disable=too-many-statements

    # SCALAR INPUT - ABSOLUTE=FALSE
    s = magpy.Sensor()

    # move object with start='auto'
    apply_move(s, (1,2,3))
    assert np.all(s.position == (1,2,3))

    # move object with start=0
    apply_move(s, (1,2,3), start=0)
    assert np.all(s.position == (2,4,6))

    # move object with start=-1
    apply_move(s, (1,2,3), start=-1)
    assert np.all(s.position == (3,6,9))

    # pad behind
    apply_move(s, (-1,-2,-3), start=1)
    assert np.all(s.position == [(3,6,9), (2,4,6)])

    # move whole path
    apply_move(s, (-1,-2,-3))
    assert np.all(s.position == [(2,4,6), (1,2,3)])

    # pad before
    apply_move(s, (-1,-2,-3), start=-3)
    assert np.all(s.position == [(1,2,3), (1,2,3), (0,0,0)])

    # move whole path starting in the middle
    apply_move(s, (1,2,3), start=1)
    assert np.all(s.position == [(1,2,3), (2,4,6), (1,2,3)])

    # move whole path starting in the middle with negative start
    apply_move(s, (1,2,3), start=-2)
    assert np.all(s.position == [(1,2,3), (3,6,9), (2,4,6)])


def test_apply_move_v4_pt2():
    """ v4 path functionality tests """
    # SCALAR INPUT - ABSOLUTE=True
    s = magpy.Sensor()

    # move object with start='auto'
    apply_move(s, (1,2,3), absolute=True)
    assert np.all(s.position == (1,2,3))

    # move object with start=0
    apply_move(s, (1,2,3), start=0, absolute=True)
    assert np.all(s.position == (1,2,3))

    # move object with start=-1
    apply_move(s, (2,3,4), start=-1, absolute=True)
    assert np.all(s.position == (2,3,4))

    # pad behind
    apply_move(s, (1,2,3), start=1, absolute=True)
    assert np.all(s.position == [(2,3,4), (1,2,3)])

    # move whole path
    apply_move(s, (2,2,2), absolute=True)
    assert np.all(s.position == [(2,2,2), (2,2,2)])

    # pad before
    apply_move(s, (1,2,3), start=-3, absolute=True)
    assert np.all(s.position == [(1,2,3), (1,2,3), (1,2,3)])

    # move whole path starting in the middle
    apply_move(s, (3,3,3), start=1, absolute=True)
    assert np.all(s.position == [(1,2,3), (3,3,3), (3,3,3)])

    # move whole path starting in the middle with negative start
    apply_move(s, (2,2,2), start=-2, absolute=True)
    assert np.all(s.position == [(1,2,3), (2,2,2), (2,2,2)])


def test_apply_move_v4_pt3():
    """ v4 path functionality tests """
    # VECTOR INPUT - ABSOLUTE=FALSE
    s = magpy.Sensor()

    # vector + start=0: simple append
    apply_move(s, [(1,2,3)])
    assert np.all(s.position == [(0,0,0), (1,2,3)])

    # vector + start in middle: merge
    apply_move(s, [(1,2,3)], start=1)
    assert np.all(s.position == [(0,0,0), (2,4,6)])

    # vector + start in middle: merge + pad behind
    apply_move(s, [(-1,-2,-3), (-2,-4,-6)], start=1)
    assert np.all(s.position == [(0,0,0), (1,2,3), (0,0,0)])

    # vector + start before: merge + pad before
    apply_move(s, [(1,2,3), (1,2,3)], start=-4)
    assert np.all(s.position == [(1,2,3), (1,2,3), (1,2,3), (0,0,0)])


def test_apply_move_v4_pt4():
    """ v4 path functionality tests """
    # VECTOR INPUT - ABSOLUTE=TRUE
    s = magpy.Sensor()

    # vector + start=0: simple append
    apply_move(s, [(1,2,3)], absolute=True)
    assert np.all(s.position == [(0,0,0), (1,2,3)])

    # vector + start in middle: merge
    apply_move(s, [(2,2,2)], start=1, absolute=True)
    assert np.all(s.position == [(0,0,0), (2,2,2)])

    # vector + start in middle: merge + pad behind
    apply_move(s, [(-1,-2,-3), (-2,-4,-6)], start=1, absolute=True)
    assert np.all(s.position == [(0,0,0), (-1,-2,-3), (-2,-4,-6)])

    # vector + start before: merge + pad before
    apply_move(s, [(1,2,3), (1,2,3)], start=-4, absolute=True)
    assert np.all(s.position == [(1,2,3), (1,2,3), (-1,-2,-3), (-2,-4,-6)])


def test_apply_rotation_v4_pt1():
    """ v4 path functionality tests """

    # SCALAR INPUT
    s = magpy.Sensor()
    # rotate object with start='auto'
    apply_rotation(s, R.from_rotvec((.1,.2,.3)))

    assert np.allclose(s.position, (0,0,0))
    assert np.allclose(s.orientation.as_rotvec(), (.1,.2,.3))

    # rotate object with start=0
    apply_rotation(s, R.from_rotvec((.1,.2,.3)), start=0)
    assert np.allclose(s.position, (0,0,0))
    assert np.allclose(s.orientation.as_rotvec(), (.2,.4,.6))

    # rotate object with start=-1
    apply_rotation(s, R.from_rotvec((-.2,-.4,-.6)), start=-1)
    assert np.allclose(s.position, (0,0,0))
    assert np.allclose(s.orientation.as_rotvec(), (0,0,0))

    # rotate object with anchor
    apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0))
    assert np.allclose(s.position, (1,-1,0))
    assert np.allclose(s.orientation.as_rotvec(), (0,0,np.pi/2))

    # pad behind
    apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=1)
    assert np.allclose(s.position, ((1,-1,0), (2,0,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,np.pi)))

    # rotate whole path
    apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0))
    assert np.allclose(s.position, ((2,0,0), (1,1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi), (0,0,-np.pi/2)))

    # pad before
    apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=-3)
    assert np.allclose(s.position, ((1,1,0), (1,1,0), (0,0,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,-np.pi/2), (0,0,0)))

    # rotate whole path starting in the middle
    apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=1)
    assert np.allclose(s.position, ((1,1,0), (0,0,0), (1,-1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,0), (0,0,np.pi/2)))

    # rotate whole path starting in the middle without anchor
    apply_rotation(s, R.from_rotvec((0,0,np.pi/4)), start=1)
    assert np.allclose(s.position, ((1,1,0), (0,0,0), (1,-1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,np.pi/4), (0,0,3*np.pi/4)))


def test_apply_rotation_v4_pt2():
    """ v4 path functionality tests """
    # VECTOR INPUT - ABSOLUTE=FALSE
    s = magpy.Sensor()

    # simple append start=auto behavior
    apply_rotation(s, R.from_rotvec(((0,0,np.pi/2),)), anchor=(1,0,0))
    assert np.allclose(s.position, ((0,0,0), (1,-1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,0), (0,0,np.pi/2)))

    # vector + start=0: simple merge
    apply_rotation(s, R.from_rotvec(((0,0,np.pi/2),)), anchor=(1,0,0), start=0)
    assert np.allclose(s.position, ((1,-1,0), (1,-1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,np.pi/2)))

    # vector + start in middle: merge + pad behind
    apply_rotation(s, R.from_rotvec(((0,0,np.pi), (0,0,np.pi))), anchor=(1,0,0), start=1)
    assert np.allclose(s.position, ((1,-1,0), (1,1,0), (1,1,0)))
    assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2)))


    # vector + start before: merge + pad before
    apply_rotation(s, R.from_rotvec(((0,0,0), (0,0,np.pi))), anchor=(1,0,0), start=-4)
    assert np.allclose(s.position, ((1,-1,0), (1,1,0), (1,1,0), (1,1,0)))
    assert np.allclose(
        s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2)))
