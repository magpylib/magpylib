import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.obj_classes.class_BaseTransform import apply_move
from magpylib._src.obj_classes.class_BaseTransform import apply_rotation


@pytest.mark.parametrize(
    (
        "description",
        "old_position",
        "displacement",
        "new_position",
        "start",
    ),
    [
        # SCALAR INPUT
        ("01_ with start='auto'", (0, 0, 0), (1, 2, 3), (1, 2, 3), "auto"),
        ("02_ with start=0", (1, 2, 3), (1, 2, 3), (2, 4, 6), 0),
        ("03_ with start=-1", (2, 4, 6), (1, 2, 3), (3, 6, 9), -1),
        ("04_ pad behind", (3, 6, 9), (-1, -2, -3), [(3, 6, 9), (2, 4, 6)], 1),
        (
            "05_ whole path",
            [(3, 6, 9), (2, 4, 6)],
            (-1, -2, -3),
            [(2, 4, 6), (1, 2, 3)],
            "auto",
        ),
        (
            "06_ pad before",
            [(2, 4, 6), (1, 2, 3)],
            (-1, -2, -3),
            [(1, 2, 3), (1, 2, 3), (0, 0, 0)],
            -3,
        ),
        (
            "07_ whole path starting in the middle",
            [(1, 2, 3), (1, 2, 3), (0, 0, 0)],
            (1, 2, 3),
            [(1, 2, 3), (2, 4, 6), (1, 2, 3)],
            1,
        ),
        (
            "08_ whole path starting in the middle with negative start",
            [(1, 2, 3), (2, 4, 6), (1, 2, 3)],
            (1, 2, 3),
            [(1, 2, 3), (3, 6, 9), (2, 4, 6)],
            -2,
        ),
        # VECTOR INPUT
        (
            "17_ vector + start=0: simple append",
            (0, 0, 0),
            [(1, 2, 3)],
            [(0, 0, 0), (1, 2, 3)],
            "auto",
        ),
        (
            "18_ vector + start in middle: merge",
            [(0, 0, 0), (1, 2, 3)],
            [(1, 2, 3)],
            [(0, 0, 0), (2, 4, 6)],
            1,
        ),
        (
            "19_ vector + start in middle: merge + pad behind",
            [(0, 0, 0), (2, 4, 6)],
            [(-1, -2, -3), (-2, -4, -6)],
            [(0, 0, 0), (1, 2, 3), (0, 0, 0)],
            1,
        ),
        (
            "20_ vector + start before: merge + pad before",
            [(0, 0, 0), (1, 2, 3), (0, 0, 0)],
            [(1, 2, 3), (1, 2, 3)],
            [(1, 2, 3), (1, 2, 3), (1, 2, 3), (0, 0, 0)],
            -4,
        ),
    ],
)
def test_apply_move(description, old_position, displacement, new_position, start):
    """v4 path functionality tests"""
    print(description)
    s = magpy.Sensor(position=old_position)
    apply_move(s, displacement, start=start)
    assert np.all(s.position == np.array(new_position))


@pytest.mark.parametrize(
    (
        "description",
        "old_position",
        "new_position",
        "old_orientation_rotvec",
        "rotvec_to_apply",
        "new_orientation_rotvec",
        "start",
        "anchor",
    ),
    [
        # SCALAR INPUT
        (
            "01_ with start='auto'",
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3),
            "auto",
            None,
        ),
        (
            "02_ with start=0",
            (0, 0, 0),
            (0, 0, 0),
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3),
            (0.2, 0.4, 0.6),
            0,
            None,
        ),
        (
            "03_ with start=-1",
            (0, 0, 0),
            (0, 0, 0),
            (0.2, 0.4, 0.6),
            (-0.2, -0.4, -0.6),
            (0, 0, 0),
            -1,
            None,
        ),
        (
            "04_  with anchor",
            (0, 0, 0),
            (1, -1, 0),
            (0, 0, 0),
            (0, 0, np.pi / 2),
            (0, 0, np.pi / 2),
            -1,
            (1, 0, 0),
        ),
        (
            "05_  pad behind",
            (1, -1, 0),
            [(1, -1, 0), (2, 0, 0)],
            (0, 0, np.pi / 2),
            (0, 0, np.pi / 2),
            [(0, 0, np.pi / 2), (0, 0, np.pi)],
            1,
            (1, 0, 0),
        ),
        (
            "06_  whole path",
            [(1, -1, 0), (2, 0, 0)],
            [(2, 0, 0), (1, 1, 0)],
            [(0, 0, np.pi / 2), (0, 0, np.pi)],
            (0, 0, np.pi / 2),
            [(0, 0, np.pi), (0, 0, -np.pi / 2)],
            "auto",
            (1, 0, 0),
        ),
        (
            "07_ pad before",
            [(2, 0, 0), (1, 1, 0)],
            [(1, 1, 0), (1, 1, 0), (0, 0, 0)],
            [(0, 0, np.pi), (0, 0, -np.pi / 2)],
            (0, 0, np.pi / 2),
            [(0, 0, -np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
            -3,
            (1, 0, 0),
        ),
        (
            "08_ whole path starting in the middle",
            [(1, 1, 0), (1, 1, 0), (0, 0, 0)],
            [(1, 1, 0), (0, 0, 0), (1, -1, 0)],
            [(0, 0, -np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
            (0, 0, np.pi / 2),
            [(0, 0, -np.pi / 2), (0, 0, 0), (0, 0, np.pi / 2)],
            1,
            (1, 0, 0),
        ),
        (
            "09_ whole path starting in the middle without anchor",
            [(1, 1, 0), (0, 0, 0), (1, -1, 0)],
            [(1, 1, 0), (0, 0, 0), (1, -1, 0)],
            [(0, 0, -np.pi / 2), (0, 0, 0), (0, 0, np.pi / 2)],
            ((0, 0, np.pi / 4)),
            [(0, 0, -np.pi / 2), (0, 0, np.pi / 4), (0, 0, 3 * np.pi / 4)],
            1,
            None,
        ),
        # VECTOR INPUT
        (
            "11_ simple append start=auto behavior",
            (0, 0, 0),
            [(0, 0, 0), (1, -1, 0)],
            (0, 0, 0),
            [(0, 0, np.pi / 2)],
            [(0, 0, 0), (0, 0, np.pi / 2)],
            "auto",
            (1, 0, 0),
        ),
        (
            "12_ vector + start=0: simple merge",
            [(0, 0, 0), (1, -1, 0)],
            [(1, -1, 0), (1, -1, 0)],
            [(0, 0, 0), (0, 0, np.pi / 2)],
            [(0, 0, np.pi / 2)],
            [(0, 0, np.pi / 2), (0, 0, np.pi / 2)],
            0,
            (1, 0, 0),
        ),
        (
            "13_ vector + start in middle: merge + pad behind",
            [(1, -1, 0), (1, -1, 0)],
            [(1, -1, 0), (1, 1, 0), (1, 1, 0)],
            [(0, 0, np.pi / 2), (0, 0, np.pi / 2)],
            [(0, 0, np.pi), (0, 0, np.pi)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, -np.pi / 2)],
            1,
            (1, 0, 0),
        ),
        (
            "14_ vector + start before: merge + pad before",
            [(1, -1, 0), (1, 1, 0), (1, 1, 0)],
            [(1, -1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, -np.pi / 2)],
            [(0, 0, 0), (0, 0, np.pi)],
            [
                (0, 0, np.pi / 2),
                (0, 0, -np.pi / 2),
                (0, 0, -np.pi / 2),
                (0, 0, -np.pi / 2),
            ],
            -4,
            (1, 0, 0),
        ),
    ],
)
def test_apply_rotation(
    description,
    old_position,
    new_position,
    old_orientation_rotvec,
    rotvec_to_apply,
    new_orientation_rotvec,
    start,
    anchor,
):
    """v4 path functionality tests"""
    print(description)
    s = magpy.Sensor(
        position=old_position, orientation=R.from_rotvec(old_orientation_rotvec)
    )
    apply_rotation(s, R.from_rotvec(rotvec_to_apply), start=start, anchor=anchor)
    assert np.allclose(s.position, np.array(new_position))
    assert np.allclose(
        s.orientation.as_matrix(), R.from_rotvec(new_orientation_rotvec).as_matrix()
    )


# def test_apply_move_v4_pt1():
#     """ v4 path functionality tests """
#     # pylint: disable=too-many-statements

#     # SCALAR INPUT - ABSOLUTE=FALSE
#     s = magpy.Sensor()

#     # move object with start='auto'
#     apply_move(s, (1,2,3))
#     assert np.all(s.position == (1,2,3))

#     # move object with start=0
#     apply_move(s, (1,2,3), start=0)
#     assert np.all(s.position == (2,4,6))

#     # move object with start=-1
#     apply_move(s, (1,2,3), start=-1)
#     assert np.all(s.position == (3,6,9))

#     # pad behind
#     apply_move(s, (-1,-2,-3), start=1)
#     assert np.all(s.position == [(3,6,9), (2,4,6)])

#     # move whole path
#     apply_move(s, (-1,-2,-3))
#     assert np.all(s.position == [(2,4,6), (1,2,3)])

#     # pad before
#     apply_move(s, (-1,-2,-3), start=-3)
#     assert np.all(s.position == [(1,2,3), (1,2,3), (0,0,0)])

#     # move whole path starting in the middle
#     apply_move(s, (1,2,3), start=1)
#     assert np.all(s.position == [(1,2,3), (2,4,6), (1,2,3)])

#     # move whole path starting in the middle with negative start
#     apply_move(s, (1,2,3), start=-2)
#     assert np.all(s.position == [(1,2,3), (3,6,9), (2,4,6)])


# def test_apply_move_v4_pt3():
#     """ v4 path functionality tests """
#     # VECTOR INPUT - ABSOLUTE=FALSE
#     s = magpy.Sensor()

#     # vector + start=0: simple append
#     apply_move(s, [(1,2,3)])
#     assert np.all(s.position == [(0,0,0), (1,2,3)])

#     # vector + start in middle: merge
#     apply_move(s, [(1,2,3)], start=1)
#     assert np.all(s.position == [(0,0,0), (2,4,6)])

#     # vector + start in middle: merge + pad behind
#     apply_move(s, [(-1,-2,-3), (-2,-4,-6)], start=1)
#     assert np.all(s.position == [(0,0,0), (1,2,3), (0,0,0)])

#     # vector + start before: merge + pad before
#     apply_move(s, [(1,2,3), (1,2,3)], start=-4)
#     assert np.all(s.position == [(1,2,3), (1,2,3), (1,2,3), (0,0,0)])


# def test_apply_rotation_v4_pt1():
#     """ v4 path functionality tests """

#     # SCALAR INPUT
#     s = magpy.Sensor()
#     # rotate object with start='auto'
#     apply_rotation(s, R.from_rotvec((.1,.2,.3)))

#     assert np.allclose(s.position, (0,0,0))
#     assert np.allclose(s.orientation.as_rotvec(), (.1,.2,.3))

#     # rotate object with start=0
#     apply_rotation(s, R.from_rotvec((.1,.2,.3)), start=0)
#     assert np.allclose(s.position, (0,0,0))
#     assert np.allclose(s.orientation.as_rotvec(), (.2,.4,.6))

#     # rotate object with start=-1
#     apply_rotation(s, R.from_rotvec((-.2,-.4,-.6)), start=-1)
#     assert np.allclose(s.position, (0,0,0))
#     assert np.allclose(s.orientation.as_rotvec(), (0,0,0))

#     # rotate object with anchor
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0))
#     assert np.allclose(s.position, (1,-1,0))
#     assert np.allclose(s.orientation.as_rotvec(), (0,0,np.pi/2))

#     # pad behind
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=1)
#     assert np.allclose(s.position, ((1,-1,0), (2,0,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,np.pi)))

#     # rotate whole path
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0))
#     assert np.allclose(s.position, ((2,0,0), (1,1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi), (0,0,-np.pi/2)))

#     # pad before
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=-3)
#     assert np.allclose(s.position, ((1,1,0), (1,1,0), (0,0,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,-np.pi/2), (0,0,0)))

#     # rotate whole path starting in the middle
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/2)), anchor=(1,0,0), start=1)
#     assert np.allclose(s.position, ((1,1,0), (0,0,0), (1,-1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,0), (0,0,np.pi/2)))

#     # rotate whole path starting in the middle without anchor
#     apply_rotation(s, R.from_rotvec((0,0,np.pi/4)), start=1)
#     assert np.allclose(s.position, ((1,1,0), (0,0,0), (1,-1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,-np.pi/2), (0,0,np.pi/4), (0,0,3*np.pi/4)))


# def test_apply_rotation_v4_pt2():
#     """ v4 path functionality tests """
#     # VECTOR INPUT - ABSOLUTE=FALSE
#     s = magpy.Sensor()

#     # simple append start=auto behavior
#     apply_rotation(s, R.from_rotvec(((0,0,np.pi/2),)), anchor=(1,0,0))
#     assert np.allclose(s.position, ((0,0,0), (1,-1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,0), (0,0,np.pi/2)))

#     # vector + start=0: simple merge
#     apply_rotation(s, R.from_rotvec(((0,0,np.pi/2),)), anchor=(1,0,0), start=0)
#     assert np.allclose(s.position, ((1,-1,0), (1,-1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,np.pi/2)))

#     # vector + start in middle: merge + pad behind
#     apply_rotation(s, R.from_rotvec(((0,0,np.pi), (0,0,np.pi))), anchor=(1,0,0), start=1)
#     assert np.allclose(s.position, ((1,-1,0), (1,1,0), (1,1,0)))
#     assert np.allclose(s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2)))


#     # vector + start before: merge + pad before
#     apply_rotation(s, R.from_rotvec(((0,0,0), (0,0,np.pi))), anchor=(1,0,0), start=-4)
#     assert np.allclose(s.position, ((1,-1,0), (1,1,0), (1,1,0), (1,1,0)))
#     assert np.allclose(
#         s.orientation.as_rotvec(), ((0,0,np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2), (0,0,-np.pi/2)))
