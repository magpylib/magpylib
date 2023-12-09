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
