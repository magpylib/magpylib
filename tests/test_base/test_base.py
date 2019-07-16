from magpylib.source import magnet
from numpy import isnan, array
from magpylib._lib.classes import base
from magpylib._lib.classes.base import MagMoment
import pytest


def test_initialization_bad_axis_error():
    pos = [1, 23, 2]
    angle = 90
    badAxis = (1, 1, "str")
    with pytest.raises(AssertionError):
        base.RCS(pos, angle, badAxis)


def test_initialization_bad_axis_error2():
    pos = [1, 23, 2]
    angle = 90
    badAxis = (0, 0, 0)
    with pytest.raises(AssertionError):
        base.RCS(pos, angle, badAxis)


def test_initialization_bad_axis_error3():
    pos = [23, 2, 20]
    angle = 90
    badAxis = (1, 1)
    with pytest.raises(SystemExit):
        base.RCS(pos, angle, badAxis)

def test_initialization_bad_pos_error():
    badPos = [23, 2]
    angle = 90
    axis = (1, 1, 1)
    with pytest.raises(SystemExit):
        base.RCS(badPos, angle, axis)


def test_initialization_bad_angle_error():
    pos = [23, 2, 20]
    badAngle = "string"
    axis = (1, 1)
    with pytest.raises(SystemExit):
        base.RCS(pos, badAngle, axis)




def test_setOrientation_bad_axis_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    angle = 90
    badAxis = [1, 2, 3, 4]
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.setOrientation(angle, badAxis)


def test_setOrientation_bad_angle_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    badAngle = "string"
    axis = [1, 2, 3]
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.setOrientation(badAngle, axis)


def test_setOrientation():
    # Check if setOrientation() is working as expected.
    errMsg_angle = "Unexpected RCS angle result for orientation"
    errMsg_axis = "Unexpected RCS axis result for orientation"
    startPos = [1, 2, 3.5]
    expectedAngle = 180
    expectedAxis = (0, 1, 0)

    angle = 180
    axis = (0, 1, 0)

    rcs = base.RCS(startPos, 90, [0, 0, 1])
    rcs.setOrientation(angle, axis)
    rounding = 4
    assert round(rcs.angle, rounding) == expectedAngle, errMsg_angle
    assert all(round(rcs.axis[i], rounding) == expectedAxis[i]
               for i in range(0, 3)), errMsg_axis


def test_setPosition_bad_pos_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    crashValue = [1, 2, 3, 4]

    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.setPosition(crashValue)


def test_setPosition():
    # Check if setPosition() is working as expected.
    errMsg = "Unexpected RCS position result for rotation"
    startPos = [1, 2, 3.5]
    expectedPos = [-4, 9.2, 0.0001]
    rcs = base.RCS(startPos, 90, [0, 0, 1])
    rcs.setPosition(expectedPos)
    rounding = 4
    assert all(round(rcs.position[i], rounding) ==
               expectedPos[i] for i in range(0, 3)), errMsg


def test_rotate_bad_axis_error1():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    invalidAxis = [0, 0, 0]
    angle = 90
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.rotate(angle, invalidAxis)


def test_rotate_bad_axis_error2():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    invalidAxis = [0, 0, 1, 2]
    angle = 90
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.rotate(angle, invalidAxis)


def test_rotate_bad_angle_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    axis = [0, 0, 1]
    badAngle = "string"
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.rotate(badAngle, axis)


def test_rotate_bad_anchor_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    axis = [0, 0, 1]
    angle = 90
    badAnchor = [0, 0, 0, 0]
    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.rotate(angle, axis, badAnchor)


def test_rotate_anchor():
    # Check if rotate() is working as expected WITH ANCHOR.
    errMsg_init = "Unexpected RCS position at initialization"
    errMsg_pos = "Unexpected RCS position result for rotation"
    errMsg_angle = "Unexpected RCS angle result for rotation"
    startPos = [1, 2, 3.5]
    expectedPos = [-2, 1, 3.5]
    expectedAngle = 90
    angle = 90
    axis = (0, 0, 1)
    anchor = [0, 0, 0]
    rcs = base.RCS(startPos, 90, axis)
    rounding = 4
    assert all(round(rcs.position[i], rounding) ==
               startPos[i] for i in range(0, 3)), errMsg_init
    rcs.rotate(angle, axis, anchor)
    assert all(round(rcs.position[i], rounding) ==
               expectedPos[i] for i in range(0, 3)), errMsg_pos
    assert round(expectedAngle, rounding) == angle, errMsg_angle


def test_rotate():
    # Check if rotate() is working as expected WITH ANCHOR.
    errMsg_init = "Unexpected RCS position at initialization"
    errMsg_pos = "Unexpected RCS position result for rotation"
    errMsg_angle = "Unexpected RCS angle result for rotation"
    startPos = [1, 2, 3.5]
    expectedPos = [1,  2,  3.5]
    expectedAngle = 90
    angle = 90
    axis = (0, 0, 1)
    rcs = base.RCS(startPos, 90, axis)
    rounding = 4
    assert all(round(rcs.position[i], rounding) ==
               startPos[i] for i in range(0, 3)), errMsg_init
    rcs.rotate(angle, axis)
    assert all(round(rcs.position[i], rounding) ==
               expectedPos[i] for i in range(0, 3)), errMsg_pos
    assert round(expectedAngle, rounding) == angle, errMsg_angle


def test_move_bad_pos_error():
    # Check if setPosition() is working as expected.
    startPos = [1, 2, 3.5]
    crashValue = [1, 2, 3, 4]

    with pytest.raises(SystemExit):
        rcs = base.RCS(startPos, 90, [0, 0, 1])
        rcs.move(crashValue)


def test_move():
    # Check if move() is working as expected.
    errMsg_init = "Unexpected RCS position at initialization"
    errMsg_pos = "Unexpected RCS position result for translation"
    startPos = [1, 2, 3.5]
    expectedPos = [2, 4, 7]
    moveArg = [1, 2, 3.5]
    rcs = base.RCS(startPos, 90, [0, 0, 1])
    rounding = 4
    assert all(round(rcs.position[i], rounding) ==
               startPos[i] for i in range(0, 3)), errMsg_init
    rcs.move(moveArg)
    assert all(round(rcs.position[i], rounding) ==
               expectedPos[i] for i in range(0, 3)), errMsg_pos


def test_LineCurrent_badCurrent():
    # Check if setPosition() is working as expected.
    badCurrent = "string"
    pos = [1, 2, 3.5]
    angle = 90
    axis = [1, 2, 3]
    with pytest.raises(SystemExit):
        base.LineCurrent(pos, angle, axis, badCurrent)


def test_MagMoment_initialization_bad_Mag():
    badMoment = [0, 0, 0]
    pos = [23, 2, 2]
    angle = 90
    axis = (1, 1, 1)
    with pytest.raises(AssertionError):
        MagMoment(moment=badMoment, pos=pos,
                  angle=angle, axis=axis)
