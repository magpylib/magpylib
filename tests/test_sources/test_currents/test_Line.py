from magpylib.source.current import Line
from magpylib.source import current
from numpy import isnan, array
import pytest


def test_LineNumpyArray():
    # Test some valid variations of numpy arrays for Line vertices.
    cur = 6
    pos = (9, 2, 4)
    vertices = [array([0, 0, 0]), array([4, 6, 2]), array([20, 3, 6])]
    current.Line(cur, vertices, pos)
    vertices = array([[0, 0, 0], [4, 6, 2], [20, 3, 6]])
    current.Line(cur, vertices, pos)


def test_LineGetB():
    # Test a single getB calculation.
    erMsg = "Results from getB are unexpected"
    # Expected 3 results for this input
    mockResults = array([0.00653909, -0.01204138,  0.00857173])

    cur = 6
    vertices = [[0, 0, 0], [4, 6, 2], [20, 3, 6]]
    pos = (9, 2, 4)
    fieldPos = (.5, .5, 5)

    pm = current.Line(cur, vertices, pos)
    result = pm.getB(fieldPos)

    rounding = 4  # Round for floating point error
    for i in range(3):
        assert round(result[i], rounding) == round(
            mockResults[i], rounding), erMsg


def test_Line_rotation_GetB():
    errMsg = "Results from getB are unexpected"
    from numpy import pi
    # Setup

    def applyRotAndReturnB(arg, obj):
        obj.rotate(arg[1], arg[2], arg[3])
        return obj.getB(arg[0])

    arguments = [[[2, 3, 4], 36, (1, 2, 3), (.1, .2, .3)],  # Position, Angle, Axis, Anchor
                 [[-3, -4, -5], -366, [3, -2, -1], [-.5, -.6, .7]],
                 [[-1, -3, -5], pi, [0, 0, 0.0001], [-.5, -2, .7]],
                 [[2, 3, 4], 36, [1, 2, 3], [.1, .2, .3]],
                 [[-3, -4, -5], -124, [3, -2, -1], [-.5, -.6, .7]],
                 [[-1, -3, -5], 275, [-2, -2, 4], [0, 0, 0]]]
    mockResults = [[0.020356, -0.052759, 0.139554],
                   [-0.002636, 0.007098, 0.002922],
                   [-0.007704, 0.01326, 0.003852],
                   [0.101603, -0.004805, 0.073423],
                   [-0.018195, -0.017728, -0.00885],
                   [-0.010693, -0.004727, -0.002245], ]

    cur = 0.69
    pos = [2, 2, 2]

    vertices = [[-4, -4, -3], [3.5, -3.5, -2], [3, 3, -1],
                [-2.5, 2.5, 0], [-2, -2, 1], [1.5, -1.5, 2], [1, 1, 3]]

    pm = current.Line(cur, vertices, pos)
    results = [applyRotAndReturnB(arg, pm) for arg in arguments]

    rounding = 3
    for i in range(0, len(mockResults)):
        for j in range(0, 3):
            assert round(mockResults[i][j], rounding) == round(
                results[i][j], rounding), errMsg


def test_LineGetB_rotation():
    erMsg = "Results from getB are unexpected"
    from numpy import pi
    from numpy import array_equal

    def applyRotationAndReturnStatus(arg, obj):
        obj.rotate(arg[0], arg[1], arg[2])
        result = {"cur": obj.current,
                  "pos": obj.position,
                  "ang": obj.angle,
                  "axi": obj.axis,
                  "ver": obj.vertices, }
        return result

    arguments = [[36, (1, 2, 3), (.1, .2, .3)],
                 [-366, [3, -2, -1], [-.5, -.6, .7]],
                 [pi, [0, 0, 0.0001], [-.5, -2, .7]]]

    vertices = [[-4, -4, -3], [3.5, -3.5, -2], [3, 3, -1],
                [-2.5, 2.5, 0], [-2, -2, 1], [1.5, -1.5, 2], [1, 1, 3]]

    mockResults = [{'cur': 0.69, 'pos': array([1.46754927, 2.57380229, 1.79494871]), 'ang': 36.00000000000002,
                    'axi': array([0.26726124, 0.53452248, 0.80178373]), 'ver': vertices},
                   {'cur': 0.69, 'pos': array([1.4274764, 2.70435404, 1.41362661]), 'ang': 321.8642936876839,
                    'axi': array([-0.14444227, -0.62171816, -0.76980709]), 'ver':vertices},
                   {'cur': 0.69, 'pos': array([1.16676385, 2.80291687, 1.41362661]), 'ang': 319.3981749889049,
                    'axi': array([-0.11990803, -0.58891625, -0.79924947]), 'ver': vertices}]
    cur = 0.69
    pos = [2, 2, 2]

    pm = current.Line(cur, vertices, pos)

    results = [applyRotationAndReturnStatus(arg, pm,) for arg in arguments]
    print(results)
    rounding = 4  # Round for floating point error
    for i in range(0, len(mockResults)):
        for j in range(3):
            assert round(results[i]['axi'][j], rounding) == round(
                mockResults[i]['axi'][j], rounding), erMsg
        for j in range(3):
            assert round(results[i]['pos'][j], rounding) == round(
                mockResults[i]['pos'][j], rounding), erMsg
        assert array_equal(results[i]['ver'], mockResults[i]['ver']), erMsg
        assert round(results[i]['cur'], rounding) == round(
            mockResults[i]['cur'], rounding), erMsg
        assert round(results[i]['ang'], rounding) == round(
            mockResults[i]['ang'], rounding), erMsg


def test_LineGetBAngle():
    # Create the line with a rotated position then verify with getB.
    erMsg = "Results from getB are unexpected"
    # Expected 3 results for this input
    mockResults = (-0.00493354,  0.00980648,  0.0119963)

    # Input
    curr = 2.45
    vertices = [[2, .35, 2], [10, 2, -4], [4, 2, 1], [102, 2, 7]]
    pos = (4.4, 5.24, 0.5)
    angle = 45
    fieldPos = [.5, 5, .35]

    # Run
    pm = current.Line(curr, vertices, pos, angle)
    result = pm.getB(fieldPos)

    rounding = 4  # Round for floating point error
    for i in range(3):
        assert round(result[i], rounding) == round(
            mockResults[i], rounding), erMsg


def test_LineGetBSweep():
    # Perform multipoint getB calculations with Line (sequential).
    erMsg = "Results from getB are unexpected"
    mockResults = ((-0.00493354,  0.00980648,  0.0119963),
                   (-0.00493354,  0.00980648,  0.0119963),
                   (-0.00493354,  0.00980648,  0.0119963),)  # Expected 3 results for this input

    # Input
    curr = 2.45
    vertices = [[2, .35, 2], [10, 2, -4], [4, 2, 1], [102, 2, 7]]
    pos = (4.4, 5.24, 0.5)
    angle = 45
    arrayOfPos = array([[.5, 5, .35],
                        [.5, 5, .35],
                        [.5, 5, .35]])

    # Run
    pm = current.Line(curr, vertices, pos, angle)

    # Positions list
    result = pm.getBsweep(arrayOfPos, multiprocessing=False)

    # Rounding for floating point error
    rounding = 4

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_ToString():
    curr = 2.45
    vertices = [[2, .35, 2], [10, 2, -4], [4, 2, 1], [102, 2, 7]]
    position = (4.4, 5.24, 0.5)
    angle = 45.0
    axis = [0.2,0.61, 1.0]
    expected = "type: {} \n current: {} \n dimensions: vertices \n position: x: {}, y: {}, z: {} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format("current.Line", curr, *position, angle, *axis)

    myLine = current.Line(curr, vertices, position, angle, axis)

    result = myLine.__repr__()
    assert result == expected
