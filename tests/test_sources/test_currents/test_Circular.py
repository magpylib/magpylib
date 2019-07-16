from magpylib.source import current
from numpy import isnan, array
import pytest


def test_CircularNegDimError():
    with pytest.raises(AssertionError):
        current.Circular(5, dim=-1)


def test_CircularGetB():
    erMsg = "Results from getB are unexpected"
    # Expected results for this input
    mockResults = array([-0.11843504, -0.11843504,  0.4416876])

    # Input
    cur = 6
    dim = 9
    pos = (2, 2, 2)
    fieldPos = (.5, .5, 5)

    # Run
    pm = current.Circular(cur, dim, pos)
    result = pm.getB(fieldPos)

    rounding = 4  # Round for floating point error
    for i in range(3):
        assert round(result[i], rounding) == round(
            mockResults[i], rounding), erMsg


def test_CircularLine_rotation_GetB():
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
    mockResults = [[0.524354, 0.697093, 2.587274],
                   [0.01134, 0.028993, 0.010118],
                   [0.005114, 0.04497, 0.023742],
                   [0.857106, -0.01403, 1.168066],
                   [-0.036555, -0.239513, 0.038604],
                   [-0.009827, -0.027402, 0.018106], ]

    cur = 69
    pos = [2, 2, 2]
    dim = 2
    pm = current.Circular(cur, dim, pos)

    results = [applyRotAndReturnB(arg, pm) for arg in arguments]
    rounding = 3
    for i in range(0, len(mockResults)):
        for j in range(0, 3):
            assert round(mockResults[i][j], rounding) == round(
                results[i][j], rounding), errMsg


def test_CircularLineGetB_rotation():
    erMsg = "Results from getB are unexpected"
    from numpy import pi

    def applyRotationAndReturnStatus(arg, obj):
        obj.rotate(arg[0], arg[1], arg[2])
        result = {"cur": obj.current,
                  "pos": obj.position,
                  "ang": obj.angle,
                  "axi": obj.axis,
                  "dim": obj.dimension, }
        return result

    arguments = [[36, (1, 2, 3), (.1, .2, .3)],
                 [-366, [3, -2, -1], [-.5, -.6, .7]],
                 [pi, [0, 0, 0.0001], [-.5, -2, .7]]]
    mockResults = [{'cur': 0.69, 'pos': array([1.46754927, 2.57380229, 1.79494871]),
                    'ang': 36.00000000000002, 'axi': array([0.26726124, 0.53452248, 0.80178373]), 'dim': 2.0},
                   {'cur': 0.69, 'pos': array([1.4274764, 2.70435404, 1.41362661]),
                    'ang': 321.8642936876839, 'axi': array([-0.14444227, -0.62171816, -0.76980709]), 'dim': 2.0},
                   {'cur': 0.69, 'pos': array([1.16676385, 2.80291687, 1.41362661]),
                    'ang': 319.3981749889049, 'axi': array([-0.11990803, -0.58891625, -0.79924947]), 'dim': 2.0}, ]
    cur = 0.69
    pos = [2, 2, 2]
    dim = 2
    pm = current.Circular(cur, dim, pos)

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

        assert round(results[i]['dim'], rounding) == round(
            mockResults[i]['dim'], rounding), erMsg
        assert round(results[i]['cur'], rounding) == round(
            mockResults[i]['cur'], rounding), erMsg
        assert round(results[i]['ang'], rounding) == round(
            mockResults[i]['ang'], rounding), erMsg


def test_CurrentGetBAngle():
    erMsg = "Results from getB are unexpected"
    # Expected results for this input
    mockResults = (0.00509327,  0.00031343, -0.0385829)

    # Input
    curr = 2.45
    dim = 3.1469
    pos = (4.4, 5.24, 0.5)
    angle = 45
    fieldPos = [.5, 5, .35]

    # Run
    pm = current.Circular(curr, dim, pos, angle)
    result = pm.getB(fieldPos)

    rounding = 4  # Round for floating point error
    for i in range(3):
        assert round(result[i], rounding) == round(
            mockResults[i], rounding), erMsg


def test_CircularMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = (
        # Expected results for this input
        (0.00509327,  0.00031343, -0.0385829),
        (0.00509327,  0.00031343, -0.0385829),
        (0.00509327,  0.00031343, -0.0385829),)

    # Input
    curr = 2.45
    dim = 3.1469
    pos = (4.4, 5.24, 0.5)
    angle = 45
    arrayPos = [[.5, 5, .35],
                [.5, 5, .35],
                [.5, 5, .35]]

    pm = current.Circular(curr, dim, pos, angle)

    # Positions list
    result = pm.getBsweep(arrayPos)

    # Rounding for floating point error
    rounding = 4

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_ToString():
    curr = 2.45
    dimension = 3.1469
    position = (4.4, 5.24, 0.5)
    angle = 45.0
    axis = [0.2, 0.61, 1.0]
    expected = "type: {} \n current: {}  \n dimension: d: {} \n position: x: {}, y: {}, z: {} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format(
        "current.Circular", curr, dimension, *position, angle, *axis)

    myCircular = current.Circular(curr, dimension, position, angle, axis)

    result = myCircular.__repr__()
    assert result == expected

    