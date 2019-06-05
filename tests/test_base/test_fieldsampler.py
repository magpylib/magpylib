
from magpylib.source import magnet
from numpy import isnan, array,ndarray
from magpylib._lib.classes import fieldsampler
import pytest


def test_getBsweepRot_sequential():
    # Check if getBsweep for Box is performing
    # displacement input sequentially.
    erMsg = "Results from getB are unexpected"
    type_erMsg =  "getBsweep did not return a numpy array."
    mockResults = array(([0.00453617, -0.07055326,  0.03153698],
                         [0.00488989, 0.04731373, 0.02416068],
                         [0.0249435,  0.00106315, 0.02894469]))

    # Input
    mag = [1, 2, 3]
    dim = [1, 2, 3]
    pos = [0, 0, 0]

    listOfArgs = [[[1, 2, 3],  # pos
                   [0, 0, 1],  # MPos
                   (180, (0, 1, 0)), ],  # Morientation
                  [[1, 2, 3],
                   [0, 1, 0],
                   (90, (1, 0, 0)), ],
                  [[1, 2, 3],
                   [1, 0, 0],
                   (255, (0, 1, 0)), ], ]

    # Run
    pm = magnet.Box(mag, dim, pos)
    result = pm.getBsweep(listOfArgs, multiprocessing=False)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 4  # Round for floating point error
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_getBsweepRot_multiprocessing():
    # Check if getBsweep for Box is performing
    # displacement input with multiprocessing.
    erMsg = "Results from getB are unexpected"
    type_erMsg =  "getBsweep did not return a numpy array."
    mockResults = array(([0.00453617, -0.07055326,  0.03153698],
                         [0.00488989, 0.04731373, 0.02416068],
                         [0.0249435,  0.00106315, 0.02894469]))

    # Input
    mag = [1, 2, 3]
    dim = [1, 2, 3]
    pos = [0, 0, 0]

    listOfArgs = [[[1, 2, 3],  # pos
                   [0, 0, 1],  # MPos
                   (180, (0, 1, 0)), ],  # Morientation
                  [[1, 2, 3],
                   [0, 1, 0],
                   (90, (1, 0, 0)), ],
                  [[1, 2, 3],
                   [1, 0, 0],
                   (255, (0, 1, 0)), ], ]

    # Run
    pm = magnet.Box(mag, dim, pos)
    result = pm.getBsweep(listOfArgs, multiprocessing=True)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 4  # Round for floating point error
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_getBsweepList():
    # Check if getBsweep for Box is calculating
    # multipoint input sequentially over a List.
    erMsg = "Results from getB are unexpected"
    mockResults = array(((3.99074612, 4.67238469, 4.22419432),
                         (3.99074612, 4.67238469, 4.22419432),
                         (3.99074612, 4.67238469, 4.22419432)))

    # Input
    mag = [6, 7, 8]
    dim = [10, 10, 10]
    pos = [2, 2, 2]
    listOfPos = [array([.5, .5, 5]), array([.5, .5, 5]), array([.5, .5, 5])]

    # Run
    pm = magnet.Box(mag, dim, pos)
    result = pm.getBsweep(listOfPos, multiprocessing=True)

    rounding = 4  # Round for floating point error
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_getBsweepList_multiprocessing():
    # Check if getBsweep for Box is calculating
    # multipoint input with multiprocessing over a List.
    erMsg = "Results from getB are unexpected"
    type_erMsg =  "getBsweep did not return a numpy array."
    mockResults = array(((3.99074612, 4.67238469, 4.22419432),
                         (3.99074612, 4.67238469, 4.22419432),
                         (3.99074612, 4.67238469, 4.22419432)))

    # Input
    mag = [6, 7, 8]
    dim = [10, 10, 10]
    pos = [2, 2, 2]
    listOfPos = [[.5, .5, 5], [.5, .5, 5], [.5, .5, 5]]

    # Run
    pm = magnet.Box(mag, dim, pos)
    result = pm.getBsweep(listOfPos, multiprocessing=True)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 4  # Round for floating point error
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockResults[i][j], rounding), erMsg


def test_GetB():
    # Check if getB for Box is calculating
    # a field sample.
    erMsg = "Results from getB are unexpected"
    # Expected 3 results for this input
    mockResults = (3.99074612, 4.67238469, 4.22419432)

    # Input
    mag = [6, 7, 8]
    dim = [10, 10, 10]
    pos = [2, 2, 2]

    # Run
    pm = magnet.Box(mag, dim, pos)
    result = pm.getB([.5, .5, 5])

    rounding = 4  # Round for floating point error
    for i in range(3):
        assert round(result[i], rounding) == round(
            mockResults[i], rounding), erMsg


def test_GetBsweep_Array():
    # Check if getB sweep for box is calculating for an array
    # of field positions.
    erMsg = "Results from getB are unexpected"
    type_erMsg =  "getBsweep did not return a numpy array."
    pm = magnet.Box(mag=[6, 7, 8], dim=[10, 10, 10], pos=[2, 2, 2])

    # Positions list
    arrayOfPos = array([(.5, .5, 5),
                        [30, 20, 10],
                        [1, .2, 60], ])

    result = pm.getBsweep(arrayOfPos)
    assert isinstance(result,ndarray), type_erMsg
    # Expected Results
    mockRes = array(((3.99074612, 4.67238469, 4.22419432),  # .5,.5,.5
                    (0.03900578,  0.01880832, -0.00134112),  # 30,20,10
                    (-0.00260347, -0.00313962,  0.00610886), ))

    # Rounding for floating point error
    rounding = 4

    # Loop through predicted cases and check if the positions from results are
    # valid
    for i in range(len(mockRes)):
        for j in range(3):
            assert round(result[i][j], rounding) == round(
                mockRes[i][j], rounding), erMsg


def test_GetBSequentialList_error():
    # Check if getB fails to calculate
    # a series of different iterables

    # erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6, 7, 8], dim=[10, 10, 10], pos=[2, 2, 2])

    # Positions list
    with pytest.raises(TypeError):
        pm.getB((.5, .5, 5),  # pylint: disable=too-many-function-args
                [30, 20, 10],
                array([1, .2, 60]))


def test_GetB_unintialized():
    # Test if unitialized getB in fieldSampler throws a warning.
    pos = [1, 2, 3.5]
    with pytest.warns(Warning):
        FS = fieldsampler.FieldSampler()
        FS.getB(pos)
