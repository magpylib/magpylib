import unittest
from magpylib._lib.mathLibPublic import randomAxis, axisFromAngles, anglesFromAxis, rotatePosition
import numpy

def test_randomAxis():
    """
    #Test if returned random axis are valid
    """
    result = randomAxis()
    assert len(result)==3, "Returning axis should be 3"
    assert all(type(axis)==numpy.float64 for axis in result), "Axis values are not float64"
    assert all(abs(axis)<=1 for axis in result), "Absolute axis values returned greater than 1"

def test_rotatePosition():
    expectedResult = [-0.26138058, 0.59373138, 3.28125372]
    position = [1,2,3]
    result = rotatePosition(position,234.5,(0,0.2,1),anchor=[0,1,0])
    rounding = 4
    for i in range(0,3):
        assert round(result[i],rounding)==round(expectedResult[i],rounding)

def test_anglesFromAxis():
    expectedResult = [90.,11.30993247]
    axis = (0,0.2,1)
    result = anglesFromAxis(axis)

    rounding = 4
    for i in range(0,2):
        assert round(result[i],rounding)==round(expectedResult[i],rounding)
def test_axisFromAngles():
    azimuth = 90
    polar = 120
    expectedResult = [ 5.30287619e-17,  8.66025404e-01, -5.00000000e-01]

    # Run 
    result = axisFromAngles([azimuth,polar])
    rounding = 4
    for i in range(0,3):
        assert round(result[i],rounding)==round(expectedResult[i],rounding)