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
