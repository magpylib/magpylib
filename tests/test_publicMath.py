import unittest
from magpylib._lib.mathLibPublic import randomAxis, axisFromAngles, anglesFromAxis, rotatePosition
import numpy

# -------------------------------------------------------------------------------
def test_randomAxis():
    result = randomAxis()
    assert len(result)==3, "bad randomAxis"
    assert all(type(axis)==numpy.float64 for axis in result), "bad randomAxis"
    assert all(abs(axis)<=1 for axis in result), "bad randomAxis"

# -------------------------------------------------------------------------------
def test_rotatePosition():
    sol = [-0.26138058, 0.59373138, 3.28125372]
    result = rotatePosition([1,2,3],234.5,(0,0.2,1),anchor=[0,1,0])
    for r,s in zip(result,sol):
        assert round(r,4) == round(s,4), "bad rotatePosition"

# -------------------------------------------------------------------------------
def test_anglesFromAxis():
    sol = [90.,11.30993247]
    result = anglesFromAxis([0,.2,1])
    for r,s in zip(result,sol):
        assert round(r,4)==round(s,4), "bad anglesFromAxis"

# -------------------------------------------------------------------------------
def test_axisFromAngles():
    sol = [ 5.30287619e-17,  8.66025404e-01, -5.00000000e-01]
    result = axisFromAngles([90,120])
    for r,s in zip(result,sol):
        assert round(r,4)==round(s,4), "bad axisFromAngles"