from magpylib._lib.fields.PM_Box import Bfield_Box
from numpy import array
import pytest

def test_BfieldBox():
    errMsg = "Wrong field calculation for BfieldBox"
    
    mag=array([5,5,5])
    dim=array([1,1,1])
    rotatedPos = array([-19. ,   1.2,   8. ])
    mockResults = array([ 1.40028858e-05, -4.89208175e-05, -7.01030695e-05])

    result = Bfield_Box(mag,rotatedPos,dim)
    rounding = 4
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), errMsg