from typing import Tuple
from magpylib._lib.utility import checkDimensions,rotateToCS,getBField
from numpy import float64, isnan, array
import pytest

def test_checkDimensionZero():
    # errMsg = "Did not raise all zeros Error"
    with pytest.raises(AssertionError):
        checkDimensions(3,dim=(.0,0,.0))

    with pytest.raises(AssertionError):
        checkDimensions(0,dim=[])

def test_checkDimensionMembers():
    # errMsg = "Did not raise expected Value Error"
    with pytest.raises(ValueError):
        checkDimensions(3,dim=(3,'r',6))

def test_checkDimensionSize():
    errMsg = "Did not raise wrong dimension size Error"
    with pytest.raises(AssertionError) as error:
        checkDimensions(3,dim=(3,5,9,6))
    assert error.type == AssertionError, errMsg

def test_checkDimensionReturn():
    errMsg = "Wrong return dimension size"
    result = checkDimensions(4,dim=(3,5,9,10))
    assert len(result) == 4, errMsg
    result = checkDimensions(3,dim=(3,5,9))
    assert len(result) == 3, errMsg
    result = checkDimensions(2,dim=(3,5))
    assert len(result) == 2, errMsg
    result = checkDimensions(1,dim=(3))
    assert len(result) == 1, errMsg
    
def test_rotateToCS():
    errMsg =  "Wrong rotation for Box in CS"

    mag=[5,5,5]
    dim=[1,1,1]
    pos=[63,.8,2]
    mockResults = array([-19. ,   1.2,   8. ])
    
    from magpylib import source
    b = source.magnet.Box(mag,dim,pos)
    result = rotateToCS([44,2,10],b)
    
    rounding=4
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), errMsg

def test_getBField():
    errMsg =  "Wrong field for Box in CS"
    mockField = array([ 1.40028858e-05, -4.89208175e-05, -7.01030695e-05])
    axis = [1,1,1]
    angle = 90
    mockResults = array([-4.72365793e-05,  1.35515955e-05, -7.13360174e-05])
    
    class MockSource: ## Mock a Source object.
        def __init__(self, axis, angle):
            self.axis = axis
            self.angle = angle

    mockSource = MockSource(axis,angle)
    result =  getBField(mockField, mockSource)

    rounding = 4
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), errMsg