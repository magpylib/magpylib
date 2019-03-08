from typing import Tuple
from magpylib._lib.utility import checkDimensions
from numpy import float64, isnan, array
import pytest

def test_checkDimensionZero():
    errMsg = "Did not raise all zeros Error"
    with pytest.raises(AssertionError):
        checkDimensions(3,dim=(.0,0,.0))

    with pytest.raises(AssertionError):
        checkDimensions(0,dim=[])

def test_checkDimensionMembers():
    errMsg = "Did not raise expected Value Error"
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
    