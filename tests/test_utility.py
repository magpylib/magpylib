from typing import Tuple
from magpylib._lib.utility import checkDimensions, isPosVector,isDisplayMarker
from magpylib._lib.utility import isSensor
from numpy import float64, isnan, array
import pytest


def test_IsDisplayMarker_error():
    marker1=[0,0,0]
    marker2=[0,0,1]
    marker3=[0,0,1,"hello world!"] 
    marker4=[0,0,1,-1] # Should fail!

    markerList = [marker1,marker2,marker3,marker4]
    with pytest.raises(AssertionError):
        for marker in markerList:
            assert isDisplayMarker(marker)

def test_IsDisplayMarker():
    errMsg = "marker identifier has failed: "
    marker1=[0,0,0]
    marker2=[0,0,1]
    marker3=[0,0,1,"hello world!"]

    markerList = [marker1,marker2,marker3]
    for marker in markerList:
        assert isDisplayMarker(marker), errMsg + str(marker)


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
    

def test_isPosVector():
    errMsg = "isPosVector returned unexpected False value"
    position = [1,2,3]
    assert isPosVector(position), errMsg

def test_isPosVectorArray():
    from numpy import array
    errMsg = "isPosVector returned unexpected False value"
    position = array([1,2,3])
    assert isPosVector(position), errMsg

def test_isPosVectorArray2():
    from numpy import array
    errMsg = "isPosVector returned unexpected False value"
    position = array([1,4,-24.242])
    assert isPosVector(position), errMsg

def test_isSensor():
    from magpylib._lib.classes.sensor import Sensor 
    s = Sensor()

    assert isSensor(s)

def test_isPosVectorArgs():
    from numpy import array
    errMsg = "isPosVector returned unexpected False value"
    position = array([1,4,-24.242])
    

    listOfArgs = [  [   [1,2,3],        #pos
                    [0,0,1],        #MPos
                    (180,(0,1,0)),],#Morientation
                 [   [1,2,3],
                     [0,1,0],
                     (90,(1,0,0)),],
                 [   [1,2,3],
                     [1,0,0],
                     (255,(0,1,0)),],]
    assert any(isPosVector(a)==False for a in listOfArgs)                 
