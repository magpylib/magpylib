from magpylib.source import magnet
import pytest 

def test_BoxZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[0,0,0],dim=[1,1,1])

def test_BoxZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[1,1,1],dim=[0,0,0])

def test_BoxEdgeCase1():
    ## For now this returns NaN, may be an analytical edge case
    ## Test the Methods in getB() before moving onto this
    from numpy import isnan
    pm = magnet.Box(mag=[0,0,1000],dim=[0.5,0.1,1],pos=[.25,.55,-1111])
    result = pm.getB([.5,.5,5])
    assert any(isnan(result)), "Results from getB is not NaN"