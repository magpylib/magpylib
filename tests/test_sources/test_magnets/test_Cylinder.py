from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_CylinderZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Cylinder(mag=(0,0,0),dim=(1,1))

def test_CylinderZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Cylinder(mag=(1,1,1),dim=(0,0))

def test_CylinderGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 0.62431573,  0.53754927, -0.47024376]) ## Expected 3 results for this input
    
    mag=(6,7,8)
    dim=(2,9)
    pos=(2,2,2)
    fieldPos = (.5,.5,5)

    pm = magnet.Cylinder(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg