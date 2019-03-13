from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_SphereZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Sphere(mag=[0,0,0],dim=1)

def test_SphereZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Sphere(mag=[1,1,1],dim=0)

def test_SphereGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([-0.05040102, -0.05712116, -0.03360068]) ## Expected 3 results for this input
    
    mag=[6,7,8]
    dim=2
    pos=[2,2,2]
    fieldPos = [.5,.5,5]

    pm = magnet.Sphere(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg