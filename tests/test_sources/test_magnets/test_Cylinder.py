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
    mockResults = array([ 0.62431573,  0.53754927, -0.47024376]) ## Expected results for this input

    # Input
    mag=(6,7,8)
    dim=(2,9)
    pos=(2,2,2)
    fieldPos = (.5,.5,5)

    # Run
    pm = magnet.Cylinder(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_BoxGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 0.01576884,  0.01190684, -0.01747232 ) ## Expected 3 results for this input

    # Input
    mag=(0.2,32.5,5.3)
    dim=(1,2.4)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    fieldPos=[5,5,.35]

    # Run
    pm = magnet.Cylinder(mag,dim,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg