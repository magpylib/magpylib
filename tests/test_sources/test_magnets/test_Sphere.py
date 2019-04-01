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
    
    # Input
    mag=[6,7,8]
    dim=2
    pos=[2,2,2]
    fieldPos = [.5,.5,5]

    # Run
    pm = magnet.Sphere(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_SphereGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = (-0.00047774, -0.00535384, -0.00087997) ## Expected 3 results for this input

    # Input
    mag=(0.2,32.5,5.3)
    dim=1
    pos=(1,0.2,3)
    axis=[0.2,.61,1]
    angle=89
    fieldPos=[5,5,.35]

    # Run
    pm = magnet.Sphere(mag,dim,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_SphereMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( (-0.00047774, -0.00535384, -0.00087997), ## Expected results for this input
                    (-0.00047774, -0.00535384, -0.00087997),
                    (-0.00047774, -0.00535384, -0.00087997),)

    mag=(0.2,32.5,5.3)
    dim=1
    pos=(1,0.2,3)
    axis=[0.2,.61,1]
    angle=89
    arrayOfPos =array ([  [5,5,.35],
                [5,5,.35],
                [5,5,.35],])

    # Run
    pm = magnet.Sphere(mag,dim,pos,angle,axis)
    result = pm.getBparallel(arrayOfPos  ) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg
    