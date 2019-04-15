from magpylib.source.moment import Dipole
from numpy import isnan, array
import pytest 


def test_DipoleGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 1.23927518e-06,  6.18639685e-06, -1.67523560e-06]) ## Expected 3 results for this input

    # Input
    moment=[5,2,10]
    pos=(24,51,22)
    fieldPos = (.5,.5,5)

    # Run
    pm = Dipole(moment,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_DipoleGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = (-0.00836643 , 0.01346,   -0.01833964) ## Expected 3 results for this input

    # Input
    moment=(0.2,32.5,5.3)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    fieldPos=[.5,5,.35]

    # Run
    pm = Dipole(moment,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_DipoleMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = (     (-0.00836643 , 0.01346,   -0.01833964), ## Expected 3 results for this input
                        (-0.00836643 , 0.01346,   -0.01833964),
                        (-0.00836643 , 0.01346,   -0.01833964)) 

    # Input
    moment=(0.2,32.5,5.3)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    arrayOfPos =  array([[.5,5,.35],
                [.5,5,.35],
                [.5,5,.35],])

    # Run
    pm = Dipole(moment,pos,angle,axis)
    ## Positions list
    result = pm.getBsweep(arrayOfPos) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg