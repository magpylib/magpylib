from magpylib.source import current
from numpy import isnan, array
import pytest 



def test_LineNumpyArray():
    # Test some valid variations of numpy arrays for Line vertices.
    cur=6
    pos=(9,2,4)
    vertices = [array([0,0,0]),array([4,6,2]),array([20,3,6])]
    current.Line(cur,vertices,pos)
    vertices = array([[0,0,0],[4,6,2],[20,3,6]])
    current.Line(cur,vertices,pos)

def test_LineGetB():
    # Test a single getB calculation.
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 0.00653909, -0.01204138,  0.00857173]) ## Expected 3 results for this input
    
    cur=6
    vertices = [[0,0,0],[4,6,2],[20,3,6]]
    pos=(9,2,4)
    fieldPos = (.5,.5,5)

    pm = current.Line(cur,vertices,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_LineGetBAngle():
    # Create the line with a rotated position then verify with getB. 
    erMsg = "Results from getB are unexpected"
    mockResults = (-0.00493354,  0.00980648,  0.0119963 ) ## Expected 3 results for this input
    
    # Input
    curr=2.45
    vertices=[[2,.35,2],[10,2,-4],[4,2,1],[102,2,7]]
    pos=(4.4,5.24,0.5)
    angle=45
    fieldPos=[.5,5,.35]
    
    # Run
    pm = current.Line(curr,vertices,pos,angle)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_LineGetBSweep():
    # Perform multipoint getB calculations with Line (sequential).
    erMsg = "Results from getB are unexpected"
    mockResults = (     (-0.00493354,  0.00980648,  0.0119963 ),
                        (-0.00493354,  0.00980648,  0.0119963 ),
                        (-0.00493354,  0.00980648,  0.0119963 ),) ## Expected 3 results for this input
    
    # Input
    curr=2.45
    vertices=[[2,.35,2],[10,2,-4],[4,2,1],[102,2,7]]
    pos=(4.4,5.24,0.5)
    angle=45
    arrayOfPos=array([  [.5,5,.35],
                [.5,5,.35],
                [.5,5,.35]])
    
    # Run
    pm = current.Line(curr,vertices,pos,angle)

    ## Positions list
    result = pm.getBsweep(arrayOfPos,multiprocessing=False) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg
    