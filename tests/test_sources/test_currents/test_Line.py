from magpylib.source import current
from numpy import isnan, array
import pytest 


def test_LineGetB():
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