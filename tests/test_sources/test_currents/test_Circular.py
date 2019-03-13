from magpylib.source import current
from numpy import isnan, array
import pytest 

def test_CircularNegDimError():
    with pytest.raises(AssertionError):
        current.Circular(5,dim=-1)

def test_CircularGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([-0.11843504, -0.11843504,  0.4416876 ]) ## Expected 3 results for this input
    
    cur=6
    dim=9
    pos=(2,2,2)
    fieldPos = (.5,.5,5)

    pm = current.Circular(cur,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg