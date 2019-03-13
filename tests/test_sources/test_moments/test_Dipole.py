from magpylib.source.moment import Dipole
from numpy import isnan, array
import pytest 


def test_DipoleGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 1.23927518e-06,  6.18639685e-06, -1.67523560e-06]) ## Expected 3 results for this input
    moment=[5,2,10]
    pos=(24,51,22)
    fieldPos = (.5,.5,5)

    pm = Dipole(moment,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg