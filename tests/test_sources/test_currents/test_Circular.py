from magpylib.source import current
from numpy import isnan, array
import pytest 

def test_CircularNegDimError():
    with pytest.raises(AssertionError):
        current.Circular(5,dim=-1)

def test_CircularGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([-0.11843504, -0.11843504,  0.4416876 ]) ## Expected results for this input
    
    # Input
    cur=6
    dim=9
    pos=(2,2,2)
    fieldPos = (.5,.5,5)
    
    # Run
    pm = current.Circular(cur,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg
    
def test_CurrentGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 0.00509327,  0.00031343, -0.0385829 ) ## Expected results for this input
    
    # Input
    curr=2.45
    dim=3.1469
    pos=(4.4,5.24,0.5)
    angle=45
    fieldPos=[.5,5,.35]

    # Run
    pm = current.Circular(curr,dim,pos,angle)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_CircularMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = (
                        ( 0.00509327,  0.00031343, -0.0385829 ), ## Expected results for this input
                        ( 0.00509327,  0.00031343, -0.0385829 ),
                        ( 0.00509327,  0.00031343, -0.0385829 ),) 
    
    # Input
    curr=2.45
    dim=3.1469
    pos=(4.4,5.24,0.5)
    angle=45
    arrayPos=array([  [.5,5,.35],
                [.5,5,.35],
                [.5,5,.35]])

    pm = current.Circular(curr,dim,pos,angle)

    ## Positions list
    result = pm.getBMulticore(arrayPos )

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg
    