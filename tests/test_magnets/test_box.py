from magpylib.source import magnet
import pytest 

def test_BoxZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[0,0,0],dim=[1,1,1])

def test_BoxZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[1,1,1],dim=[0,0,0])

def test_BoxEdgeCase1():
    ## For now this returns NaN, may be an analytical edge case
    ## Test the Methods in getB() before moving onto this
    from numpy import isnan
    pm = magnet.Box(mag=[0,0,1000],dim=[0.5,0.1,1],pos=[.25,.55,-1111])
    result = pm.getB([.5,.5,5])
    assert any(isnan(result)), "Results from getB is not NaN"

def test_BoxGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 3.99074612, 4.67238469, 4.22419432) ## Expected 3 results for this input
    rounding = 4 ## Round for floating point error 
    
    from numpy import isnan, array
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])
    result = pm.getB([.5,.5,5])

    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_BoxMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])

    ## Positions list
    result = pm.multiGetB(  (.5,.5,5),
                            (30,20,10),
                            (1,.2,60), ) 

    ## Expected Results
    mockRes = ( ( 3.99074612, 4.67238469, 4.22419432), # .5,.5,.5
                ( 0.03900578,  0.01880832, -0.00134112), # 30,20,10
                ( -0.00260347, -0.00313962,  0.00610886), ) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockRes)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockRes[i][j],rounding), erMsg
    