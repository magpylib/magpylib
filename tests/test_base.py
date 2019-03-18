from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_RCSGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 3.99074612, 4.67238469, 4.22419432) ## Expected 3 results for this input

    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getB([.5,.5,5])

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_RCSGetBSequential():
    erMsg = "Results from getB are unexpected"
    mockResults = ( (3.99074612, 4.67238469, 4.22419432),
                    (3.99074612, 4.67238469, 4.22419432),
                    (3.99074612, 4.67238469, 4.22419432),)## Expected 3 results for this input

    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    fieldPos = [[.5,.5,5],
                [.5,.5,5],
                [.5,.5,5]]
    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getB(fieldPos,multicore=False)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg


def test_RCSMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    with pytest.raises(TypeError):
        pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])
        
        ## Positions list
        result = pm.getB(   (.5,.5,5), #pylint: disable=redundant-keyword-arg
                            (30,20,10),
                            (1,.2,60),
                            multicore=True ) 

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


def test_RCSMulticoreGetBList():
    erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])

    ## Positions list
    posList = [ (.5,.5,5),
                [30,20,10],
                [1,.2,60],]
                
    result = pm.getB(posList,multicore=True ) 

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

def test_RCSGetBSequentialList():
    #erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])

    ## Positions list
    with pytest.raises(TypeError):         
        pm.getB(   (.5,.5,5), #pylint: disable=redundant-keyword-arg
                    [30,20,10],
                    [1,.2,60],multicore=False ) 
        