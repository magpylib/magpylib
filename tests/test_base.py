from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_RCSgetBsweepRot_sequential():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [ 0.00453617, -0.07055326,  0.03153698],
                    [0.00488989, 0.04731373, 0.02416068],
                    [0.0249435,  0.00106315, 0.02894469])
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]

    listOfArgs = [  [   [1,2,3],        #pos
                        [0,0,1],        #MPos
                        (180,(0,1,0)),],#Morientation
                    [   [1,2,3],
                        [0,1,0],
                        (90,(1,0,0)),],
                    [   [1,2,3],
                        [1,0,0],
                        (255,(0,1,0)),],]
                    

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBsweep(listOfArgs)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    


def test_RCSgetBsweepRot_multiprocessing():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [ 0.00453617, -0.07055326,  0.03153698],
                    [0.00488989, 0.04731373, 0.02416068],
                    [0.0249435,  0.00106315, 0.02894469])
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]

    listOfArgs = [  [   [1,2,3],        #pos
                        [0,0,1],        #MPos
                        (180,(0,1,0)),],#Morientation
                    [   [1,2,3],
                        [0,1,0],
                        (90,(1,0,0)),],
                    [   [1,2,3],
                        [1,0,0],
                        (255,(0,1,0)),],]
                    

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBsweep(listOfArgs,multiprocessing=True)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    

def test_RCSgetBsweepList():
    erMsg = "Results from getB are unexpected"
    mockResults = ( ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432))
    
    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    listOfPos = [[.5,.5,5],[.5,.5,5],[.5,.5,5]]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBsweep(listOfPos)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    

def test_RCSgetBsweepList_multiprocessing():
    erMsg = "Results from getB are unexpected"
    mockResults = ( ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432))
    
    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    listOfPos = [[.5,.5,5],[.5,.5,5],[.5,.5,5]]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBsweep(listOfPos,multiprocessing=True)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    



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


def test_RCSGetBSequential_Error():
    erMsg = "Results from getB are unexpected"
    mockResults = ( (3.99074612, 4.67238469, 4.22419432),
                    (3.99074612, 4.67238469, 4.22419432),
                    (3.99074612, 4.67238469, 4.22419432),)## Expected 3 results for this input

    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    with pytest.raises(ValueError):
        fieldPos = [[.5,.5,5],
                    [.5,.5,5],
                    [.5,.5,5]]
        # Run
        pm = magnet.Box(mag,dim,pos)
        result = pm.getB(fieldPos)

        rounding = 4 ## Round for floating point error 
        for i in range(len(mockResults)):
            for j in range(3):
                assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg


def test_RCSMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])
    arrayOfPos = array([(.5,.5,5),(30,20,10),(1,.2,60)] )
    ## Positions list
    result = pm.getBsweep(arrayOfPos)

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
    arrayOfPos = array([ (.5,.5,5),
                [30,20,10],
                [1,.2,60],])
                
    result = pm.getBsweep(arrayOfPos) 
    
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
        pm.getB(   (.5,.5,5), #pylint: disable=too-many-function-args
                    [30,20,10],
                    [1,.2,60]) 
        