from magpylib.source import magnet
from numpy import isnan, array
from magpylib._lib.classes import base
import pytest 

def test_RCSsetOrientation():
    # Check if setOrientation() is working as expected.
    errMsg_angle = "Unexpected RCS angle result for orientation"
    errMsg_axis = "Unexpected RCS axis result for orientation"
    startPos = [1,2,3.5]
    expectedAngle = 180
    expectedAxis = (0,1,0)

    angle = 180
    axis = (0,1,0)
    
    rcs = base.RCS(startPos,90,[0,0,1])
    rcs.setOrientation(angle,axis)
    rounding = 4
    assert round(rcs.angle,rounding) == expectedAngle,errMsg_angle
    assert all(round(rcs.axis[i],rounding) == expectedAxis[i] for i in range(0,3)),errMsg_axis


def test_RCSsetPosition():
    # Check if setPosition() is working as expected.
    errMsg = "Unexpected RCS position result for rotation"
    startPos = [1,2,3.5]
    expectedPos = [-4,9.2,0.0001]
    rcs = base.RCS(startPos,90,[0,0,1])
    rcs.setPosition(expectedPos)
    rounding = 4
    assert all(round(rcs.position[i],rounding) == expectedPos[i] for i in range(0,3)), errMsg

def test_RCSrotate():
    # Check if rotate() is working as expected.
    errMsg_init = "Unexpected RCS position at initialization"
    errMsg_pos = "Unexpected RCS position result for rotation"
    errMsg_angle = "Unexpected RCS angle result for rotation"
    startPos = [1,2,3.5]
    expectedPos = [-2, 1,3.5]
    expectedAngle = 90
    angle = 90
    axis = (0,0,1)
    anchor = [0,0,0]
    rcs = base.RCS(startPos,90,[0,0,1])
    rounding = 4
    assert all(round(rcs.position[i],rounding) == startPos[i] for i in range(0,3)), errMsg_init
    rcs.rotate(angle,axis,anchor)
    assert all(round(rcs.position[i],rounding) == expectedPos[i] for i in range(0,3)), errMsg_pos
    assert round(expectedAngle,rounding) == angle, errMsg_angle

def test_RCSmove():
    # Check if move() is working as expected.
    errMsg_init = "Unexpected RCS position at initialization"
    errMsg_pos = "Unexpected RCS position result for translation"
    startPos = [1,2,3.5]
    expectedPos = [2, 4, 7]
    moveArg = [1,2,3.5]
    rcs = base.RCS(startPos,90,[0,0,1])
    rounding = 4
    assert all(round(rcs.position[i],rounding) == startPos[i] for i in range(0,3)), errMsg_init
    rcs.move(moveArg)
    assert all(round(rcs.position[i],rounding) == expectedPos[i] for i in range(0,3)), errMsg_pos

def test_RCSgetBsweepRot_sequential():
    # Check if getBsweep for Box is performing
    # displacement input sequentially.
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
    result = pm.getBsweep(listOfArgs, multiprocessing = False)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    


def test_RCSgetBsweepRot_multiprocessing():
    # Check if getBsweep for Box is performing
    # displacement input with multiprocessing.
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
    # Check if getBsweep for Box is calculating
    # multipoint input sequentially over a List.
    erMsg = "Results from getB are unexpected"
    mockResults = ( ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432),
                    ( 3.99074612, 4.67238469, 4.22419432))
    
    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    listOfPos = [array([.5,.5,5]),array([.5,.5,5]),array([.5,.5,5])]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBsweep(listOfPos,multiprocessing=True)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg    

def test_RCSgetBsweepList_multiprocessing():
    # Check if getBsweep for Box is calculating
    # multipoint input with multiprocessing over a List.
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
    # Check if getB for Box is calculating
    # a field sample.
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


def test_RCSGetBsweep_Array():
    # Check if getB sweep for box is calculating for an array
    # of field positions.

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

def test_RCSGetBSequentialList_error():
    # Check if getB fails to calculate 
    # a series of different iterables

    #erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])

    ## Positions list
    with pytest.raises(TypeError):         
        pm.getB(   (.5,.5,5), #pylint: disable=too-many-function-args
                    [30,20,10],
                    array([1,.2,60])) 
        