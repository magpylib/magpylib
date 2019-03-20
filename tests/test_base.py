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

def test_RCSGetBDisplacement_pos():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [0.02596336, 0.04530334, 0.05840059],
                    [0.0670977,  0.11136226, 0.06586208],
                    [0.04866612, 0.02393485, 0.10462079],
                    [-0.0127271,  0.05444677,  0.06974074],
                    [0.02596336, 0.04530334, 0.05840059] )

    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]
    listOfDisplacement=[[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0,0]]
    Bpos = [1,2,3]
    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBDisplacement(Bpos,listOfPos=listOfDisplacement)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

def test_RCSGetBDisplacement_rotAnchored():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [ 0.05545237, -0.03737276,  0.08780605],
                    [0.00064953, 0.02294665, 0.0141079 ],
                    [ 0.0133219,  -0.00171035 , 0.01607478])
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]

    #(angle, axis, anchor)
    listOfRotations = [ (180,(0,1,0),[0,0,1]),
                        (90,(1,0,0),[0,1,0]),
                        (255,(0,1,0),[1,0,0])]
    Bpos = [1,2,3]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBDisplacement(Bpos,
                                listOfRotations=listOfRotations)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

def test_RCSGetBDisplacement_rotAndMove():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [ 0.00453617, -0.07055326,  0.03153698],
                    [0.00488989, 0.04731373, 0.02416068],
                    [0.0249435,  0.00106315, 0.02894469])
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]
    listOfDisplacement=[[0,0,1],
                        [0,1,0],
                        [1,0,0]]

    listOfRotations = [ (180,(0,1,0)),
                        (90,(1,0,0)),
                        (255,(0,1,0))]
    Bpos = [1,2,3]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBDisplacement(Bpos,
                                listOfPos=listOfDisplacement,
                                listOfRotations=listOfRotations)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg


def test_RCSGetBDisplacement_rot():
    erMsg = "Results from getB are unexpected"
    mockResults = ( [0.02596336, 0.04530334, 0.05840059],
                    [0.03795013, 0.03112309, 0.02175846],
                    [-0.00347107, -0.04641532, -0.00475225])
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]
    #listOfDisplacement=None
    listOfRotations = [ (0,(0,0,1)),
                        (90,(0,0,1)),
                        (180,(0,1,0))]
    Bpos = [1,2,3]

    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getBDisplacement(Bpos,listOfRotations=listOfRotations)

    rounding = 4 ## Round for floating point error 
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

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
    result = pm.getBMulticore(arrayOfPos)

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

def test_RCSMulticoreGetBArray_Error():
    erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])

    ## Positions list
    P1=(.5,.5,5)
    P2=[30,20,10]
    P3=[1,.2,60]

    with pytest.raises(ValueError):
        arrayOfPos = array( [ [P1,P2,P3],
                            [P1,P2,P3],
                            [P1,P2,P3] ])
                    
        result = pm.getBMulticore(arrayOfPos) 
        
        ## Expected Results
        B1 = ( 3.99074612, 4.67238469, 4.22419432)
        B2 = ( 0.03900578,  0.01880832, -0.00134112)
        B3 = ( -0.00260347, -0.00313962,  0.00610886)
        mockRes = array( [  [B1,B2,B3],
                            [B1,B2,B3],
                            [B1,B2,B3] ] ) 

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
                
    result = pm.getBMulticore(arrayOfPos) 
    
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
        