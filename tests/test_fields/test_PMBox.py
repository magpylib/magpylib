from magpylib._lib.fields.PM_Box import Bfield_Box
from numpy import array, isnan
import pytest

def test_BfieldBox_OLD():
    errMsg = "Wrong field calculation for BfieldBox"
    
    mag=array([5,5,5])
    dim=array([1,1,1])
    rotatedPos = array([-19. ,   1.2,   8. ])
    mockResults = array([ 1.40028858e-05, -4.89208175e-05, -7.01030695e-05])

    result = Bfield_Box(mag,rotatedPos,dim)
    rounding = 4
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), errMsg

def test_BfieldBox_Edges():
    from numpy import array,array_equal,append
    from magpylib import source, Collection


    mag=array([-111,222,-333])
    a,b,c = 2,3,4
    dim=array([a,b,c])
    testPosEdge = []
    corners = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2
    
    testPosEdge.extend(corners / array([2,1,1])) # testPosEdgesX = 
    testPosEdge.extend(corners  / array([1,2,1])) # testPosEdgesY = 
    testPosEdge.extend(corners / array([1,1,2])) # testPosEdgesZ =

    with pytest.warns(RuntimeWarning):
        results = [Bfield_Box(mag,pos,dim) for pos in testPosEdge]
        assert all(all(isnan(val) for val in result) for result in results), "Results from getB is not NaN"


def test_BfieldBox_Faces():
    from numpy import array,array_equal,append
    from magpylib import source, Collection

    mag=array([-111,222,-333])
    a,b,c = 2,3,4
    dim=array([a,b,c])
    testPosFaces = []
    corners = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2
    testPosFaces.extend(corners / array([2,2,1]))  # testPosFaceX = 
    testPosFaces.extend(corners / array([2,1,2])) # testPosFaceY = 
    testPosFaces.extend(corners / array([1,2,2])) # testPosFaceZ = 

    with pytest.warns(RuntimeWarning):
        results = [Bfield_Box(mag,pos,dim) for pos in testPosFaces]
        assert all(all(isnan(val) for val in result) for result in results), "Results from getB is not NaN"

def test_BfieldBox_OuterLines():
    errMsg = "Unexpected Results for getB in Outer Lines"
    from numpy import array,array_equal,append
    from magpylib import source, Collection

    mag=array([-111,222,-333])
    mockResults = [array([ -7.66913751, -11.43130392,   3.90940536]), array([ 0.5814601 , -5.45527776, 10.643622  ]), 
                  array([-19.62118983,   3.13850731,  -1.81978469]), array([12.53351242, -2.83751885,  4.91443196]), 
                  array([ 0.5814601 , -5.45527776, 10.643622  ]), array([-19.62118983,   3.13850731,  -1.81978469]), 
                  array([12.53351242, -2.83751885,  4.91443196]), array([ -7.66913751, -11.43130392,   3.90940536]), 
                  array([ 2.40147269, -0.80424712,  5.73409625]), array([0.66977042, 1.19674994, 6.4908602 ]), 
                  array([-1.60052144, 10.81979527, -0.6812673 ]), array([4.67176454, 8.81879821, 0.07549665]), 
                  array([0.66977042, 1.19674994, 6.4908602 ]), array([-1.60052144, 10.81979527, -0.6812673 ]), 
                  array([4.67176454, 8.81879821, 0.07549665]), array([ 2.40147269, -0.80424712,  5.73409625]), 
                  array([-0.30055594, -3.65531213, -4.07927409]), array([ 2.06817563, -3.40748556, -3.12447919]), 
                  array([-0.79620907,  0.60480782, -6.75413634]), array([ 2.56382876,  0.35698125, -5.79934144]), 
                  array([ 2.06817563, -3.40748556, -3.12447919]), array([-0.79620907,  0.60480782, -6.75413634]), 
                  array([ 2.56382876,  0.35698125, -5.79934144]), array([-0.30055594, -3.65531213, -4.07927409])]
    a,b,c = 2,3,4
    dim=array([a,b,c])
    testPos = []
    corners = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2
    testPos.extend(corners * array([3,1,1]))  # testPosOuterX = 
    testPos.extend(corners  * array([1,3,1])) # testPosOuterY = 
    testPos.extend(corners  * array([1,1,3])) # testPosOuterZ = 


    results = [Bfield_Box(mag,pos,dim) for pos in testPos]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_BfieldBox_Corners():
    from numpy import array,array_equal,append
    from magpylib import source, Collection

    mag=array([-111,222,-333])
    a,b,c = 2,3,4
    dim=array([a,b,c])

    testPosCorners = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2

    with pytest.warns(RuntimeWarning):
        results = [Bfield_Box(mag,pos,dim) for pos in testPosCorners]
        assert all(all(isnan(val) for val in result) for result in results), "Results from getB is not NaN"
    

def test_BfieldBox_outside():
    # Fundamental Positions in every 8 Octants, but inside
    errMsg = "Field sample outside of Box is unexpected"
    mockResults = [ [-487.520576, -575.369828, -104.423566],
                    [364.861085, 382.575024, -106.896362],
                    [-243.065706, 267.987035, -79.954987],
                    [154.533798, -177.245393, -17.067326],
                    [467.108616, -413.895715, 234.294815],
                    [-364.043702, 300.956661, 72.402694],
                    [242.976273, 191.057477, 54.841929],
                    [-150.641259, -150.43341, 42.180744],]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],
                        [-8,9,10],[7,-6,-5],[-8,7,-6],
                        [-9,-8,7],[-10,-9,-8]])                  
    
    #check field values to be within [1,100] adjust magnetization

    mag=array([-11111,22222,-333333])
    
    a,b,c = 2,3,4
    dim=array([a,b,c])

    results=[Bfield_Box(mag,pos,dim) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_BfieldBox_inside():
    # Fundamental Positions in every 8 Octants, but inside
    errMsg = "Field sample inside of Box is unexpected"
    mockResults = [ [-57.457487, 133.687466, -259.77011],
                    [-56.028444, 147.488799, -250.092873],
                    [-85.060153, 175.141795, -278.20544],
                    [-28.425778, 161.340462, -268.528204],
                    [-56.028444, 147.488799, -250.092873],
                    [-85.060153, 175.141795, -278.20544],
                    [-28.425778, 161.340462, -268.528204],
                    [-57.457487, 133.687466, -259.77011],]

    mag=array([-111,222,-333])

    a,b,c = 2,3,4
    dim=array([a,b,c])

    testPosInside = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/4

    results=[Bfield_Box(mag,pos,dim) for pos in testPosInside]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
            
    #add special cases (surface, edges, corners)

    #calc and test fields
    

