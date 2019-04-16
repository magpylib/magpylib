from magpylib._lib.fields.PM_Box import Bfield_Box
from numpy import array
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

def test_BfieldBox_outside():
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
    errMsg = "Field sample inside of Box is unexpected"
    mockResults = [ [-115.690442, -473.627914, 113.62309],
                    [130.518528, -264.213509, 358.713317],
                    [-625.378401, 457.382421, -388.191264],
                    [610.012102, 238.448129, -99.225648],
                    [105.212958, -254.693617, 402.588708],
                    [-595.183966, 577.902265, -344.315869],
                    [640.206475, 247.968012, -221.850201],
                    [-201.384834, -372.147757, -96.752349],]

    mag=array([-111,222,-333])

    a,b,c = 2,3,4
    dim=array([a,b,c])

    testPosInside = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2

    results=[Bfield_Box(mag,pos,dim) for pos in testPosInside]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
            
    #add special cases (surface, edges, corners)

    #calc and test fields
    

