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


def test_Bfield_Box():
    errMsg = "Wrong output of Bfield_Box function"

    mag=array([-111,222,-333])
    
    #check field values to be within [1,100] adjust magnetization

    a,b,c = 2,3,4
    dim=array([a,b,c])

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],[-8,9,10],[7,-6,-5],[-8,7,-6],[-9,-8,7],[-10,-9,-8]])
    testPosInside = array([[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],[a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/2
    #add special cases (surface, edges, corners)

    #calc and test fields

def test_Bfield_Box():




def test_Bfield_CurrentLine():
    errMsg = "Wrong output of Bfield_CurrentLine function"

    vertices = array([[-4,-4,-3],[3.5,-3.5,-2],[3,3,-1],[-2.5,2.5,0],[-2,-2,1],[1.5,-1.5,2],[1,1,3]])

    #calcualte field - pick current so that Bfield values lie within [1,100]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],[-8,9,10],[7,-6,-5],[-8,7,-6],[-9,-8,7],[-10,-9,-8]])
    #add special cases (surface, edges, corners)

    #calc and test fields
