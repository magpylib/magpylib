from magpylib._lib.fields.Current_Line import Bfield_CurrentLine
from numpy import array
import pytest

def test_Bfield_CurrentLine_outside():
    errMsg = "Field sample outside of Box is unexpected"
    mockResults = [ [-15.426123, -42.10796, -12.922307],
                    [67.176642, -3.154985, -10.209148],
                    [-52.57675, 14.702422, 16.730058],
                    [12.5054, 15.171589, -22.647928],
                    [33.504425, -104.324783, 93.824852],
                    [17.274412, 31.725278, -41.418518],
                    [-22.39969, 56.344393, -3.576432],
                    [-11.270571, -9.00747, 3.640508],]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],
                        [-8,9,10],[7,-6,-5],[-8,7,-6],
                        [-9,-8,7],[-10,-9,-8]])                  
    
    #check field values to be within [1,100] adjust magnetization

    current = -11111

    vertices = array([ [-4,-4,-3],[3.5,-3.5,-2],[3,3,-1],
                 [-2.5,2.5,0],[-2,-2,1],[1.5,-1.5,2],[1,1,3]])

    results=[Bfield_CurrentLine(pos,vertices,current) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
