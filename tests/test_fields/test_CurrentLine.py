from magpylib._lib.fields.Current_Line_vector import Bfield_CurrentLineV
from numpy import array
import pytest
import numpy as np

# -------------------------------------------------------------------------------
def test_Bfield_Zero_Length_segment():
    # Check if Zero-length segments in vertices return valid 
    errMsg = "Field sample outside of Line is unexpected"
    mockResult = [0,0.72951356,0]

    current = 5
    pos = [0,0,0]

    vertices = array([[-1,0,0],[1,0,5],[1,0,5]])
    
    results=Bfield_CurrentLineV(vertices,current,pos)
    rounding = 4

    for i in range(0,3):
        assert round(mockResult[i],rounding)==round(results[i],rounding), errMsg
    

# -------------------------------------------------------------------------------
def test_Bfield_CurrentLine_outside():
    # Fundamental Positions in every 8 Octants
    errMsg = "Field sample outside of Line is unexpected"
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

    results=[Bfield_CurrentLineV(vertices,current,pos) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

# -------------------------------------------------------------------------------
def test_Bfield_onLine():
    # Check if points that are on the line but 
    # not on the segment still return valid results
    # Expected for collinear points: [0,0,0]

    vertices = np.array([[1,2,2],[1,2,30]])
    current = 5
    mockResults = np.zeros((2,3))

    points = [vertices[0] + array([0,0,-3]), vertices[1] + array([0,0,3])] #on line
    
    results = array([Bfield_CurrentLineV(vertices,current,point) for point in points])
    
    assert np.all((results==mockResults).ravel())