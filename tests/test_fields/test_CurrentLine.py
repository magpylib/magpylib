from magpylib._lib.fields.Current_Line import Bfield_CurrentLine
from numpy import array
import pytest

def test_Bfield_Zero_Length_segment():
    # Check if Zero-length segments in vertices return valid 
    errMsg = "Field sample outside of Line is unexpected"
    mockResult = [0,0.72951356,0]

    current = 5
    pos = [0,0,0]

    vertices = array([[-1,0,0],[1,0,5],[1,0,5]])
    with pytest.warns(RuntimeWarning):
        results=Bfield_CurrentLine(pos,vertices,current)
        rounding = 4

        for i in range(0,3):
            assert round(mockResult[i],rounding)==round(results[i],rounding), errMsg


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

    results=[Bfield_CurrentLine(pos,vertices,current) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_Bfield_singularity():
    # Test the result fo field sampleS on the line current
    # Each origin vertix, collinear center point
    # Expected: [nan,nan,nan]
    from magpylib import source,Collection
    from numpy import array, isnan
    vertices = [array([1,2,2]),array([1,2,30])]
    current = 5

    origin1 = vertices[0]
    origin2 = vertices[1]
    middle = ((vertices[0]) + (vertices[1])) / 2

    testPos = [origin1,origin2,middle]
    with pytest.warns(RuntimeWarning):
        results = [Bfield_CurrentLine(pos,vertices,current) for pos in testPos]
        assert all(all(isnan(val) for val in result) for result in results), "Results from getB is not NaN"

def test_Bfield_onLine():
    # Check if points that are on the line but 
    # not on the segment still return valid results
    # Expected for collinear points: [0,0,0]
    errMsg = "Points on Line (not on segment) are not being calculated"
    from magpylib import source,Collection
    from numpy import array
    vertices = [array([1,2,2]),array([1,2,30])]
    current = 5
    mockResults = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    points = [vertices[0] + [0,0,-3], vertices[1] + [0,0,3]]

    
    results = [Bfield_CurrentLine(point,vertices,current) for point in points]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg