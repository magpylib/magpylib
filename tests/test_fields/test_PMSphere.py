from magpylib._lib.fields.PM_Sphere import Bfield_Sphere
from numpy import array, isnan
import pytest

def test_Bfield_singularity():
    # Test the result for a field sample on the Sphere
    # 3 points in faces for each axis
    # Expected: [nan,nan,nan]
    from magpylib import source,Collection
    from numpy import array
    
    # Definitions
    mag = [1,2,3]
    diam = 5
    r = diam/2
    pos1 = [0,0,r]
    pos2 = [0,r,0]
    pos3 = [r,0,0]
    testPos = [pos1,pos2,pos3]

    # Run
    with pytest.warns(RuntimeWarning):
        results = [Bfield_Sphere(mag,pos,diam) for pos in testPos]
        assert all(all(isnan(axis) for axis in result) for result in results)

def test_Bfield_outside():
    # Fundamental Positions in every 8 Octants
    errMsg = "Field sample outside of Box is unexpected"
    mockResults = [ [-84.219706, -101.128492, -21.194554],
                    [63.04539, 67.104513, -20.254327],
                    [-42.102189, 46.941937, -14.773532],
                    [26.888641, -31.094788, -3.429599],
                    [83.497283, -75.23799, 38.961397],
                    [-64.220733, 53.902248, 11.398246],
                    [42.766322, 34.054828, 8.923501],
                    [-26.470789, -26.624502, 7.024701],]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],
                        [-8,9,10],[7,-6,-5],[-8,7,-6],
                        [-9,-8,7],[-10,-9,-8]])                  
    
    #check field values to be within [1,100] adjust magnetization

    mag=array([-11111,22222,-333333])
    
    a = 2
    diam = a

    results=[Bfield_Sphere(mag,pos,diam) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_Bfield_inside():
    # Fundamental Positions in every 8 Octants, but inside
    from numpy import pi
    errMsg = "Field sample inside of Box is unexpected"
    mockResults = [ [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],
                    [-7407.333333, 14814.666667, -222222.0],]
    mag=array([-11111,22222,-333333])
    a,b,c = 2,3,4
    diam=a
    testPosIn = array([ [0,0,0],[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],
                        [a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/(2*pi)

    results=[Bfield_Sphere(mag,pos,diam) for pos in testPosIn]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
    

