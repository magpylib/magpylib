from magpylib._lib.fields.Moment_Dipole import Bfield_Dipole
from numpy import array, isnan
import pytest


def test_Bfield_singularity():
    # Test the result for a field sample on the dipole itself
    # Expected: NaN
    from numpy import array

    # Definitions
    mag=array([-1,2,-3])
    calcPos = array([0,0,0])
    
    # Run
    with pytest.warns(RuntimeWarning):
        results = Bfield_Dipole(mag,calcPos)
        assert all(isnan(axis) for axis in results)

def test_Bfield_outside():
    # Fundamental Positions in every 8 Octants
    errMsg = "Field sample outside of Box is unexpected"
    mockResults = [ [-20.105974, -24.142649, -5.059827],
                    [15.050978, 16.020022, -4.835365],
                    [-10.051157, 11.206562, -3.526921],
                    [6.41919, -7.423334, -0.818756],
                    [19.933508, -17.961747, 9.301348],
                    [-15.331571, 12.868214, 2.721131],
                    [10.209707, 8.129991, 2.130329],
                    [-6.319435, -6.356132, 1.677024],]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],
                        [-8,9,10],[7,-6,-5],[-8,7,-6],
                        [-9,-8,7],[-10,-9,-8]])                  
    
    #check field values to be within [1,100] adjust magnetization

    
    mag=array([-11111,22222,-333333])

    results=[Bfield_Dipole(mag,pos) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
