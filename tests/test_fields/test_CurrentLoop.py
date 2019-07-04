from magpylib._lib.fields.Current_CircularLoop import Bfield_CircularCurrentLoop
from numpy import array, isnan
import pytest

def test_Bfield_singularity():
    # Test the result for a field sample on the circular loop
    # Expected: NaN
    from numpy import array
    
    # Definitions
    current = 5
    diameter = 5
    calcPos = [0,2.5,0]

    # Run
    with pytest.warns(RuntimeWarning):
        results = Bfield_CircularCurrentLoop(current,diameter,calcPos)
        assert all(isnan(axis) for axis in results)


def test_CircularGetB_OuterLines():
    errMsg = "Results from getB are unexpected"
    mockResults = [ [-0.0, -51.469971, 30.236739],
                    [0.0, 51.469971, 30.236739],
                    [0.0, 63.447185, -2.422005],
                    [0.0, -63.447185, -2.422005],
                    [-0.0, -63.447185, -2.422005],
                    [-0.0, 63.447185, -2.422005],
                    [-0.0, -36.0076, 68.740859],
                    [0.0, 36.0076, 68.740859],
                    [0.0, 0.0, -133.812661], ] 

    sideSurface = [0,3,0]
    upperSurface = [0,1.5,1.5]
    lowerSurface = [0,1.5,-1.5]
    edgeBot = [0,2.5,1.5]
    edgeTop = [0,2.5,-1.5]
    edge = [0,3,1]
    edge2 = [0,-3,1]
    edge3 = [0,3,-1]
    edge4 = [0,-3,-1]

    testPosOut = array([  edgeTop, edgeBot, edge, edge2,
                          edge3, edge4, lowerSurface,upperSurface,
                        sideSurface])                  

    current = 500
    diameter = 5

    results=[Bfield_CircularCurrentLoop(current,diameter,pos) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_Bfield_outside():
    # Fundamental Positions in every 8 Octants
    errMsg = "Field sample outside of Box is unexpected"
    mockResults = [ [-28.275526, -30.846029, -8.08718],
                    [18.54694, 21.638096, -5.704383],
                    [-12.588134, 14.386439, -3.348427],
                    [8.919783, -10.034756, -2.091007],
                    [29.112211, -24.953324, 9.416569],
                    [-18.649955, 16.318711, 5.173529],
                    [12.635187, 11.231277, 3.070725],
                    [-8.943273, -8.048946, 1.934808],
                    [0.0, 0.0, -69813.100267],]

    testPosOut = array([ [5.5,6,7],[6,7,-8],[7,-8,9],
                         [-8,9,10],[7,-6,-5],[-8,7,-6],
                         [-9,-8,7],[-10,-9,-8],[0,0,0] ])                  
    
    #check field values to be within [1,100] adjust magnetization
    current = -111111
    diameter = 2
    results=[Bfield_CircularCurrentLoop(current,diameter,pos) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
