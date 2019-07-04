from magpylib._lib.fields.PM_Cylinder import Bfield_Cylinder
from numpy import array, isnan
import pytest

def test_Bfield_singularity():
    # Test the result for a field sample on the Cylinder
    # Circular top face, Circular lower face, Side face
    # Expected: [nan,nan,nan]
    from magpylib import source,Collection
    from numpy import array
    
    # Definitions
    iterDia = 50
    mag = [1,2,3]
    dim = [5,2]
    sideSurface = [0,2.5,0]
    upperSurface = [0,1.5,1]
    lowerSurface = [0,1.5,-1]
    edge = [0,2.5,1]
    testPos = [sideSurface,upperSurface,lowerSurface,edge]

    # Run
    with pytest.warns(RuntimeWarning):
        results = [Bfield_Cylinder(mag,pos,dim,iterDia) for pos in testPos]
        assert all(all(isnan(axis) for axis in result) for result in results)

def test_Bfield_OuterPlanes():
    # Field samples that are coplanar with samples that lead to singularity 
    # These should be fine
    errMsg = "Unexpected Results for Cylinder getB in Outer Lines"
    mockResults = [ [253.334094, -4272.910608, 5590.91282],
                    [253.334094, 5986.100534, 4906.984899],
                    [253.334094, 5986.100534, 4906.984899],
                    [253.334094, -4272.910608, 5590.91282],
                    [1111.337474, 45438.762267, 3550.174662],
                    [1111.337474, -40140.199172, 9255.362182],
                    [283.635853, 1039.317535, 7080.751148],
                    [344.814448, -9923.631738, 4923.14504],
                    [344.814448, 11862.477551, 3470.750825],]

    sideSurface = [0,3,0]
    upperSurface = [0,1.5,1.5]
    lowerSurface = [0,1.5,-1.5]
    edge = [0,3,1]
    edge2 = [0,-3,1]
    edge3 = [0,3,-1]
    edge4 = [0,-3,-1]
    edgeBot = [0,2.5,1.5]
    edgeTop = [0,2.5,-1.5]

    testPosOut = array([ edge, edge2,edge3,
                        edge4,lowerSurface,upperSurface,
                        sideSurface,edgeBot,edgeTop])                              

    mag=array([-11111,22222,-333333])

    a,b = 2,3
    dim=array([a,b])
    iterDia = 50 #  Iterations calculating B-field from non-axial magnetization
    rounding = 4
    results = [Bfield_Cylinder(mag,pos,dim,iterDia) for pos in testPosOut]
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg


def test_Bfield_outside():
    # Fundamental Positions in every 8 Octants
    errMsg = "Field sample outside of Cylinder is unexpected"
    mockResults = [ [-189.256324, -227.431954, -43.180409],
                    [141.959926, 151.027113, -43.199818],
                    [-94.766897, 105.650499, -32.010555],
                    [60.407258, -69.865622, -7.040305],
                    [184.386143, -166.307916, 90.330253],
                    [-143.126907, 120.076143, 27.304572],
                    [95.526535, 75.994148, 20.934673],
                    [-59.166487, -59.557521, 16.283876],]

    testPosOut = array([[5.5,6,7],[6,7,-8],[7,-8,9],
                        [-8,9,10],[7,-6,-5],[-8,7,-6],
                        [-9,-8,7],[-10,-9,-8]])                  
    
    #check field values to be within [1,100] adjust magnetization

    mag=array([-11111,22222,-333333])
    
    a,b = 2,3
    dim=array([a,b])
    iterDia = 50 #  Iterations calculating B-field from non-axial magnetization
    results=[Bfield_Cylinder(mag,pos,dim,iterDia) for pos in testPosOut]
    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg

def test_Bfield_inside():
    # Fundamental Positions in every 8 Octants, but inside
    from numpy import pi
    errMsg = "Field sample inside of Cylinder is unexpected"
    mockResults = [ [-6488.54459, 12977.08918, -277349.820763],
                    [-15392.748076, 350.405436, -269171.80507],
                    [2099.31683, 199.530241, -268598.798536],
                    [-15091.053488, 25985.080376, -270890.824672],
                    [1797.602656, 26135.931534, -270317.818138],
                    [2099.297244, 199.554278, -268598.798536],
                    [-15091.033902, 25985.056339, -270890.824672],
                    [1797.599514, 26135.911134, -270317.818138],
                    [-15392.751218, 350.385036, -269171.80507],]

    mag=array([-11111,22222,-333333])
    a,b,c = 2,3,4
    dim=a,b
    testPosIn = array([[0,0,0],[a,b,c],[-a,b,c],[a,-b,c],[a,b,-c],
                       [a,-b,-c],[-a,b,-c],[-a,-b,c],[-a,-b,-c]])/(2*pi)
    iterDia = 50 #  Iterations calculating B-field from non-axial magnetization
    results=[Bfield_Cylinder(mag,pos,dim,iterDia) for pos in testPosIn]
    rounding = 4

    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg


    

