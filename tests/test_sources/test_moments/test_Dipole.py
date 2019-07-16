from magpylib.source.moment import Dipole
from numpy import isnan, array
import pytest 


def test_Dipole_rotation_GetB():
    errMsg = "Results from getB are unexpected"
    from numpy import pi
    # Setup
    def applyRotAndReturnB(arg,obj):
        obj.rotate(arg[1],arg[2],arg[3])
        return obj.getB(arg[0])

    arguments = [[[2,3,4],36,(1,2,3),(.1,.2,.3)], #Position, Angle, Axis, Anchor
                [[-3,-4,-5],-366, [3,-2,-1],[-.5,-.6,.7]],
                [[-1,-3,-5],pi, [0,0,0.0001],[-.5,-2,.7]],
                [[2,3,4], 36, [1,2,3],[.1,.2,.3]],
                [[-3,-4,-5], -124, [3,-2,-1],[-.5,-.6,.7]],
                [[-1,-3,-5], 275, [-2,-2,4],[0,0,0]] ]
    mockResults = [ [0.013621, -0.019497, 0.133084],
                    [0.000791, 0.001062, 0.001182],
                    [0.000485, 0.001537, 0.002138],
                    [0.034823, -0.028378, 0.03007],
                    [0.005527, -0.007046, 0.003182],
                    [-0.000343, 0.000741, 0.002156],]
    mag=[6,7,8]
    pos=[2,2,2]
    pm = Dipole(mag,pos)
    results = [applyRotAndReturnB(arg,pm) for arg in arguments]

    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
    
def test_DipoleGetB_rotation():
    erMsg = "Results from getB are unexpected"
    from numpy import pi

    def applyRotationAndReturnStatus(arg,obj):
        obj.rotate(arg[0],arg[1],arg[2])
        result = {  "mom": obj.moment, 
                    "pos": obj.position, 
                    "ang": obj.angle, 
                    "axi": obj.axis, }
        return result

    arguments = [ [36,(1,2,3),(.1,.2,.3)],
                  [-366, [3,-2,-1],[-.5,-.6,.7]],
                  [pi, [0,0,0.0001],[-.5,-2,.7]]]
    mockResults = [{'mom': array([6., 7., 8.]), 'pos': array([1.46754927, 2.57380229, 1.79494871]), 
                    'ang': 36.00000000000002, 'axi': array([0.26726124, 0.53452248, 0.80178373])},
                   {'mom': array([6., 7., 8.]), 'pos': array([1.4274764 , 2.70435404, 1.41362661]), 
                    'ang': 321.8642936876839, 'axi': array([-0.14444227, -0.62171816, -0.76980709])},
                   {'mom': array([6., 7., 8.]), 'pos': array([1.16676385, 2.80291687, 1.41362661]), 
                    'ang': 319.3981749889049, 'axi': array([-0.11990803, -0.58891625, -0.79924947])},]
    mag=[6,7,8]
    pos=[2,2,2]
    pm = Dipole(mag,pos)

    results = [applyRotationAndReturnStatus(arg,pm,) for arg in arguments]
    print(results)
    rounding = 4 ## Round for floating point error 
    for i in range(0,len(mockResults)):
        for j in range(3):
            assert round(results[i]['mom'][j],rounding)==round(mockResults[i]['mom'][j],rounding), erMsg
        for j in range(3):
            assert round(results[i]['axi'][j],rounding)==round(mockResults[i]['axi'][j],rounding), erMsg
        for j in range(3):
            assert round(results[i]['pos'][j],rounding)==round(mockResults[i]['pos'][j],rounding), erMsg

        assert round(results[i]['ang'],rounding)==round(mockResults[i]['ang'],rounding), erMsg


def test_DipoleGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 1.23927518e-06,  6.18639685e-06, -1.67523560e-06]) ## Expected 3 results for this input

    # Input
    moment=[5,2,10]
    pos=(24,51,22)
    fieldPos = (.5,.5,5)

    # Run
    pm = Dipole(moment,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_DipoleGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = (-0.00836643 , 0.01346,   -0.01833964) ## Expected 3 results for this input

    # Input
    moment=(0.2,32.5,5.3)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    fieldPos=[.5,5,.35]

    # Run
    pm = Dipole(moment,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_DipoleMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = (     (-0.00836643 , 0.01346,   -0.01833964), ## Expected 3 results for this input
                        (-0.00836643 , 0.01346,   -0.01833964),
                        (-0.00836643 , 0.01346,   -0.01833964)) 

    # Input
    moment=(0.2,32.5,5.3)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    arrayOfPos =  array([[.5,5,.35],
                [.5,5,.35],
                [.5,5,.35],])

    # Run
    pm = Dipole(moment,pos,angle,axis)
    ## Positions list
    result = pm.getBsweep(arrayOfPos) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

def test_ToString():
    moment=(0.2,32.5,5.3)
    position=(1.0,0.2,3.0)
    axis=[0.2,1.0,0.0]
    angle=90.0
    expected="type: {} \n moment: x: {}, y: {}, z: {} \n position: x: {}, y: {}, z:{} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format("moments.Dipole", *moment, *position, angle, *axis)

    myDipole = Dipole(moment, position, angle, axis)

    result = myDipole.__repr__()
    assert result == expected