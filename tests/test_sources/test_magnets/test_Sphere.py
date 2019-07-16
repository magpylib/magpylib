from magpylib.source.magnet import Sphere
from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_SphereZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Sphere(mag=[0,0,0],dim=1)

def test_SphereZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Sphere(mag=[1,1,1],dim=0)

def test_SphereGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([-0.05040102, -0.05712116, -0.03360068]) ## Expected 3 results for this input
    
    # Input
    mag=[6,7,8]
    dim=2
    pos=[2,2,2]
    fieldPos = [.5,.5,5]

    # Run
    pm = magnet.Sphere(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_SphereGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = (-0.00047774, -0.00535384, -0.00087997) ## Expected 3 results for this input

    # Input
    mag=(0.2,32.5,5.3)
    dim=1
    pos=(1,0.2,3)
    axis=[0.2,.61,1]
    angle=89
    fieldPos=[5,5,.35]

    # Run
    pm = magnet.Sphere(mag,dim,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_Box_rotation_GetB():
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
    mockResults = [ [0.057055, -0.081669, 0.557462],
                    [0.003315, 0.004447, 0.004952],
                    [0.002031, 0.006438, 0.008955],
                    [0.145866, -0.118869, 0.125958],
                    [0.023151, -0.029514, 0.013329],
                    [-0.001438, 0.003102, 0.009031],]
    mag=[6,7,8]
    dim=2
    pos=[2,2,2]
    pm = magnet.Sphere(mag,dim,pos)
    results = [applyRotAndReturnB(arg,pm) for arg in arguments]

    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
    
def test_BoxGetB_rotation():
    erMsg = "Results from getB are unexpected"
    from numpy import pi

    def applyRotationAndReturnStatus(arg,obj):
        obj.rotate(arg[0],arg[1],arg[2])
        result = {  "mag": obj.magnetization, 
                    "pos": obj.position, 
                    "ang": obj.angle, 
                    "axi": obj.axis, 
                    "dim": obj.dimension}
        return result

    arguments = [ [36,(1,2,3),(.1,.2,.3)],
                  [-366, [3,-2,-1],[-.5,-.6,.7]],
                  [pi, [0,0,0.0001],[-.5,-2,.7]]]
    mockResults = [     {'mag': array([6., 7., 8.]), 'pos': array([1.46754927, 2.57380229, 1.79494871]), 'ang': 36.00000000000002, 
                         'axi': array([0.26726124, 0.53452248, 0.80178373]), 'dim': 2.0},
                        {'mag': array([6., 7., 8.]), 'pos': array([1.4274764 , 2.70435404, 1.41362661]), 'ang': 321.8642936876839, 
                         'axi': array([-0.14444227, -0.62171816, -0.76980709]), 'dim': 2.0},
                        {'mag': array([6., 7., 8.]), 'pos': array([1.16676385, 2.80291687, 1.41362661]), 'ang': 319.3981749889049, 
                         'axi': array([-0.11990803, -0.58891625, -0.79924947]), 'dim': 2.0},]
    mag = [6,7,8]
    dim = 2
    pos = [2,2,2]
    pm = magnet.Sphere(mag,dim,pos)

    results = [applyRotationAndReturnStatus(arg,pm,) for arg in arguments]
    print(results)
    rounding = 4 ## Round for floating point error 
    for i in range(0,len(mockResults)):
        for j in range(3):
            assert round(results[i]['mag'][j],rounding)==round(mockResults[i]['mag'][j],rounding), erMsg
        for j in range(3):
            assert round(results[i]['axi'][j],rounding)==round(mockResults[i]['axi'][j],rounding), erMsg
        for j in range(3):
            assert round(results[i]['pos'][j],rounding)==round(mockResults[i]['pos'][j],rounding), erMsg

        assert round(results[i]['ang'],rounding)==round(mockResults[i]['ang'],rounding), erMsg


def test_SphereMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( (-0.00047774, -0.00535384, -0.00087997), ## Expected results for this input
                    (-0.00047774, -0.00535384, -0.00087997),
                    (-0.00047774, -0.00535384, -0.00087997),)

    mag=(0.2,32.5,5.3)
    dim=1
    pos=(1,0.2,3)
    axis=[0.2,.61,1]
    angle=89
    arrayOfPos =array ([  [5,5,.35],
                [5,5,.35],
                [5,5,.35],])

    # Run
    pm = magnet.Sphere(mag,dim,pos,angle,axis)
    result = pm.getBsweep(arrayOfPos  ) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

def test_ToString():
    magnetization=(0.2,32.5,5.3)
    dimension=1.0
    position=(1.0,0.2,3.0)
    axis=[0.2,.61,1.0]
    angle=89.0
    expected="type: {} \n magnetization: x: {}, y: {}, z: {}mT \n dimensions: d: {} \n position: x: {}, y:{}, z: {} \n angle: {} Degrees \n axis: x: {}, y: {}, z:{}".format("magnet.Sphere", *magnetization, dimension, *position, angle, *axis)

    mySphere = Sphere(magnetization, dimension, position, angle, axis)

    result = mySphere.__repr__()
    assert result == expected 