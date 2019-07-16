from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_CylinderZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Cylinder(mag=(0,0,0),dim=(1,1))

def test_CylinderZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Cylinder(mag=(1,1,1),dim=(0,0))

def test_Cylinder_rotation_GetB(): 
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
    mockResults = [ [2.541652, 7.258253, 2.985087],
                    [0.287798, 0.311855, 0.470142],
                    [0.212632, 0.382307, 0.787474],
                    [2.16498, -1.070064, 1.745844],
                    [-5.858558, 1.331839, -6.035032],
                    [-0.173693, 0.283437, 0.999618],]
    mag=[6,7,8]
    dim=[10,5]
    pos=[2,2,2]
    pm = magnet.Cylinder(mag,dim,pos)
    results = [applyRotAndReturnB(arg,pm) for arg in arguments]

    rounding = 4
    for i in range(0,len(mockResults)):
        for j in range(0,3):
            assert round(mockResults[i][j],rounding)==round(results[i][j],rounding), errMsg
    
def test_Cylinder_GetB_rotation():
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
    mockResults = [ {'mag': array([6., 7., 8.]), 'pos': array([1.46754927, 2.57380229, 1.79494871]), 
                     'ang': 36.00000000000002, 'axi': array([0.26726124, 0.53452248, 0.80178373]), 
                     'dim': array([10.,  5.])},
                    {'mag': array([6., 7., 8.]), 'pos': array([1.4274764 , 2.70435404, 1.41362661]), 
                     'ang': 321.8642936876839, 'axi': array([-0.14444227, -0.62171816, -0.76980709]), 
                     'dim': array([10.,  5.])},
                    {'mag': array([6., 7., 8.]), 'pos': array([1.16676385, 2.80291687, 1.41362661]), 
                     'ang': 319.3981749889049, 'axi': array([-0.11990803, -0.58891625, -0.79924947]), 
                     'dim': array([10.,  5.])},]
    mag=[6,7,8]
    dim=[10,5]
    pos=[2,2,2]
    pm = magnet.Cylinder(mag,dim,pos)

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

def test_CylinderGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = array([ 0.62431573,  0.53754927, -0.47024376]) ## Expected results for this input

    # Input 
    mag=(6,7,8)
    dim=(2,9)
    pos=(2,2,2)
    fieldPos = (.5,.5,5)

    # Run
    pm = magnet.Cylinder(mag,dim,pos)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_CylinderGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 0.01576884,  0.01190684, -0.01747232 ) ## Expected 3 results for this input

    # Input
    mag=(0.2,32.5,5.3)
    dim=(1,2.4)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    fieldPos=[5,5,.35]

    # Run
    pm = magnet.Cylinder(mag,dim,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg


def test_CylinderMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( array([ 0.62431573,  0.53754927, -0.47024376]), ## Expected results for this input
                    array([ 0.62431573,  0.53754927, -0.47024376]),
                    array([ 0.62431573,  0.53754927, -0.47024376]),)
    mag=(6,7,8)
    dim=(2,9)
    pos=(2,2,2)
    arrayPos = array([[.5,.5,5],
                [.5,.5,5],
                [.5,.5,5]])

    pm = magnet.Cylinder(mag,dim,pos)

    ## Positions list
    result = pm.getBsweep(arrayPos ) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockResults)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockResults[i][j],rounding), erMsg

def test_ToString():
    magnetization=(0.2,32.5,5.3)
    dimension=(2.0,9.0)
    position=(1.0,0.2,3.0)
    axis=[0.2,1.0,0.0]
    angle=90.0
    expected="type: {} \n magnetization: x: {}, y: {}, z: {} \n dimensions: d: {}, h: {} \n position: x: {}, y:{}, z: {} \n angle: {} \n axis: x: {}, y: {}, z:{}".format("magnet.Cylinder", *magnetization, *dimension, *position, angle, *axis)

    myCylinder = magnet.Cylinder(magnetization, dimension, position, angle, axis)

    result = myCylinder.__repr__()
    assert result == expected