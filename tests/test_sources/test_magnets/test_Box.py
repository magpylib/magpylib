from magpylib.source import magnet
from numpy import isnan, array
import pytest 

def test_BoxZeroMagError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[0,0,0],dim=[1,1,1])

def test_BoxZeroDimError():
    with pytest.raises(AssertionError):
        magnet.Box(mag=[1,1,1],dim=[0,0,0])


def test_BoxEdgeCase_rounding():
    ## For now this returns NaN, may be an analytical edge case
    ## Test the Methods in getB() before moving onto this
    expectedResult = [ 1.90833281e-12, -4.06404209e-13,  5.72529193e-09]
    pm = magnet.Box(mag=[0,0,1000],dim=[0.5,0.1,1],pos=[.25,.55,-1111])
    result = pm.getB([.5,.5,5])
    rounding=10
    for i in range(0,3):
        assert round(result[i],rounding)==round(expectedResult[i],rounding), "Rounding edge case is wrong"
        

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
    mockResults = [ [3.042604, 5.947039, 4.764305],
                    [0.868926, 1.003817, 1.445178],
                    [0.500882, 1.20951, 2.481127],
                    [0.471373, 6.661388, 4.280969],
                    [-3.444099, 2.669495, -6.237409],
                    [0.004294, 0.952229, 2.079012],]
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    pm = magnet.Box(mag,dim,pos)
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
    mockResults = [{'mag': array([6., 7., 8.]), 'pos': array([1.46754927, 2.57380229, 1.79494871]),
                    'ang': 36.00000000000002, 'axi': array([0.26726124, 0.53452248, 0.80178373]), 
                    'dim': array([10., 10., 10.])}, 
                   {'mag': array([6., 7., 8.]), 'pos': array([1.4274764 , 2.70435404, 1.41362661]), 
                    'ang': 321.8642936876839, 'axi': array([-0.14444227, -0.62171816, -0.76980709]), 
                    'dim': array([10., 10., 10.])}, 
                    {'mag': array([6., 7., 8.]), 'pos': array([1.16676385, 2.80291687, 1.41362661]), 
                    'ang': 319.3981749889049, 'axi': array([-0.11990803, -0.58891625, -0.79924947]), 
                    'dim': array([10., 10., 10.])}]
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]
    pm = magnet.Box(mag,dim,pos)

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

def test_BoxGetB():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 3.99074612, 4.67238469, 4.22419432) ## Expected 3 results for this input

    # Input
    mag=[6,7,8]
    dim=[10,10,10]
    pos=[2,2,2]



    # Run
    pm = magnet.Box(mag,dim,pos)
    result = pm.getB([.5,.5,5])

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_BoxGetBAngle():
    erMsg = "Results from getB are unexpected"
    mockResults = ( 0.08779447,  0.0763171,  -0.11471596 ) ## Expected 3 results for this input

    # Input
    mag=(0.2,32.5,5.3)
    dim=(1,2.4,5)
    pos=(1,0.2,3)
    axis=[0.2,1,0]
    angle=90
    fieldPos=[5,5,.35]

    # Run
    pm = magnet.Box(mag,dim,pos,angle,axis)
    result = pm.getB(fieldPos)

    rounding = 4 ## Round for floating point error 
    for i in range(3):
        assert round(result[i],rounding)==round(mockResults[i],rounding), erMsg

def test_BoxMulticoreGetB():
    erMsg = "Results from getB are unexpected"
    pm = magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])
    pos = array([(.5,.5,5),(30,20,10),(1,.2,60)])
    ## Positions list
    result = pm.getBsweep(pos) 


    ## Expected Results
    mockRes = ( ( 3.99074612, 4.67238469, 4.22419432), # .5,.5,.5
                ( 0.03900578,  0.01880832, -0.00134112), # 30,20,10
                ( -0.00260347, -0.00313962,  0.00610886), ) 

    ## Rounding for floating point error 
    rounding = 4 

    # Loop through predicted cases and check if the positions from results are valid
    for i in range(len(mockRes)):
        for j in range(3):
            assert round(result[i][j],rounding)==round(mockRes[i][j],rounding), erMsg

def test_ToString():
    magnetization=(0.2,32.5,5.3)
    dimension=(1.0,2.4,5.0)
    position=(1.0,0.2,3.0)
    axis=[0.2,1.0,0.0]
    angle=90.0
    expected="type: {} \n magnetization: x: {}, y: {}, z: {} \n dimensions: a: {}, b: {}, c: {} \n position: x: {}, y:{}, z: {} \n angle: {} Degrees \n axis: x: {}, y: {}, z:{}".format("magnet.Box", *magnetization, *dimension, *position, angle, *axis)

    myBox = magnet.Box(magnetization, dimension, position, angle, axis)

    result = myBox.__repr__()
    assert result == expected