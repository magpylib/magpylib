import unittest
from magpylib._lib.classes.collection import Collection
from magpylib._lib.classes.base import RCS
from numpy import array, ndarray
import pytest

def test_motion():
    # Check if rotate() and move() are
    # behaving as expected for Collection
    expectedAngles = { "box1": 90,
                       "box2": 90 }
    expectedPositions = { "box1": [-4,  2,  6],
                          "box2": [-4,  4,  4] }

    from magpylib import source,Collection
    box = source.magnet.Box([1,2,3],[1,2,3],[1,2,3])
    # box.position # [1,2,3]
    # box.angle # 0
    box2 = source.magnet.Box([1,2,3],[1,2,3],[3,2,1])
    # box2.position # [3,2,1]
    # box2.angle # 0


    col = Collection(box,box2)
    col.move([1,2,3])
    col.rotate(90,(0,0,1),[0,0,0])
    assert box.angle == expectedAngles["box1"]
    assert all(round(box.position[i]) == expectedPositions["box1"][i] for i in range(0,3))
    assert box2.angle == expectedAngles["box2"]
    assert all(round(box2.position[i]) == expectedPositions["box2"][i] for i in range(0,3))

def test_addSource_Duplicate():
    # Check if addSource is  throwing a warning
    # and ignoring duplicates

    errMsg = "Duplicate/Extra copy on collection detected"
    from magpylib import source
    # Setup
    mag=(2,3,5)
    dim=(2,2,2)

    # Run
    with pytest.warns(Warning):
        b0 = source.magnet.Box(mag,dim)
        col = Collection() 
        col.addSources([b0,b0])
        assert len(col.sources) == 1, errMsg

def test_addSource_Duplicate_force():
    # Check if addSource is NOT throwing a warning
    # and NOT ignoring duplicates when ignore kwarg is given
    errMsg = "Threw a warning on false statement"
    errMsg_list = "Extra copy on collection not detected"
    from magpylib import source
    import warnings
    # Setup
    mag=(2,3,5)
    dim=(2,2,2)

    # Run
    with pytest.warns(Warning) as record:
        warnings.warn("Ok so far, check if this is the only warning.", RuntimeWarning)
        b0 = source.magnet.Box(mag,dim)
        col = Collection() 
        col.addSources(b0,b0, dupWarning = False)
        assert len(record) == 1, errMsg
        assert len(col.sources) == 2, errMsg_list

def test_initialization_Duplicate():
    # Check if initialization is  throwing a warning
    # and ignoring duplicates
    from magpylib import source
    # Setup
    mag=(2,3,5)
    dim=(2,2,2)

    # Run
    with pytest.warns(Warning):
        b0 = source.magnet.Box(mag,dim)
        col = Collection( b0,b0) 
        assert len(col.sources) == 1

def test_initialization_Duplicate_force():
    # Check if addSource is NOT throwing a warning
    # and NOT ignoring duplicates when ignore kwarg is given
    errMsg = "Threw a warning on false statement"
    errMsg_list = "Extra copy on collection not detected"
    from magpylib import source
    import warnings
    # Setup
    mag=(2,3,5)
    dim=(2,2,2)

    # Run
    with pytest.warns(Warning) as record:
        warnings.warn("Ok so far, check if this is the only warning.", RuntimeWarning)
        b0 = source.magnet.Box(mag,dim)
        col = Collection( b0,b0, dupWarning = False) 
        assert len(record) == 1, errMsg
        assert len(col.sources) == 2, errMsg_list

def test_removeSource():
    # Check if removeSource is removing the provided source 
    # and the last added source to the Collection
    from magpylib import source
    mag=(2,3,5)
    dim=(2,2,2)

    case = unittest.TestCase()
    ## Define boxes in different position of memory
    b0 = source.magnet.Box(mag,dim)
    b1 = source.magnet.Box(mag,dim)
    b2 = source.magnet.Box(mag,dim)
    b3 = source.magnet.Box(mag,dim)
    b4 = source.magnet.Box(mag,dim)
    b5 = source.magnet.Box(mag,dim)
    b6 = source.magnet.Box(mag,dim)
    allBoxes = [b0,b1,b2,b3,b4,b5,b6]
    removedSet = [b1,b2,b3,b4,b5]

    # Run
    col1 = Collection(allBoxes)
    col1.removeSource(b0)
    col1.removeSource()
    removedSet = [b1,b2,b3,b4,b5]
    case.assertCountEqual(removedSet,col1.sources)


def test_removeSource_no_index_error():
    # Check if removeSource is throwing an error for
    # removing the an idnex that is not there.
    from magpylib import source
    mag=(2,3,5)
    dim=(2,2,2)

    with pytest.raises(IndexError):
        ## Define boxes in different position of memory
        b0 = source.magnet.Box(mag,dim)
        # Run
        col1 = Collection(b0)
        col1.removeSource(2)

def test_removeSource_no_source_error():
    # Check if removeSource is throwing an error for
    # removing the a source that is not there.
    from magpylib import source
    mag=(2,3,5)
    dim=(2,2,2)

    with pytest.raises(ValueError):
        ## Define boxes in different position of memory
        b0 = source.magnet.Box(mag,dim)
        b1 = source.magnet.Box(mag,dim)
        b2 = source.magnet.Box(mag,dim)
        # Run
        col1 = Collection(b0,b1)
        col1.removeSource(b2)
def test_initialization():
    # Check if initialization accepts mixed arguments
    # source objects, list, collection
    case = unittest.TestCase()
    from magpylib import source
    mag=(2,3,5)
    dim=(2,2,2)

    ## Define boxes in different position of memory
    b0 = source.magnet.Box(mag,dim)
    b1 = source.magnet.Box(mag,dim)
    b2 = source.magnet.Box(mag,dim)
    b3 = source.magnet.Box(mag,dim)
    b4 = source.magnet.Box(mag,dim)
    b5 = source.magnet.Box(mag,dim)
    b6 = source.magnet.Box(mag,dim)
    allBoxes = [b0,b1,b2,b3,b4,b5,b6]
    col1 = Collection( b0,b1 )
    col2 = Collection([b2,b3])
    col3 = Collection(col1,col2,b4,[b5,b6])
    
    ## Check if all items are in the col3 list, regardless of order
    case.assertCountEqual(allBoxes,col3.sources)

def test_AddSource_mix():
    # Check if addSource method accepts mixed arguments
    # source objects, list, collection
    case = unittest.TestCase()
    from magpylib import source
    mag=(2,3,5)
    dim=(2,2,2)

    ## Define boxes in different position of memory
    b0 = source.magnet.Box(mag,dim)
    b1 = source.magnet.Box(mag,dim)
    b2 = source.magnet.Box(mag,dim)
    b3 = source.magnet.Box(mag,dim)
    b4 = source.magnet.Box(mag,dim)
    b5 = source.magnet.Box(mag,dim)
    b6 = source.magnet.Box(mag,dim)
    allBoxes = [b0,b1,b2,b3,b4,b5,b6]
    col1 = Collection()
    col1.addSources(b0,b1)
    col2 = Collection()
    col2.addSources([b2,b3])
    col3 = Collection()
    col3.addSources(col1,col2,b4,[b5,b6])

    ## Check if all items are in the col3 list, regardless of order
    case.assertCountEqual(allBoxes,col3.sources)

def test_GetB():
    # Check if getB is being called correctly,
    # getB in collection superimposes (adds) all the fields
    # generated by the objects in the collection.
    from magpylib._lib.classes.magnets import Box

    #Input
    mockList = (    array([0.12488298, 0.10927261, 0.07805186]),
                    array([0.12488298, 0.10927261, 0.07805186]))
    mockResult = sum(mockList)

    mag=(2,3,5)
    dim=(2,2,2)
    pos=[2,2,2]

    #Run   
    b = Box(mag,dim)
    b2 = Box(mag,dim)
    c = Collection(b,b2)
    result = c.getB(pos) 

    rounding = 4
    for j in range(3):
        assert round(result[j],rounding) == round(mockResult[j],rounding)

def test_GetB_markerInput():
    # Check if marker inputs are acceptable
    # TODO: find out a good way to compare plot results.
    # Have output plot data saved in text maybe
    # Just test input validity for now.
    from magpylib._lib.classes.magnets import Box
    b = Box([1,1,1],[1,1,1],pos=(5,5,5))

    a = Collection()
    markers = [[0,0,0],    # int
               [.1,.1,.1], # float
               b.position] # float64

    a.displaySystem(markers=markers,suppress=True)

def test_sensorDraw():
    # Check if marker inputs are acceptable
    # TODO: find out a good way to compare plot results.
    # Have output plot data saved in text maybe
    # Just test input validity for now.
    from magpylib._lib.classes.magnets import Box
    from magpylib._lib.classes.sensor import Sensor
    b = Box([1,1,1],[1,1,1],pos=(5,5,5))

    a = Collection()
    sensor = Sensor()

    a.displaySystem(sensors=[sensor],suppress=True)
    from matplotlib import pyplot

def test_GetBList():
    # Check if sole getB throws an error
    # for Lists.
    from magpylib._lib.classes.magnets import Box
    #Input
    mockList = (    array([0.12488298, 0.10927261, 0.07805186]),
                    array([0.12488298, 0.10927261, 0.07805186]),)
    mockResult = [  sum(mockList),
                    sum(mockList)]

    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([   [2,2,2],
                [2,2,2]])


    with pytest.raises(IndexError):
    #Run   
        b = Box(mag,dim)
        b2 = Box(mag,dim)
        c = Collection(b,b2)
        result = c.getB(pos) 

        rounding = 4
        for i in range(len(mockResult)):
            for j in range(3):
                assert round(result[i][j],rounding) == round(mockResult[i][j],rounding)

def test_GetBSweep_displacement_error():
    # Check if getBsweep throws an error
    # if a displacement input is provided.
    from magpylib._lib.classes.magnets import Box
    #Input
    erMsg = "Results from getB are unexpected"
    type_erMsg =  "getBsweep did not return a numpy array."
    
    mockResults = array((   [ 0.00453617, -0.07055326,  0.03153698],
                            [0.00488989, 0.04731373, 0.02416068],
                            [0.0249435,  0.00106315, 0.02894469]))
    
    # Input
    mag=[1,2,3]
    dim=[1,2,3]
    pos=[0,0,0]

    listOfArgs = [  [   [1,2,3],        #pos
                        [0,0,1],        #MPos
                        (180,(0,1,0)),],#Morientation
                    [   [1,2,3],
                        [0,1,0],
                        (90,(1,0,0)),],
                    [   [1,2,3],
                        [1,0,0],
                        (255,(0,1,0)),],]
                    

    # Run
    pm = Box(mag,dim,pos)

    with pytest.raises(AssertionError):
    #Run   
        c = Collection(pm)
        result = c.getBsweep(listOfArgs) 
        assert isinstance(result,ndarray), type_erMsg
        rounding = 4
        for i in range(len(mockResults)):
            for j in range(3):
                assert round(result[i][j],rounding) == round(mockResults[i][j],rounding),erMsg

def test_AddList():
    # Check if adding a list of generics to collection 
    # does not throw an error
    errMsg = "Failed to place items in collection, got: "
    from magpylib._lib.classes.magnets import Box
    def boxFactory():
        mag=(2,3,5)
        dim=(2,2,2)
        return Box(mag,dim)
    listSize = 5
    boxList = list(boxFactory() for a in range(0,listSize))

    c = Collection(boxList)
    lenOfCol = len(c.sources)
    assert lenOfCol == listSize, errMsg + str(lenOfCol) + "; expected: " + str(listSize)

def test_GetBSweepList():
    # Check if getB sweep is performing multipoint
    # calculations sequentially
    from magpylib._lib.classes.magnets import Box
    type_erMsg =  "getBsweep did not return a numpy array."
    #Input
    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([     [2,2,2],
                    [2,2,2],
                    [2,2,3]])


    mockResult = [  [0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],]
    #Run   
    b = Box(mag,dim)
    b2 = Box(mag,dim)
    c = Collection(b,b2)
    result = c.getBsweep(pos,multiprocessing=False)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 4
    for i in range(len(mockResult)):
        for j in range(3):
            assert round(result[i][j],rounding) == round(mockResult[i][j],rounding)

def test_GetBSweepList_multiprocessing_many_positions_few_objects():
    # Check if getB sweep is performing multipoint
    # calculations utilizing multiple processes

    from magpylib._lib.classes.magnets import Box
    type_erMsg =  "getBsweep did not return a numpy array."
    #Input
    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([     [2,2,2],
                    [2,2,2],
                    [2,2,3],
                    [2,2,2],
                    [2,2,2],
                    [2,2,3]])


    mockResult = [  [0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],]
    #Run   
    b = Box(mag,dim)
    b2 = Box(mag,dim)
    c = Collection(b,b2)
    result = c.getBsweep(pos,multiprocessing=True)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 4
    for i in range(len(mockResult)):
        for j in range(3):
            assert round(result[i][j],rounding) == round(mockResult[i][j],rounding)


def test_GetBSweepList_multiprocessing_many_objects_few_positions():
    # Check if getB sweep is performing multipoint
    # calculations utilizing multiple processes

    from magpylib._lib.classes.magnets import Box
    type_erMsg =  "getBsweep did not return a numpy array."
    #Input
    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([     [2,2,2],
                    [2,2,2],
                    [2,2,3]])

    numberOfBoxes = 30 
    mockResult = [  array([0.12488298, 0.10927261, 0.07805186]) * numberOfBoxes,
                    array([0.12488298, 0.10927261, 0.07805186]) * numberOfBoxes,
                    array([0.06221036, 0.05307679, 0.0756595 ]) * numberOfBoxes,]
    #Run   
    
    c = Collection([Box(mag,dim) for i in range(numberOfBoxes)])
    result = c.getBsweep(pos,multiprocessing=True)
    assert isinstance(result,ndarray), type_erMsg
    rounding = 3
    for i in range(len(mockResult)):
        for j in range(3):
            assert round(result[i][j],rounding) == round(mockResult[i][j],rounding)

def test_GetBSweepArray_Error():
    # Check if getBsweep throws an error when 
    # crazy n dimensional arrays are provided

    errMsg = "unexpected getB for collection"
    type_erMsg =  "getBsweep did not return a numpy array."
    from magpylib._lib.classes.magnets import Box
    from numpy import allclose
    #Input

    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([     [[2,2,2],
                    [2,2,2],
                    [2,2,3]],
                    [[2,2,2],
                    [2,2,2],
                    [2,2,3]]])

    mockResult = [  [[0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],],
                    [[0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],]]
    #Run   
    with pytest.raises(AssertionError):
        b = Box(mag,dim)
        b2 = Box(mag,dim)
        c = Collection(b,b2)
        result = c.getBsweep(pos,multiprocessing=False)
        assert isinstance(result,ndarray), type_erMsg
        assert allclose(result,mockResult), errMsg  #check if the field results are the same as the mock results in the array

def test_GetSweepArray_multiprocessing_error():
    # Check if getBsweep throws an error when 
    # crazy n dimensional arrays are provided 
    # when multiprocessing is on

    errMsg = "unexpected getB for collection"
    type_erMsg =  "getBsweep did not return a numpy array."
    
    from magpylib._lib.classes.magnets import Box
    from numpy import allclose
    # Input

    mag=(2,3,5)
    dim=(2,2,2)
    pos=array([     [[2,2,2],
                    [2,2,2],
                    [2,2,3]],
                    [[2,2,2],
                    [2,2,2],
                    [2,2,3]]])

    mockResult = [  [[0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.10615358, 0.151319  ],],
                    [[0.24976596, 0.21854521, 0.15610372],
                    [0.24976596, 0.21854521, 0.15610372],
                    [0.12442073, 0.1061 ],]]
    # Run   
    with pytest.raises(AssertionError):    
        b = Box(mag,dim)
        b2 = Box(mag,dim)
        c = Collection(b,b2)
        result = c.getBsweep(pos,multiprocessing=True)
        assert isinstance(result,ndarray), type_erMsg
        assert allclose(result,mockResult), errMsg  #check if the field results are the same as the mock results in the array