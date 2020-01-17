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
