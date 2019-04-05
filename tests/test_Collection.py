import unittest
from magpylib._lib.classes.collection import Collection
from magpylib._lib.classes.base import RCS
from numpy import array
import pytest

def test_collectionGetB():
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

def test_collectionGetB_markerInput():
    # TODO: find out a good way to compare plot results.
    # Have output plot dava saved in text maybe
    # Just test input validity for now.
    from magpylib._lib.classes.magnets import Box
    b = Box([1,1,1],[1,1,1],pos=(5,5,5))

    a = Collection()
    markers = [[0,0,0],    # int
               [.1,.1,.1], # float
               b.position] # float64

    a.displaySystem(markers,suppress=True)

def test_collectionGetBList():
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

def test_CollectionAddList():
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

def test_collectionGetBMulticoreList():
    from magpylib._lib.classes.magnets import Box

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
    result = c.getBMulticore(pos)

    rounding = 4
    for i in range(len(mockResult)):
        for j in range(3):
            assert round(result[i][j],rounding) == round(mockResult[i][j],rounding)

def test_collectionGetBMulticoreArray():
    errMsg = "unexpected getB for collection"
    
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
        result = c.getBMulticore(pos)

        assert allclose(result,mockResult), errMsg  #check if the field results are the same as the mock results in the array