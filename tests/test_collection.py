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