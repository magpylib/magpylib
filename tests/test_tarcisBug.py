import magpylib as magpy
import numpy
import unittest

def test_rounding_error():
    ## This describes the rotation bug. 
    ## Once it is fixed, this test will break.
    I = 1
    d = 10e+05
    p0 = [0,0,0]

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[0,0,1],anchor=[0,0,0])
    c.rotate(90,[1,0,0],anchor=[0,0,0])
    #print(c.angle) #These turn out differently for both cases.
    #print(c.axis)

    result1 = c.getB(p0)

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[1,0,0],anchor=[0,0,0])
    #print(c.angle)
    #print(c.axis)

    result2 = c.getB(p0)
    assert all(numpy.isclose(result1,result2)==True) is False


def test_rounding_error_small_dimension():
    ## This describes a case where the bug does not occur.
    I = 1
    d = 10e+03
    p0 = [0,0,0]

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[0,0,1],anchor=[0,0,0])
    c.rotate(90,[1,0,0],anchor=[0,0,0])

    result1 = c.getB(p0)

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[1,0,0],anchor=[0,0,0])

    result2 = c.getB(p0)
    assert all(numpy.isclose(result1,result2)==True)

def test_rounding_error_large_dimension():
    ## This describes a case where the bug does not occur.
    I = 1
    d = 10e+08
    p0 = [0,0,0]

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[0,0,1],anchor=[0,0,0])
    c.rotate(90,[1,0,0],anchor=[0,0,0])

    result1 = c.getB(p0)

    c = magpy.source.current.Circular(curr=I, dim=d,pos=[0,0,d/4])

    c.rotate(90,[1,0,0],anchor=[0,0,0])

    result2 = c.getB(p0)
    assert all(numpy.isclose(result1,result2)==True)

