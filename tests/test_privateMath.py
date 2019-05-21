import unittest
from magpylib._lib.mathLibPrivate import fastSum3D, fastNorm3D, arccosSTABLE
import numpy


def test_fastSum3D():
    """
    #Test if it can sum a position vector
    """
    position=[2.3,5.6,2.0]
    assert round(fastSum3D(position),2)==9.90, "Error, not adding correctly"

###

def test_fastNorm3D():
    from numpy import isnan
    result=round(fastNorm3D([58.2,25,25]),4)
    assert result==68.0973, "Expected 68.0973, got " + str(result)

def test_arccosSTABLE():
    assert arccosSTABLE(2) == 0
    assert arccosSTABLE(-2) == numpy.pi