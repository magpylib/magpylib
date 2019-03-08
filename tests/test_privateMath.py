import unittest
from magpylib._lib.mathLibPrivate import fastSum3D
import numpy


def test_fastSum3D():
    """
    #Test if it can sum a position vector
    """
    position=[2.3,5.6,2.0]
    assert round(fastSum3D(position),2)==9.90, "Error, not adding correctly"
