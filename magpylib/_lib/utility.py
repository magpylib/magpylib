from typing import Tuple
from numpy import float64, isnan, array

## Helper function for validating input dimensions
def checkDimensions(expectedD: int, dim: Tuple[float,float,float], exitMsg: str="Bad dim input") -> array:
    assert all(coord == 0 for coord in dim) is False, exitMsg + ", all values are zero"
    dimension = array(dim, dtype=float64, copy=False) 
    assert (not any(isnan(dimension))  and  len(dimension) == expectedD), exitMsg
    return dimension
