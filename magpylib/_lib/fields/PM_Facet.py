from numpy import array
from warnings import warn

def Bfield_Facet(MAG, pos, vertices):
    warn('Warning: getB for facet not yet implemented, returning [0,0,0]', RuntimeWarning)
    return array([0,0,0])