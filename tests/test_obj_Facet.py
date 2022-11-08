import magpylib as magpy
import numpy as np
from magpylib._src.exceptions import MagpylibMissingInput

def test_Facet_repr():
    """Facet repr test"""
    line = magpy.magnet.Facet()
    assert line.__repr__()[:5] == "Facet", "Facet repr failed"


def test_facet_input1():
    """test obj-oriented facet vs cube"""
    obs = (1,2,3)
    mag = (0,0,333)
    fac = np.array([
        [(-1,-1,1), (1,-1,1), (-1,1,1)], #top1
        [(1,-1,-1), (-1,-1,-1), (-1,1,-1)], #bott1
        [(1,-1,1), (1,1,1), (-1,1,1)],   #top2
        [(1,1,-1), (1,-1,-1), (-1,1,-1)],   #bott2
        ])
    face = magpy.magnet.Facet(mag, fac)
    cube = magpy.magnet.Cuboid(mag, (2,2,2))

    b = face.getB(obs)
    bb = cube.getB(obs)

    np.testing.assert_allclose(b, bb)


def test_facet_input2():
    """test variable Facet class inputs against each other"""
    obs = (1,2,3)
    mag = (0,0,333)
    fac1 = [(-1,-1,1), (1,-1,1), (-1,1,1)]
    fac2 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]
    fac3 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]*2
    fac4 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]*3

    face1 = magpy.magnet.Facet(mag, fac1)
    b1 = face1.getB(obs)

    face2 = magpy.magnet.Facet(mag, fac2)
    b_test = face2.getB(obs)
    np.testing.assert_allclose(b1, b_test)

    face3 = magpy.magnet.Facet(mag, fac3)
    b_test = face3.getB(obs)/2
    np.testing.assert_allclose(b1, b_test)

    face4 = magpy.magnet.Facet(mag, fac4)
    b_test = face4.getB(obs)/3
    np.testing.assert_allclose(b1, b_test)

    face = magpy.magnet.Facet(mag, fac4)
    b_test = magpy.getB([face1, face2, face3, face4], obs, sumup=True)/7
    np.testing.assert_allclose(b1, b_test)


def test_facet_input3():
    """test core facet vs objOriented facet"""

    obs = np.array([(3,4,5)]*4)
    mag = np.array([(111,222,333)]*4)
    fac = np.array([
        [(0,0,0), (3,0,0), (0,10,0)],
        [(3,0,0), (5,0,0), (0,10,0)],
        [(5,0,0), (6,0,0), (0,10,0)],
        [(6,0,0), (10,0,0), (0,10,0)],
        ])
    b = magpy.core.magnet_facet_field('B', obs, mag, fac)
    b = np.sum(b, axis=0)

    face1 = magpy.magnet.Facet(mag[0], facets=fac[0])
    face2 = magpy.magnet.Facet(mag[0], facets=fac[1:])

    bb = magpy.getB([face1, face2], obs[0], sumup=True)

    np.testing.assert_allclose(b, bb)


def test_empty_object_initialization():
    """empty object init and error msg"""
    
    fac = magpy.magnet.Facet()

    def call_show():
        """dummy function call show"""
        fac.show()
    np.testing.assert_raises(MagpylibMissingInput, call_show)

    def call_getB():
        """dummy function call getB"""
        fac.getB()
    np.testing.assert_raises(MagpylibMissingInput, call_getB)


def test_barycenter():
    """call barycenter"""
    mag = (0,0,333)
    fac = np.array([
        [(-1,-1,1), (1,-1,1), (-1,1,1)], #top1
        [(1,-1,-1), (-1,-1,-1), (-1,1,-1)], #bott1
        [(1,-1,1), (1,1,1), (-1,1,1)],   #top2
        [(1,1,-1), (1,-1,-1), (-1,1,-1)],   #bott2
        ])
    face = magpy.magnet.Facet(mag, fac)
    bary = np.array([0,0,0])
    np.testing.assert_allclose(face.barycenter, bary)