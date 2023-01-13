import magpylib as magpy
import numpy as np
from magpylib._src.exceptions import MagpylibMissingInput


def test_Facet_repr():
    """Facet repr test"""
    line = magpy.misc.Facet()
    assert line.__repr__()[:5] == "Facet", "Facet repr failed"


def test_facet_input1():
    """test obj-oriented facet vs cube"""
    obs = (1,2,3)
    mag = (0,0,333)
    vert = np.array([
        [(-1,-1,1), (1,-1,1), (-1,1,1)], #top1
        [(1,-1,-1), (-1,-1,-1), (-1,1,-1)], #bott1
        [(1,-1,1), (1,1,1), (-1,1,1)],   #top2
        [(1,1,-1), (1,-1,-1), (-1,1,-1)],   #bott2
        ])
    face = magpy.misc.Facet(mag, vert)
    cube = magpy.magnet.Cuboid(mag, (2,2,2))

    b = face.getB(obs)
    bb = cube.getB(obs)

    np.testing.assert_allclose(b, bb)


def test_facet_input2():
    """test variable Facet class inputs against each other"""
    obs = (1,2,3)
    mag = (0,0,333)
    vert1 = [(-1,-1,1), (1,-1,1), (-1,1,1)]
    vert2 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]
    vert3 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]*2
    vert4 = [[(-1,-1,1), (1,-1,1), (-1,1,1)]]*3

    face1 = magpy.misc.Facet(mag, vert1)
    b1 = face1.getB(obs)

    face2 = magpy.misc.Facet(mag, vert2)
    b_test = face2.getB(obs)
    np.testing.assert_allclose(b1, b_test)

    face3 = magpy.misc.Facet(mag, vert3)
    b_test = face3.getB(obs)/2
    np.testing.assert_allclose(b1, b_test)

    face4 = magpy.misc.Facet(mag, vert4)
    b_test = face4.getB(obs)/3
    np.testing.assert_allclose(b1, b_test)

    face = magpy.misc.Facet(mag, vert4)
    b_test = magpy.getB([face1, face2, face3, face4], obs, sumup=True)/7
    np.testing.assert_allclose(b1, b_test)


def test_facet_input3():
    """test core facet vs objOriented facet"""

    obs = np.array([(3,4,5)]*4)
    mag = np.array([(111,222,333)]*4)
    vert = np.array([
        [(0,0,0), (3,0,0), (0,10,0)],
        [(3,0,0), (5,0,0), (0,10,0)],
        [(5,0,0), (6,0,0), (0,10,0)],
        [(6,0,0), (10,0,0), (0,10,0)],
        ])
    b = magpy.core.facet_field('B', obs, mag, vert)
    b = np.sum(b, axis=0)

    face1 = magpy.misc.Facet(mag[0], vertices=vert[0])
    face2 = magpy.misc.Facet(mag[0], vertices=vert[1:])

    bb = magpy.getB([face1, face2], obs[0], sumup=True)

    np.testing.assert_allclose(b, bb)


def test_empty_object_initialization():
    """empty object init and error msg"""
    
    fac = magpy.misc.Facet()

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
    vert = np.array([
        [(-1,-1,1), (1,-1,1), (-1,1,1)], #top1
        [(1,-1,-1), (-1,-1,-1), (-1,1,-1)], #bott1
        [(1,-1,1), (1,1,1), (-1,1,1)],   #top2
        [(1,1,-1), (1,-1,-1), (-1,1,-1)],   #bott2
        ])
    face = magpy.misc.Facet(mag, vert)
    bary = np.array([0,0,0])
    np.testing.assert_allclose(face.barycenter, bary)
