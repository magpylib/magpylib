from doctest import REPORT_CDIFF
import magpylib as magpy
import numpy as np
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.exceptions import MagpylibBadUserInput


def test_Tetrahedron_repr():
    """ Tetrahedron repr test"""
    tetra = magpy.magnet.Tetrahedron()
    assert tetra.__repr__()[:11] == "Tetrahedron", "Tetrahedron repr failed"


def test_tetra_input():
    """test obj-oriented triangle vs cube"""
    obs = (1,2,3)
    mag = (111,222,333)
    vert_list = [[(1,1,-1), (1,1,1), (-1,1,1), (1,-1,1)],
        [(-1,-1,1), (-1,1,1), (1,-1,1), (1,-1,-1)],
        [(-1,-1,-1), (-1,-1,1), (-1,1,-1), (1,-1,-1)],
        [(-1,1,-1), (1,-1,-1), (-1,-1,1), (-1,1,1)],
        [(1,-1,-1), (1,1,-1), (1,-1,1), (-1,1,1)],
        [(-1,1,-1), (-1,1,1), (1,1,-1), (1,-1,-1)],]

    coll = magpy.Collection()
    for v in vert_list:
        coll.add(magpy.magnet.Tetrahedron(mag, v))

    cube = magpy.magnet.Cuboid(mag, (2,2,2))

    b = coll.getB(obs)
    bb = cube.getB(obs)
    np.testing.assert_allclose(b, bb)

    h = coll.getH(obs)
    hh = cube.getH(obs)
    np.testing.assert_allclose(h, hh)


def test_tetra_bad_inputs():
    """test obj-oriented triangle vs cube"""

    bad_inputs = [
        1,
        [[(1,1,-1), (1,1,1), (-1,1,1), (1,-1,1)]]*2,
        [(1,1,-1), (1,1,1), (-1,1,1)],
        '123',
    ]

    mag = (111,222,333)
    for bad in bad_inputs:
        def test_function():
            """bad vert input"""
            magpy.magnet.Tetrahedron(mag, bad)
        np.testing.assert_raises(MagpylibBadUserInput, test_function)


def test_tetra_barycenter():
    """ get barycenter"""
    mag = (111,222,333)
    vert = [(1,1,-1), (1,1,1), (-1,1,1), (1,-1,1)]
    tetra = magpy.magnet.Tetrahedron(mag, vert)
    np.testing.assert_allclose(tetra.barycenter, (0.5, 0.5, 0.5))
