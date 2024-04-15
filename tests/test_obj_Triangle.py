import numpy as np

import magpylib as magpy
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_triangle import BHJM_triangle


def test_Triangle_repr():
    """Triangle repr test"""
    line = magpy.misc.Triangle()
    assert repr(line)[:8] == "Triangle", "Triangle repr failed"


def test_triangle_input1():
    """test obj-oriented triangle vs cube"""
    obs = (1, 2, 3)
    pol = (0, 0, 333)
    vert = np.array(
        [
            [(-1, -1, 1), (1, -1, 1), (-1, 1, 1)],  # top1
            [(1, -1, -1), (-1, -1, -1), (-1, 1, -1)],  # bott1
            [(1, -1, 1), (1, 1, 1), (-1, 1, 1)],  # top2
            [(1, 1, -1), (1, -1, -1), (-1, 1, -1)],  # bott2
        ]
    )
    coll = magpy.Collection()
    for v in vert:
        coll.add(magpy.misc.Triangle(polarization=pol, vertices=v))
    cube = magpy.magnet.Cuboid(polarization=pol, dimension=(2, 2, 2))

    b = coll.getB(obs)
    bb = cube.getB(obs)

    np.testing.assert_allclose(b, bb)


def test_triangle_input3():
    """test core triangle vs objOriented triangle"""

    obs = np.array([(3, 4, 5)] * 4)
    pol = np.array([(111, 222, 333)] * 4)
    vert = np.array(
        [
            [(0, 0, 0), (3, 0, 0), (0, 10, 0)],
            [(3, 0, 0), (5, 0, 0), (0, 10, 0)],
            [(5, 0, 0), (6, 0, 0), (0, 10, 0)],
            [(6, 0, 0), (10, 0, 0), (0, 10, 0)],
        ]
    )
    b = BHJM_triangle(field="B", observers=obs, polarization=pol, vertices=vert)
    b = np.sum(b, axis=0)

    tri1 = magpy.misc.Triangle(polarization=pol[0], vertices=vert[0])
    tri2 = magpy.misc.Triangle(polarization=pol[0], vertices=vert[1])
    tri3 = magpy.misc.Triangle(polarization=pol[0], vertices=vert[2])
    tri4 = magpy.misc.Triangle(polarization=pol[0], vertices=vert[3])

    bb = magpy.getB([tri1, tri2, tri3, tri4], obs[0], sumup=True)

    np.testing.assert_allclose(b, bb)


def test_empty_object_initialization():
    """empty object init and error msg"""

    fac = magpy.misc.Triangle()

    def call_getB():
        """dummy function call getB"""
        fac.getB()

    np.testing.assert_raises(MagpylibMissingInput, call_getB)


def test_Triangle_barycenter():
    """test Triangle barycenter"""
    pol = (0, 0, 0.333)
    vert = ((-1, -1, 0), (1, -1, 0), (0, 2, 0))
    face = magpy.misc.Triangle(polarization=pol, vertices=vert)
    bary = np.array([0, 0, 0])
    np.testing.assert_allclose(face.barycenter, bary)
