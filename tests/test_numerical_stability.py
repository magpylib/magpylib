import numpy as np

import magpylib as magpy


def test_loop_field():
    """
    test numerical instability of current loop field at r=0

    Many users have seen that by continued rotation about an anchor
    the field become instable. This is a result of small displacements from the axis
    where the field is evaluated due to floating-point errors. see paper Leitner2021.
    """
    lop = magpy.current.Circle(1000, 1)

    anch = (0, 0, 1)
    B = []
    for _ in range(1000):
        lop.rotate_from_angax(100, "x", anchor=anch, start=-1)
        B += [lop.getB(anch)]

    B = np.array(B)
    normB = np.linalg.norm(B, axis=1)
    norms = normB / normB[0]

    assert np.allclose(norms, np.ones(1000))
