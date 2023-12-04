import numpy as np

import magpylib as magpy


def test_path_old_new_move():
    """test path move and compare to old style computation"""
    n = 100
    s_pos = (0, 0, 0)

    # path style code translation
    pm1 = magpy.magnet.Cylinder((0, 0, 1000), (3, 3), position=(-5, 0, 3))
    pm1.move([(x, 0, 0) for x in np.linspace(0, 10, 100)], start=-1)
    B1 = pm1.getB(s_pos)

    # old style code translation
    pm2 = magpy.magnet.Cylinder((0, 0, 1000), (3, 3), position=(0, 0, 3))
    ts = np.linspace(-5, 5, n)
    possis = np.array([(t, 0, 0) for t in ts])
    B2 = pm2.getB(possis[::-1])

    assert np.allclose(B1, B2), "path move problem"


def test_path_old_new_rotate():
    """test path rotate
    compare to old style computation
    """

    n = 111
    s_pos = (0, 0, 0)
    ax = (1, 0, 0)
    anch = (0, 0, 10)

    # path style code rotation
    pm1 = magpy.magnet.Cuboid((0, 0, 1000), (1, 2, 3), position=(0, 0, 3))
    pm1.rotate_from_angax(-30, ax, anch)
    pm1.rotate_from_angax(np.linspace(0, 60, n), "x", anch, start=-1)
    B1 = pm1.getB(s_pos)

    # old style code rotation
    pm2 = magpy.magnet.Cuboid((0, 0, 1000), (1, 2, 3), position=(0, 0, 3))
    pm2.rotate_from_angax(-30, ax, anch)
    B2 = []
    for _ in range(n):
        B2 += [pm2.getB(s_pos)]
        pm2.rotate_from_angax(60 / (n - 1), ax, anch)
    B2 = np.array(B2)

    assert np.allclose(B1, B2), "path rotate problem"
