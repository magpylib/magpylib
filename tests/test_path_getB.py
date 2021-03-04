import numpy as np
from magpylib.magnet import Box, Cylinder

def test_path_old_new_move():
    ''' test path move
    compare to old style computation
    '''
    n = 100
    s_pos = (0,0,0)

    # path style code translation
    pm1 = Cylinder((0,0,1000),(3,3),pos=(-5,0,3))
    pm1.move_by((10,0,0),steps=n)
    B1 = pm1.getB(s_pos)

    # old style code translation
    pm2 = Cylinder((0,0,1000),(3,3),pos=(0,0,3))
    ts = np.linspace(-5,5,n+1)
    possis = np.array([(t,0,0) for t in ts])
    B2 = pm2.getB(possis[::-1])

    assert np.allclose(B1,B2), 'path move problem'


def test_path_old_new_rotate():
    ''' test path rotate
    compare to old style computation
    '''

    n = 111
    s_pos = (0,0,0)
    ax = (1,0,0)
    anch=(0,0,10)

    # path style code rotation
    pm1 = Box((0,0,1000),(1,2,3),pos=(0,0,3))
    pm1.rotate_from_angax(-30,ax,anch)
    pm1.rotate_from_angax(60,'x',anch,steps=n)
    B1 = pm1.getB(s_pos)

    # old style code rotation
    pm2 = Box((0,0,1000),(1,2,3),pos=(0,0,3))
    pm2.rotate_from_angax(-30,ax,anch)
    B2 = []
    for _ in range(n+1):
        B2 += [pm2.getB(s_pos)]
        pm2.rotate_from_angax(60/n,ax,anch)
    B2 = np.array(B2)

    assert np.allclose(B1,B2), 'path rotate problem'
