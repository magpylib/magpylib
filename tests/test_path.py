import numpy as np
import magpylib as mag3
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


def test_path_merge():
    """test path_merge with context manager
    """
    pm = mag3.magnet.Box((1,2,3),(1,2,3))

    with mag3.path_merge(pm, steps=333):
        pm.move_to((10,0,0))
        pm.rotate_from_angax(1111,'z',anchor=0)
        pm.move_by((0,0,15))
    pm.move_by((0,0,-10),steps=-111)
    B1 = pm.getB((0,0,0))


    pm = mag3.magnet.Box((1,2,3),(1,2,3))
    pm.move_to((10,0,0), steps=333)
    pm.rotate_from_angax(1111,'z',anchor=0,steps=-333)
    pm.move_by((0,0,15),steps=-333)
    pm.move_by((0,0,-10),steps=-111)
    B2 = pm.getB((0,0,0))

    assert np.allclose(B1,B2), 'simple path_merge gone wrong'


def test_path_merge_collection():
    """test multi motion with collection
    """
    pm1 = mag3.magnet.Box((11,22,33),(1,2,3),pos=(-10,0,0))
    pm2 = mag3.magnet.Cylinder((0,0,333),(1,2),pos=(10,0,0))
    col = mag3.Collection(pm1,pm2)
    with mag3.path_merge(col, steps=333):
        col.rotate_from_angax(1111,'z',anchor=0)
        col.move_by((0,0,15))
    B1 = col.getB((0,0,0))


    pm1 = mag3.magnet.Box((11,22,33),(1,2,3),pos=(-10,0,0))
    pm2 = mag3.magnet.Cylinder((0,0,333),(1,2),pos=(10,0,0))
    pm1.rotate_from_angax(1111,'z',anchor=0,steps=333)
    pm2.rotate_from_angax(1111,'z',anchor=0,steps=333)
    pm1.move_by((0,0,15),steps=-333)
    pm2.move_by((0,0,15),steps=-333)
    B2 = mag3.getB([pm1,pm2],(0,0,0),sumup=True)

    assert np.allclose(B1,B2), 'simple path_merge gone wrong'
