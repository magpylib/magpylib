import numpy as np
import magpylib3 as mag3


def test_motion_merge():
    """test motion_merge with context manager
    """
    pm = mag3.magnet.Box((1,2,3),(1,2,3))

    with mag3.motion_merge(pm, steps=333):
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

    assert np.allclose(B1,B2), 'simple motion_merge gone wrong'


def test_motion_merge_collection():
    """test multi motion with collection
    """
    pm1 = mag3.magnet.Box((11,22,33),(1,2,3),pos=(-10,0,0))
    pm2 = mag3.magnet.Cylinder((0,0,333),(1,2),pos=(10,0,0))
    col = mag3.Collection(pm1,pm2)
    with mag3.motion_merge(col, steps=333):
        col.rotate_from_angax(1111,'z',anchor=0)
        col.move_by((0,0,15))
    B1 = col.getB((0,0,0))


    pm1 = mag3.magnet.Box((11,22,33),(1,2,3),pos=(-10,0,0))
    pm2 = mag3.magnet.Cylinder((0,0,333),(1,2),pos=(10,0,0))
    pm1.rotate_from_angax(1111,'z',anchor=0,steps=333)
    pm2.rotate_from_angax(1111,'z',anchor=0,steps=333)
    pm1.move_by((0,0,15),steps=-333)
    pm2.move_by((0,0,15),steps=-333)
    B2 = mag3.getB([pm1,pm2],pos_obs=(0,0,0),sumup=True)

    assert np.allclose(B1,B2), 'simple motion_merge gone wrong'
