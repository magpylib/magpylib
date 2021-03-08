import numpy as np
import magpylib as mag3

def test_path_tile():
    """ Test if auto-tiled paths of objects will properly be reset
    in getB_level2 before returning
    """
    pm1 = mag3.magnet.Box((11,22,33),(1,2,3))
    pm2 = mag3.magnet.Box((11,22,33),(1,2,3))
    pm2.move_by((10,10,10),steps=33)

    path1p = pm1.pos
    path1r = pm1.rot

    path2p = pm2.pos
    path2r = pm2.rot

    _ = mag3.getB([pm1,pm2],[0,0,0])

    assert np.all(path1p == pm1.pos), 'FAILED: getB modified object path'
    assert np.all(path1r.as_quat() == pm1.rot.as_quat()), 'FAILED: getB modified object path'
    assert np.all(path2p == pm2.pos), 'FAILED: getB modified object path'
    assert np.all(path2r.as_quat() == pm2.rot.as_quat()), 'FAILED: getB modified object path'
