import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo

def test_BaseGeo():
    """
    fundamental simple test of setter, getter, .move and .rotate
    """

    ptest = np.array([[0,0,0], [1,2,3], [0,0,0], [0,0,0],
                      [0,0,0], [0,0,0], [0.67545246,-0.6675014,-0.21692852]])

    otest = np.array([[0,0,0], [0.1,0.2,0.3], [0.1,0.2,0.3],
                      [0,0,0], [0.20990649,0.41981298,0.62971947], [0,0,0],
                      [0.59199676,0.44281248,0.48074693]])

    poss, rots = [],[]

    bgeo = BaseGeo((0,0,0),None)
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    bgeo.pos = (1,2,3)
    bgeo.rot = R.from_rotvec((.1,.2,.3))
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    bgeo.move_by((-1,-2,-3))
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    rot = R.from_rotvec((-.1,-.2,-.3))
    bgeo.rotate(rot)
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    bgeo.rotate_from_angax(45,(1,2,3))
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    bgeo.rotate_from_angax(-np.pi/4,(1,2,3),degree=False)
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    rot = R.from_rotvec((.1,.2,.3))
    bgeo.rotate(rot,anchor=(3,2,1)).rotate_from_angax(33,(3,2,1),anchor=0)
    poss += [bgeo.pos.copy()]
    rots += [bgeo.rot.as_rotvec()]

    poss = np.array(poss)
    rots = np.array(rots)

    assert np.allclose(poss, ptest), 'test_BaseGeo bad position'
    assert np.allclose(rots, otest),  'test_BaseGeo bad orientation'
