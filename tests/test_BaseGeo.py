import numpy as np
from magpylib3._lib.obj_classes.class_BaseGeo import BaseGeo
from scipy.spatial.transform import Rotation as R

def test_BaseGeo():
    """
    fundamental simple test of setter, getter, .move and .rotate
    """

    Ptest = np.array([[0,0,0], [1,2,3], [0,0,0], [0,0,0], 
        [0,0,0], [0,0,0], [0.67545246,-0.6675014,-0.21692852]])

    Otest = np.array([[0,0,0], [0.1,0.2,0.3], [0.1,0.2,0.3],
        [0,0,0], [0.20990649,0.41981298,0.62971947], [0,0,0],
        [0.59199676,0.44281248,0.48074693]])

    P, O = [],[]

    bg = BaseGeo((0,0,0),None)
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    bg.pos = (1,2,3)
    bg.rot = R.from_rotvec((.1,.2,.3))
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    bg.move((-1,-2,-3))
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    rr = R.from_rotvec((-.1,-.2,-.3))
    bg.rotate(rr)
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    bg.rotate_from_angax(45,(1,2,3))
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    bg.rotate_from_angax(-np.pi/4,(1,2,3),degree=False)
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    rr = R.from_rotvec((.1,.2,.3))
    bg.rotate(rr,anchor=(3,2,1)).rotate_from_angax(33,(3,2,1),anchor=0)
    P += [bg.pos]
    O += [bg.rot.as_rotvec()]

    P = np.array(P)
    O = np.array(O)

    assert np.allclose(P, Ptest), 'test_BaseGeo bad position'
    
    assert np.allclose(O, Otest),  'test_BaseGeo bad orientation'
