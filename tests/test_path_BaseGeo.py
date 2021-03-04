import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo


def test_path_BaseGeo():
    """test path functionality
    """
    poss = []
    rots = []

    bgeo = BaseGeo((0,0,0), None)
    poss += [bgeo.pos]
    rots += [bgeo.rot.as_rotvec()]
    poss += [bgeo._pos]
    rots += [bgeo._rot.as_rotvec()]

    bgeo = BaseGeo([(0,0,0),(1,1,1)], None)
    poss += [bgeo.pos]
    rots += [bgeo.rot.as_rotvec()]
    poss += [bgeo._pos]
    rots += [bgeo._rot.as_rotvec()]

    bgeo = BaseGeo((0,0,0), R.from_rotvec((.1,.2,.3)))
    poss += [bgeo.pos]
    rots += [bgeo.rot.as_rotvec()]

    bgeo.pos = [(2,2,2),(3,3,3)]
    bgeo.rot = R.from_rotvec([(.1,.2,.3),(.2,.3,.4)])
    poss += [bgeo.pos]
    rots += [bgeo.rot.as_rotvec()]

    bgeo = BaseGeo((0,0,0), None)
    bgeo.move_by((3,0,0))
    bgeo.move_by((3,0,0), steps=3)
    bgeo.rotate_from_angax(330,(0,0,1),anchor=0,steps=5)
    bgeo.move_by((0,0,10),steps=-7)
    poss += [bgeo.pos]
    rots += [bgeo.rot.as_rotvec()]

    x = np.array([])
    for p,r in zip(poss,rots):
        x = np.r_[x,p.flatten(),r.flatten()]

    #pickle.dump(x,open('testdata_path_BaseGeo.p', 'wb'))

    x_test = pickle.load(open('tests/testdata/testdata_path_BaseGeo.p', 'rb'))

    assert np.allclose(x,x_test), 'path problem'
