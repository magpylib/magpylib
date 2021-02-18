import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3._lib.obj_classes.class_BaseGeo import BaseGeo

def test_path_BaseGeo():
    PP = []
    RR = []

    bg = BaseGeo((0,0,0), None)
    PP += [bg.pos]
    RR += [bg.rot.as_rotvec()]
    PP += [bg._pos]
    RR += [bg._rot.as_rotvec()]

    bg = BaseGeo([(0,0,0),(1,1,1)], None)
    PP += [bg.pos]
    RR += [bg.rot.as_rotvec()]
    PP += [bg._pos]
    RR += [bg._rot.as_rotvec()]

    bg = BaseGeo((0,0,0), R.from_rotvec((.1,.2,.3)))
    PP += [bg.pos]
    RR += [bg.rot.as_rotvec()]

    bg.pos = [(2,2,2),(3,3,3)]
    bg.rot = R.from_rotvec([(.1,.2,.3),(.2,.3,.4)])
    PP += [bg.pos]
    RR += [bg.rot.as_rotvec()]

    bg = BaseGeo((0,0,0), None)
    bg.move_by((3,0,0))
    bg.move_by((3,0,0), steps=3)
    bg.rotate_from_angax(330,(0,0,1),anchor=0,steps=5)
    bg.move_by((0,0,10),steps=-7)
    PP += [bg.pos]
    RR += [bg.rot.as_rotvec()]

    x = np.array([])
    for p,r in zip(PP,RR):
        x = np.r_[x,p.flatten(),r.flatten()]

    #pickle.dump(x,open('testdata_path_BaseGeo.p', 'wb'))

    x_test = pickle.load(open('tests/testdata/testdata_path_BaseGeo.p', 'rb'))

    assert np.allclose(x,x_test), 'path problem'