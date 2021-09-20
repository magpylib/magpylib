#import os
#import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo


def test_BaseGeo_basics():
    """ fundamental usage test
    """

    ptest = np.array([[0,0,0], [1,2,3], [0,0,0], [0,0,0],
                      [0,0,0], [0,0,0], [0.67545246,-0.6675014,-0.21692852]])

    otest = np.array([[0,0,0], [0.1,0.2,0.3], [0.1,0.2,0.3],
                      [0,0,0], [0.20990649,0.41981298,0.62971947], [0,0,0],
                      [0.59199676,0.44281248,0.48074693]])

    poss, rots = [],[]

    bgeo = BaseGeo((0,0,0),None)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.position = (1,2,3)
    bgeo.orientation = R.from_rotvec((.1,.2,.3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.move((-1,-2,-3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((-.1,-.2,-.3))
    bgeo.rotate(rotation=rot)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(angle=45, axis=(1,2,3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(-np.pi/4,(1,2,3),degrees=False)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((.1,.2,.3))
    bgeo.rotate(rot, anchor=(3,2,1)).rotate_from_angax(33,(3,2,1),anchor=0)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    poss = np.array(poss)
    rots = np.array(rots)

    assert np.allclose(poss, ptest), 'test_BaseGeo bad position'
    assert np.allclose(rots, otest),  'test_BaseGeo bad orientation'


def test_rotate_vs_rotate_from():
    """ testing rotate vs rotate_from_angax
    """
    roz = [(.1,.2,.3), (-.1,-.1,-.1),(.2,0,0),(.3,0,0),(0,0,.4),(0,-.2,0)]
    rroz = R.from_rotvec(roz)

    bg1 = BaseGeo(position=(3,4,5), orientation=R.from_quat((0,0,0,1)))
    bg1.rotate(rotation=rroz, anchor=(-3,-2,1), start=1, increment=True)
    pos1 = bg1.position
    ori1 = bg1.orientation.as_quat()

    bg2 = BaseGeo(position=(3,4,5), orientation=R.from_quat((0,0,0,1)))
    angs = np.linalg.norm(roz, axis=1)
    for ang,ax in zip(angs,roz):
        bg2.rotate_from_angax(angle=ang, degrees=False, axis=ax, anchor=(-3,-2,1), start='append')
    pos2 = bg2.position
    ori2 = bg2.orientation.as_quat()

    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)


def test_BaseGeo_reset_path():
    """ testing reset path
    """
    #pylint: disable=protected-access
    bg = BaseGeo((0,0,0),R.from_quat((0,0,0,1)))
    bg.move([(1,1,1)]*11)

    assert len(bg._position)==11, 'bad path generation'

    bg.reset_path()
    assert len(bg._position)==1, 'bad path reset'


def test_BaseGeo_anchor_None():
    """ testing rotation with None anchor
    """
    pos = np.array([1,2,3])
    bg = BaseGeo(pos,R.from_quat((0,0,0,1)))
    bg.rotate(R.from_rotvec([(0,0,0),(.1,.2,.3),(.2,.4,.6)]))

    pos3 = np.array([pos]*3)
    rot3 = np.array([(0,0,0),(.1,.2,.3),(.2,.4,.6)])
    assert np.allclose(bg.position,pos3), 'None rotation changed position'
    assert np.allclose(bg.orientation.as_rotvec(),rot3), 'None rotation did not adjust rot'


def evall(objj):
    """ return pos and orient of objject
    """
    #pylint: disable=protected-access
    pp = objj._position
    rr = objj._orientation.as_quat()
    rr = np.array([r/max(r) for r in rr])
    return (pp,rr)


def test_attach():
    """ test attach functionality
    """
    bg = BaseGeo([0,0,0], R.from_rotvec((0,0,0)))
    rot_obj = R.from_rotvec([(x,0,0) for x in np.linspace(0,10,11)])
    bg.rotate(rot_obj)

    bg2 = BaseGeo([0,0,0], R.from_rotvec((0,0,0)))
    roto = R.from_rotvec((1,0,0))
    for _ in range(10):
        bg2.rotate(roto, start='append')

    assert np.allclose(bg.position,bg2.position), 'attach p'
    assert np.allclose(bg.orientation.as_quat(),bg2.orientation.as_quat()), 'attach o'


def test_path_functionality1():
    """ testing path functionality in detail
    """
    pos0 = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5.]])
    rot0 = R.from_quat([(1,0,0,1),(2,0,0,1),(4,0,0,1),(5,0,0,1),(10,0,0,1.)])
    inpath = np.array([(.1,.1,.1), (.2,.2,.2), (.3,.3,.3)])

    b1,b2,b3,b4,b5 = pos0
    c1,c2,c3 = inpath
    q1,q2,q3,q4,q5 = np.array([(1,0,0,1), (1,0,0,.5), (1,0,0,.25), (1,0,0,.2), (1,0,0,.1)])

    pos, ori = evall(BaseGeo(pos0, rot0))
    P = np.array([b1, b2, b3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=0))
    P = np.array([b1+c1, b2+c2, b3+c3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=1))
    P = np.array([b1, b2+c1, b3+c2, b4+c3, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=2))
    P = np.array([b1, b2, b3+c1, b4+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)


def test_path_functionality2():
    """ testing path functionality in detail
    """
    pos0 = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5.]])
    rot0 = R.from_quat([(1,0,0,1),(2,0,0,1),(4,0,0,1),(5,0,0,1),(10,0,0,1.)])
    inpath = np.array([(.1,.1,.1), (.2,.2,.2), (.3,.3,.3)])

    b1,b2,b3,b4,b5 = pos0
    c1,c2,c3 = inpath
    q1,q2,q3,q4,q5 = np.array([(1,0,0,1), (1,0,0,.5), (1,0,0,.25), (1,0,0,.2), (1,0,0,.1)])

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=3))
    P = np.array([b1, b2, b3, b4+c1, b5+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=4))
    P = np.array([b1, b2, b3, b4, b5+c1, b5+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=5))
    P = np.array([b1, b2, b3, b4, b5, b5+c1, b5+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=6))
    P = np.array([b1, b2, b3, b4, b5, b5+c1, b5+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start='append'))
    P = np.array([b1, b2, b3, b4, b5, b5+c1, b5+c2, b5+c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)


def test_path_functionality3():
    """ testing path functionality in detail
    """
    pos0 = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5.]])
    rot0 = R.from_quat([(1,0,0,1),(2,0,0,1),(4,0,0,1),(5,0,0,1),(10,0,0,1.)])
    inpath = np.array([(.1,.1,.1), (.2,.2,.2), (.3,.3,.3)])

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=4))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-1))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=3))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-2))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=2))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-3))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=1))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-4))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=0))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-5))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=-6))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-5))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)
