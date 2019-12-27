from magpylib._lib.mathLib_vector import QmultV, QconjV, getRotQuatV, QrotationV, getAngAxV, angleAxisRotationV_priv
from magpylib._lib.mathLib_vector import axisFromAnglesV, anglesFromAxisV, angleAxisRotationV, ellipticV
from numpy import array, amax
from magpylib._lib.mathLib import getRotQuat, axisFromAngles, anglesFromAxis, angleAxisRotation, elliptic
from magpylib.math import randomAxisV
import numpy as np


def test_QV():

    x = array([[1,2,3,4],[2,3,4,5]])
    y = array([[4,3,2,1],[5,4,3,2]])
    v = array([[1,2,3],[2,3,4]])

    assert (QmultV(x,y) == array([[-12,6,24,12],[-24,16,40,22]])).all(), "bad QmultV"
    assert (QconjV(x) == array([[1,-2,-3,-4],[2,-3,-4,-5]])).all(), "bad Qconj"
    assert (QrotationV(x,v) == array([[54,60,78],[140,158,200]])).all(), "bad Qrotation"



def test_getRotQuatV():
    ANGS = np.random.rand(5)*360
    AXES = randomAxisV(5)

    Qs = getRotQuatV(ANGS,AXES)
    for Q,ang,axe in zip(Qs,ANGS,AXES):
        assert amax(Q - array(getRotQuat(ang,axe))) < 1e-10, "bad getRotQuatV"



def test_aaRot_priv():
    A = np.array([22,133,244])
    AX = np.array([[.1,.2,.3],[3,4,5],[-7,-8,-9]])
    V = np.array([[1,2,3],[2,3,4],[4,5,6]])

    sol = np.array([[1.,2.,3.,], 
        [2.57438857, 2.86042187, 3.76702936],
        [4.77190313, 4.65730779, 5.70424619]])

    assert amax(abs(angleAxisRotationV_priv(A,AX,V)-sol))<1e-6, "bad angleAxisRotationV_priv"



def test_randomAxisV():
    X=randomAxisV(1000)
    assert X.shape == (1000,3), "bad randomAxis"
    
    lX = np.linalg.norm(X,axis=1)
    assert np.sum(np.abs(lX-1)<1e-10)==1000, "bad randomAxis"



def test_axisFromAnglesV():

    ANG = np.array([[33,44],[-123,98],[-233,0],[0,0]])
    AXV = axisFromAnglesV(ANG)

    for axV,ang in zip(AXV,ANG):
        ax = axisFromAngles(ang)
        assert amax(abs(ax-axV)) < 1e-10, "bad axisFromAnglesV"



def test_anglesFromAxis():
    AX = np.array([[.1,.2,.3],[3,4,5],[-7,-8,-9],[1,0,0],[0,1,0],[0,0,1]])
    AXV = anglesFromAxisV(AX)
    for ax,axV in zip(AX,AXV):
        assert amax(abs(anglesFromAxis(ax)-axV))<1e-10, "bad anglesFromAxis"



def test_rot_Q_conversion():
    X = array([[.1,.2,.3,.4],[.2,.3,.4,.5],[-.33,-.55,.1,.23]])
    ang,ax = getAngAxV(X)
    Y = getRotQuatV(ang,ax)

    for x,y in zip(X,Y):
        assert amax(abs(x[0]-y[0]))<1e-10, "bad rot-Q conversion"
        assert amax(abs(x[1:]/amax(abs(x[1:]))-y[1:]/amax(abs(y[1:]))))<1e-10,"bad rot-Q conversion"



def test_angleAxisRotationV():
    POS = np.array([[.1,.2,.3],[0,1,0],[0,0,1]])
    ANG = np.array([22,233,-123])
    AXIS = np.array([[3,4,5],[-7,-8,-9],[1,0,0]])
    ANCHOR = np.array([[.1,3,.3],[5,5,5],[2,-3,1.23]])

    SOL = angleAxisRotationV(POS,ANG,AXIS,ANCHOR)
    for pos,ang,ax,anch,sol in zip(POS,ANG,AXIS,ANCHOR,SOL):
        assert np.amax(np.abs(angleAxisRotation(pos,ang,ax,anch)-sol))<1e-10, "bad angleAxisRotationV"



def test_ellipticV():
    #random input
    INP = np.random.rand(1000,4)

    # classical solution looped
    solC = []
    for inp in INP:
        solC += [elliptic(inp[0],inp[1],inp[2],inp[3])]
    solC = np.array(solC)

    #vector solution
    solV = ellipticV(INP[:,0],INP[:,1],INP[:,2],INP[:,3])

    assert np.amax(abs(solC-solV)) < 1e-10