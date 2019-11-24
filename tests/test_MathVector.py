from magpylib._lib.mathLib_vector import QmultV, QconjV, getRotQuatV
from numpy import array, amax
from magpylib._lib.mathLib import getRotQuat
from magpylib.math import randomAxisV
import numpy as np

'''
def test_Q():

    x = array([[1,2,3,4],[2,3,4,5]])
    y = array([[4,3,2,1],[5,4,3,2]])

    assert (QmultV(x,y) == array([[-12,6,24,12],[-24,16,40,22]])).all(), "bad QmultV"
    assert (QconjV(x) == array([[1,-2,-3,-4],[2,-3,-4,-5]])).all(), "bad Qconj"

    ANGS = np.random.rand(5)*360
    AXES = randomAxisV(5)

    Qs = getRotQuatV(ANGS,AXES)
    for Q,ang,axe in zip(Qs,ANGS,AXES):
        assert amax(Q - array(getRotQuat(ang,axe))) < 1e-10, "bad getRotQuatV"
'''


def test_randomAxisV():
    X=randomAxisV(1000)
    print(X)
    print( all(X.shape == (1000,3))) # , "bad randomAxis"
    
    #lX = np.linalg.norm(X,axis=1)
    #assert np.sum(lX>=1) < 1, "bad randomAxis"


test_randomAxisV()