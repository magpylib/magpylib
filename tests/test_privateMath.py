from magpylib._lib.mathLib import getPhi,fastSum3D, fastNorm3D, arccosSTABLE, fastCross3D
from magpylib._lib.mathLib import Qmult, Qnorm2, Qunit, Qconj, getRotQuat, angleAxisRotation
from magpylib._lib.mathLib import elliptic, ellipticK, ellipticE, ellipticPi
from numpy import pi,array


# -------------------------------------------------------------------------------
def test_algebraic():
    assert round(getPhi(1,2),4) ==1.1071, "bad getPhi result at (1,2)"
    assert round(getPhi(1,0),4) ==0.0, "bad getPhi result at (1,0)"
    assert round(getPhi(-1,0),4)==3.1416, "bad getPhi result at (-1,0)"
    assert round(getPhi(0,0),4) ==0.0, "bad getPhi result at (0,0)"

    assert round(arccosSTABLE(2),4) == 0.0 , "bad arccosStable at (2)"
    assert round(arccosSTABLE(-2),4) == 3.1416, "bad arccosStable at (-2)"

    assert all(fastCross3D([1,2,3],[3,2,1]) == array([-4,8,-4])), "bad fastCross3D"

    assert round(fastSum3D([2.3,5.6,2.0]),2)==9.90, "bad fastSum3D"

    assert round(fastNorm3D([58.2,25,25]),4)==68.0973, "bad fastNorm3D"


# -------------------------------------------------------------------------------
def test_Quaternion():
    Qmult([1,2,3,4],[4,3,2,1]) == [-12,6,24,12]
    Qconj([1,2,3,4]) == [1,-2,-3,-4]
    Qnorm2([1,2,3,4]) == 30
    
    Q = Qunit([1,2,3,4])
    sol = [0.1826,0.3651,0.5477,0.7303]
    for q,s in zip(Q,sol):
        assert round(q,4)==s, "bad Qunit"

    Q = getRotQuat(33,[1,2,3])
    sol = [0.9588,0.0759,0.1518,0.2277]
    for q,s in zip(Q,sol):
        assert round(q,4)==s, "bad getRotQuat"


    V = angleAxisRotation(33,[1,2,3],[4,5,6])
    sol = [3.2868,5.8042,5.7016]
    for v,s in zip(V,sol):
        assert round(v,4)==s, "bad getRotQuat"


# -------------------------------------------------------------------------------
def test_elliptic():
    assert round(elliptic(.1,.2,.3,.4),4) == 4.7173, "bad elliptic"
    assert round(ellipticK(.1),4) == 1.6124, "bad ellipticK"
    assert round(ellipticE(.1),4) == 1.5308, "bad ellipticE"
    assert round(ellipticPi(.1,.2),4) == 1.752, "bad ellipticPi"



