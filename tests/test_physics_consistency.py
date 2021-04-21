import magpylib as mag3
import numpy as np


def test_dipole_approximation():
    """ test if all source fields converge towards the correct dipole field at distance
    """
    mag = np.array([111,222,333])
    pos = (1234,-234, 345)

    # box with volume = 1 mm^3 
    src1 = mag3.magnet.Box(mag, dim=(1,1,1))
    B1 = src1.getB(pos)

    # Cylinder with volume = 1 mm^3
    dia = np.sqrt(4/np.pi)
    src2 = mag3.magnet.Cylinder(mag, dim=(dia,1))
    B2 = src2.getB(pos)

    # Sphere with volume = 1 mm^3
    dia = (6/np.pi)**(1/3)
    src3 = mag3.magnet.Sphere(mag, dia)
    B3 = src3.getB(pos)

    #  Dipole with mom=mag
    src4 = mag3.misc.Dipole(moment=mag)
    B4 = src4.getB(pos)

    assert np.allclose(B1,B2)
    assert np.allclose(B1,B3)
    assert np.allclose(B1,B4)
