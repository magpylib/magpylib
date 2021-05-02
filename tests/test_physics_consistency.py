import numpy as np
import magpylib as mag3


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
    assert np.allclose(B1,B2)

    # Sphere with volume = 1 mm^3
    dia = (6/np.pi)**(1/3)
    src3 = mag3.magnet.Sphere(mag, dia)
    B3 = src3.getB(pos)
    assert np.allclose(B1,B3)

    #  Dipole with mom=mag
    src4 = mag3.misc.Dipole(moment=mag)
    B4 = src4.getB(pos)
    assert np.allclose(B1,B4)

    # Circular loop vs Dipole
    dia = 2
    i0 = 234
    m0 = dia**2 * np.pi**2 / 10 * i0
    src1 = mag3.current.Circular(current=i0, dim=dia)
    src2 = mag3.misc.Dipole(moment=(0,0,m0))
    H1 = src1.getH(pos)
    H2 = src2.getH(pos)
    assert np.allclose(H1, H2)


def test_Circular_vs_Cylinder_field():
    """
    The H-field of a loop with radius r0[mm] and current i0[A] is the same
    as the H-field of a cylinder with radius r0[mm], height h0[mm] and
    magnetization (0, 0, 4pi/10*i0/h0) !!!
    """

    pos_obs = np.random.rand(111,3)

    r0 = 2
    h0 = 1e-6
    i0 = 1
    src1 = mag3.magnet.Cylinder(mag=(0,0,i0/h0*4*np.pi/10), dim=(r0,h0))
    src2 = mag3.current.Circular(current=i0, dim=r0)

    H1 = src1.getH(pos_obs)
    H2 = src2.getH(pos_obs)

    assert np.allclose(H1, H2)
