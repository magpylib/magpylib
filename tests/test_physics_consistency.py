import numpy as np
import magpylib as mag3


def test_dipole_approximation():
    """ test if all source fields converge towards the correct dipole field at distance
    """
    mag = np.array([111,222,333])
    pos = (1234,-234, 345)

    # box with volume = 1 mm^3
    src1 = mag3.magnet.Box(mag, dimension=(1,1,1))
    B1 = src1.getB(pos)

    # Cylinder with volume = 1 mm^3
    dia = np.sqrt(4/np.pi)
    src2 = mag3.magnet.Cylinder(mag, dimension=(dia,1))
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
    src1 = mag3.current.Circular(current=i0, diameter=dia)
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
    src1 = mag3.magnet.Cylinder(magnetization=(0,0,i0/h0*4*np.pi/10), dimension=(r0,h0))
    src2 = mag3.current.Circular(current=i0, diameter=r0)

    H1 = src1.getH(pos_obs)
    H2 = src2.getH(pos_obs)

    assert np.allclose(H1, H2)


def test_Line_vs_Circular():
    """ show that line prodices the same as circular
    """

   # finely approximated loop by lines
    ts = np.linspace(0,2*np.pi,10000)
    verts = np.array([(np.cos(t), np.sin(t), 0) for t in ts])
    ps = verts[:-1]
    pe = verts[1:]

    # positions
    ts = np.linspace(-3,3,2)
    po = np.array([(x,y,z) for x in ts for y in ts for z in ts])

    # field from line currents
    Bls = []
    for p in po:
        Bl = mag3.getBv(source_type='Line', observer=p, current=1,
            segment_start=ps, segment_end=pe)
        Bls += [np.sum(Bl, axis=0)]
    Bls = np.array(Bls)

    # field from current loop
    src = mag3.current.Circular(current=1, diameter=2)
    Bcs = src.getB(po)

    assert np.allclose(Bls,Bcs)


def test_Line_vs_Infinite():
    """ compare line current result vs analytical solution to infinite Line
    """

    pos_obs = np.array([(1.,2,3), (-3,2,-1), (2,-1,-4)])

    def Binf(i0, pos):
        """ field of inf line current on z-axis """
        x,y,_ = pos
        r = np.sqrt(x**2+y**2)
        e_phi = np.array([-y,x,0])
        e_phi = e_phi/np.linalg.norm(e_phi)
        mu0 = 4*np.pi*1e-7
        return i0*mu0/2/np.pi/r*e_phi * 1000 * 1000 #mT mm

    ps = (0,0,-1000000)
    pe = (0,0,1000000)
    Bls, Binfs = [], []
    for p in pos_obs:
        Bls += [mag3.getBv(source_type='Line', observer=p, current=1,
            segment_start=ps, segment_end=pe)]
        Binfs += [Binf(1,p)]
    Bls = np.array(Bls)
    Binfs = np.array(Binfs)

    assert np.allclose(Bls, Binfs)
