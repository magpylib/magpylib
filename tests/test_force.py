import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib import getFT

#########################################################################################
#########################################################################################
### ANALYTICS ###########################################################################
#########################################################################################
#########################################################################################


def test_force_analytic_dipole():
    """
    Compare force and torque between two magnetic Dipoles with well-known formula
    (e.g. https://en.wikipedia.org/wiki/Magnetic_dipole%E2%80%93dipole_interaction)

    and T = mu x B

    --> DIPOLE is good
    --> magnet force & torque is good, pivot_torque = ?
    """

    def FTana(p1, p2, m1, m2, src, piv):
        # analytical force solution
        r = p2 - p1
        r_abs = np.linalg.norm(r)
        r_unit = r / r_abs
        F_ana = (
            3
            * magpy.mu_0
            / (4 * np.pi * r_abs**4)
            * (
                np.cross(np.cross(r_unit, m1), m2)
                + np.cross(np.cross(r_unit, m2), m1)
                - 2 * r_unit * np.dot(m1, m2)
                + 5 * r_unit * (np.dot(np.cross(r_unit, m1), np.cross(r_unit, m2)))
            )
        )
        B = src.getB(p2)
        T_ana = np.cross(m2, B) + np.cross(p2 - piv, F_ana)
        return F_ana, T_ana

    # random numbers
    m1 = (
        np.array(
            [(0.976, 4.304, 2.055), (1.432, 0.352, 2.345), (-1.234, 4.362, -4.765)]
        )
        * 1e3
    )
    m2 = (
        np.array(
            [(0.878, -1.527, 2.918), (3.142, -2.863, 1.742), (0.591, 3.218, -2.457)]
        )
        * 1e3
    )
    p1 = np.array(
        [(-1.248, 7.835, 9.273), (2.164, -3.521, 4.896), (6.382, 1.947, -2.135)]
    )
    p2 = np.array(
        [(-2.331, 5.835, 0.578), (1.829, -4.267, 7.093), (-0.516, 2.748, 3.921)]
    )
    piv = np.array(
        [(0.727, 5.152, 5.363), (-1.635, 2.418, 8.091), (4.572, -3.864, 1.247)]
    )
    rot = R.from_rotvec([(55, -126, 222), (43, 165, -32), (94, 2, -23)], degrees=True)
    for i in range(3):
        m2_rot = rot[i].apply(m2[i])
        src1 = magpy.misc.Dipole(position=p1[i], moment=m1[i])
        tgt1 = magpy.misc.Dipole(position=p2[i], moment=m2[i]).rotate(rot[i])
        tgt2 = magpy.misc.Dipole(position=p2[i], moment=m2_rot)

        F0, T0 = FTana(p1[i], p2[i], m1[i], m2_rot, src1, piv[i])
        F1, T1 = getFT(src1, tgt1, pivot=piv[i])
        F2, T2 = getFT(src1, tgt2, pivot=piv[i])

        assert np.linalg.norm(F0 - F1) / np.linalg.norm(F0 + F1) < 1e-10
        assert np.linalg.norm(F0 - F2) / np.linalg.norm(F0 + F2) < 1e-10
        assert np.linalg.norm(T0 - T1) / np.linalg.norm(T0 + T1) < 1e-10
        assert np.linalg.norm(T0 - T2) / np.linalg.norm(T0 + T2) < 1e-10


def test_force_analytic_cocentric_loops():
    """
    compare the numerical solution against the analytical solution of the force between two
    cocentric current loops.
    See e.g. IEEE TRANSACTIONS ON MAGNETICS, VOL. 49, NO. 8, AUGUST 2013
    """
    # random numbers
    z1, z2 = 0.123, 1.321
    i1, i2 = 3.2, 5.1
    r1, r2 = 1.2, 2.3

    # magpylib
    loop1 = magpy.current.Circle(diameter=2 * r1, current=i1, position=(0, 0, z1))
    loop2 = magpy.current.Circle(
        diameter=2 * r2, current=i2, position=(0, 0, z2), meshing=1000
    )
    F, _ = magpy.getFT(loop1, loop2, pivot=(0, 0, 0))
    F_num = F[2]  # force in z-direction

    # analytical solution
    from scipy.special import ellipe, ellipk  # noqa: PLC0415

    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    k = np.sqrt(k2)
    pf = magpy.mu_0 * i1 * i2 * (z1 - z2) * k / 4 / np.sqrt(r1 * r2)
    F_ana = pf * ((2 - k2) / (1 - k2) * ellipe(k**2) - 2 * ellipk(k**2))

    assert abs((F_num - F_ana) / (F_num + F_ana)) < 1e-5


def test_force_analytic_torque_sign():
    """make sure that torque sign is in the right direction"""

    # Cuboid -> Cuboid
    mag1 = magpy.magnet.Cuboid(
        position=(2, 0, 0), polarization=(1, 0, 0), dimension=(2, 1, 1)
    )
    mag2 = magpy.magnet.Cuboid(
        position=(-2, 0, 0), polarization=(1, 0, 0), dimension=(2, 1, 1)
    )

    mag1.rotate_from_angax(15, "y")
    mag1.meshing = (3, 3, 3)

    _, T = magpy.getFT(mag2, mag1)

    assert T[1] < 0

    # Cuboid -> Polyline
    mag = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 2))

    ts = np.linspace(0, 2 * np.pi, 10)
    verts = [(2 * np.cos(t), 2 * np.sin(t), 0) for t in ts]
    loop = magpy.current.Polyline(vertices=verts, current=1)
    loop.rotate_from_angax(15, "y")

    loop.meshing = 20

    _, T = magpy.getFT(mag, loop, pivot=(0, 0, 0))

    assert T[1] < 0


def test_force_analytic_parallel_wires():
    """
    The force between straight infinite parallel wires is
    F = 2*mu0/4/pi * i1*i2/r
    """
    wire1 = magpy.current.Polyline(
        current=1,
        vertices=[(-1000, 0, 0), (1000, 0, 0)],
    )
    wire2 = wire1.copy(position=(0, 0, 1), meshing=1000)
    # force should be attractive
    F, _ = magpy.getFT(wire1, wire2)

    Fanalytic = 2 * magpy.mu_0 / 4 / np.pi * 2000

    assert abs(F[0]) < 1e-14
    assert abs(F[1]) < 1e-14
    assert abs((F[2] + Fanalytic) / Fanalytic) < 1e-3


def test_force_analytic_perpendicular_wires():
    """
    The force between straight infinite perpendicular wires is 0
    """
    wire1 = magpy.current.Polyline(
        current=1,
        vertices=[(-1000, 0, 0), (1000, 0, 0)],
    )
    wire2 = magpy.current.Polyline(
        current=1,
        vertices=[(0, -1000, 0), (0, 0, 0), (0, 1000, 0)],
        position=(0, 0, 1),
        meshing=1000,
    )

    F, _ = magpy.getFT(wire1, wire2)

    assert np.max(abs(F)) < 1e-14


def test_force_analytic_current_in_homo_field():
    """
    for a current loop in a homogeneous field the following holds
    F = 0
    T = current * loop_surface * field_normal_component
    """
    # circular loop
    cloop = magpy.current.Circle(diameter=2, current=-1, meshing=20)

    # homogeneous field
    def func(field, observers):  # noqa:  ARG001
        return np.zeros_like(observers, dtype=float) + np.array((1, 0, 0))

    hom = magpy.misc.CustomSource(field_func=func)

    # without pivot
    F, T = magpy.getFT(hom, cloop, pivot=None)
    assert np.amax(abs(F)) < 1e-14
    assert np.amax(abs(T)) == 0

    # with pivot
    F, T = magpy.getFT(hom, cloop, pivot=cloop.position)
    assert np.amax(abs(F)) < 1e-14
    assert abs(T[0]) < 1e-14
    assert abs(T[1] + np.pi) < 1e-11
    assert abs(T[2]) < 1e-14

    #########################

    # rectangular loop
    verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0)]
    rloop = magpy.current.Polyline(
        current=1,
        vertices=verts,
    )
    rloop.meshing = 20

    # without pivot
    F, T = magpy.getFT(hom, rloop, pivot=None)
    assert np.amax(abs(F)) < 1e-14
    assert np.amax(abs(T)) == 0

    # with pivot
    F, T = magpy.getFT(hom, rloop, pivot=rloop.position)
    T *= -1  # bad sign at initial test design
    assert np.amax(abs(F)) < 1e-14
    assert abs(T[0]) < 1e-14
    assert abs(T[1] + 4) < 1e-12
    assert abs(T[2]) < 1e-14


def test_force_analytic_loop_projection():
    """
    the torque of a rotated loop in a dipole field must fulfill certain geometric relations
    """
    loop = magpy.current.Circle(
        diameter=5,
        current=1e6,
        position=(0, 0, 0),
        meshing=400,
    )
    dip = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, 0))
    _, T1 = getFT(dip, loop, pivot=(0, 0, 0))
    TT0 = T1[1]

    # geometric sum relation
    loop.rotate_from_angax(45, "x")
    _, T1 = getFT(dip, loop, pivot=(0, 0, 0))
    TT1 = T1[1]
    assert TT0 - TT1 * np.sqrt(2) < 1e-12

    # reduced flux at 45° angle
    loop.orientation = None
    loop.rotate_from_angax(45, "y")
    _, T1 = getFT(dip, loop, pivot=(0, 0, 0))
    assert abs(TT1 - T1[1]) < 1e-12

    # reduced flux at -225° angle
    loop.orientation = None
    dip.rotate_from_angax(-225, "y")
    _, T1 = getFT(dip, loop, pivot=(0, 0, 0))
    assert abs(TT1 + T1[1]) < 1e-12


def test_force_analytic_rotation():
    loop = magpy.current.Circle(
        diameter=10,
        current=1e6,
        position=(0, 0, 0),
        meshing=4,
    )
    dip = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, 0))

    _, T1 = getFT(dip, loop, pivot=(0, 0, 0))
    # must have positive torque about y-axis
    assert T1[1] > 0

    loop.rotate_from_angax(90, "x")
    _, T2 = getFT(dip, loop, pivot=(0, 0, 0))
    # positive y-torque becomes positive z-torque
    assert abs(T1[1] - T2[2]) / T1[1] < 1e-10

    loop.rotate_from_angax(90, "x")
    _, T3 = getFT(dip, loop, pivot=(0, 0, 0))
    # positive y-torque becomes negative y-torque after 180° rot
    assert abs(T1[1] + T3[1]) / T1[1] < 1e-10

    loop.rotate_from_angax(90, "x")
    _, T4 = getFT(dip, loop, pivot=(0, 0, 0))
    # positive z-torque becomes negative z-torque after 180° rot
    assert abs(T2[2] + T4[2]) / T1[1] < 1e-10

    loop.rotate_from_angax(90, "x")
    _, T5 = getFT(dip, loop, pivot=(0, 0, 0))
    # back to initial values
    assert abs(T1[1] - T5[1]) / T1[1] < 1e-10

    loop.rotate_from_angax(45, "x")
    _, T6 = getFT(dip, loop, pivot=(0, 0, 0))
    # torque must now be split over y and z component
    assert abs(T6[1] ** 2 + T6[2] ** 2 - T1[1] ** 2) < 1e-10


######################################################################################
######################################################################################
### PATH #############################################################################
######################################################################################
######################################################################################


def test_force_path1():
    """
    Test force calculation with a path
    """
    # circular loop
    cloop1 = magpy.current.Circle(
        diameter=2, current=1, meshing=20, position=[(i, i, i) for i in range(10)]
    )

    # homogeneous field
    def func1(field, observers):  # noqa:  ARG001
        return np.zeros_like(observers, dtype=float) + np.array((1, 0, 0))

    hom1 = magpy.misc.CustomSource(field_func=func1)

    F, T = magpy.getFT(hom1, cloop1)

    assert F.shape == (10, 3)
    assert T.shape == (10, 3)

    assert np.max(np.abs(F)) < 1e-14
    assert np.max(np.abs(T[:, 0])) < 1e-14
    assert np.max(np.abs(T[:, 1] - np.pi)) < 1e-10
    assert np.max(np.abs(T[:, 2])) < 1e-14

    F, T = magpy.getFT(hom1, cloop1, pivot=None)
    assert np.max(np.abs(F)) < 1e-14
    assert np.max(np.abs(T)) < 1e-14


def test_force_path2():
    """
    rotation path
    """
    # circular loop
    cloop1 = magpy.current.Circle(
        diameter=2,
        current=1,
        meshing=20,
    ).rotate_from_angax([0, 45, 90], "x", start=0)

    # homogeneous field
    def func1(field, observers):  # noqa:  ARG001
        return np.zeros_like(observers, dtype=float) + np.array((1, 0, 0))

    hom1 = magpy.misc.CustomSource(field_func=func1)

    F, T = magpy.getFT(hom1, cloop1)

    # F = 0
    assert F.shape == (3, 3)
    assert np.max(np.abs(F)) < 1e-14

    # Tx = 0
    assert np.max(np.abs(T[:, 0])) < 1e-14

    # T shifted by angle projection from y to z
    assert abs(T[0, 1] - np.pi) < 1e-10
    assert abs(T[0, 2]) < 1e-10

    assert abs(T[1, 1] - np.pi / np.sqrt(2)) < 1e-10
    assert abs(T[1, 2] - np.pi / np.sqrt(2)) < 1e-10

    assert abs(T[2, 1]) < 1e-10
    assert abs(T[2, 2] - np.pi) < 1e-10


def test_force_path3():
    """multiple src and tgts"""
    src1 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(1, 2, 3),
        position=(0.5, 0.5, 0.5),
    )
    src2 = src1.copy(polarization=(2, 4, 6))

    verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0)]
    rloop1 = magpy.current.Polyline(
        current=1,
        vertices=verts,
        meshing=8,
    )
    rloop2 = rloop1.copy(current=2)
    rloop3 = rloop1.copy(current=3)

    rloop1.position = [(0, 0, 0)] * 2
    rloop2.position = [(0, 0, 0)] * 3
    src1.position = [(0.5, 0.5, 0.5)] * 4

    F, T = magpy.getFT([src1, src2], [rloop1, rloop2, rloop3])

    assert F.shape == (2, 4, 3, 3)

    assert np.allclose(2 * F[0, 1, 1], F[1, 1, 1])
    assert np.allclose(2 * F[0, 1, 0], F[0, 2, 1])
    assert np.allclose(3 * F[0, 1, 0], F[0, 2, 2])
    assert np.allclose(6 * F[0, 1, 0], F[1, 2, 2])

    assert np.allclose(2 * T[0, 1, 1], T[1, 1, 1])
    assert np.allclose(6 * T[0, 1, 0], T[1, 2, 2])


def test_force_path4():
    """
    with collection
    """
    src1 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(1, 2, 3),
        position=(0.5, 0.5, 0.5),
    )
    src2 = src1.copy(polarization=(2, 4, 6))

    loop1 = magpy.current.Circle(
        current=1,
        diameter=3,
        meshing=20,
    )
    loop2 = loop1.copy(current=2)
    loop3 = loop1.copy(current=3)
    loop4 = loop1.copy(current=4)
    loop5 = loop1.copy(current=5)

    coll1 = magpy.Collection(loop1, loop2)
    coll1.position = [(0, 0, 0)] * 4

    coll2 = magpy.Collection(loop3, loop4)
    coll2.position = [(0, 0, 0)] * 6

    F, T = magpy.getFT([src1, src2], [coll1, loop1, coll2, loop5])

    assert F.shape == (2, 6, 4, 3)
    assert np.allclose(F[0, 0, 0] * 2 * 1, 1 * 1 * F[1, 0, 0])
    assert np.allclose(F[0, 2, 0] * 2 * 7, 1 * 3 * F[1, 4, 2])
    assert np.allclose(T[1, 2, 1] * 1 * 5, 2 * 1 * T[0, 4, 3])


def test_force_path5():
    """different meshings"""
    src1 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(1, 2, 3),
        position=(0.5, 0.5, 0.5),
    )
    src2 = src1.copy(polarization=(2, 4, 6))

    verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0)]
    rloop1 = magpy.current.Polyline(
        current=1,
        vertices=verts,
        meshing=512,
    )
    rloop2 = rloop1.copy(current=2, meshing=256)
    rloop3 = rloop1.copy(current=3, meshing=1024)

    rloop1.position = [(0, 0, 0)] * 2
    rloop2.position = [(0, 0, 0)] * 3
    src1.position = [(0.5, 0.5, 0.5)] * 4

    F, T = magpy.getFT([src1, src2], [rloop1, rloop2, rloop3])

    assert F.shape == (2, 4, 3, 3)

    err = np.linalg.norm(2 * F[0, 1, 1] - F[1, 1, 1]) / np.linalg.norm(F[1, 1, 1])
    assert err < 1e-6

    err = np.linalg.norm(2 * F[0, 1, 0] - F[0, 2, 1]) / np.linalg.norm(F[0, 2, 1])
    assert err < 0.005

    err = np.linalg.norm(3 * F[0, 1, 0] - F[0, 2, 2]) / np.linalg.norm(F[0, 2, 2])
    assert err < 0.003

    err = np.linalg.norm(2 * T[0, 1, 1] - T[1, 1, 1]) / np.linalg.norm(T[1, 1, 1])
    assert err < 1e-6

    err = np.linalg.norm(2 * T[0, 1, 0] - T[0, 2, 1]) / np.linalg.norm(T[0, 2, 1])
    assert err < 0.02


def test_force_path6():
    """
    CORE PATH TEST
    orientation + all kinds of paths combined
    """

    def FTana(p1, p2, m1, m2, src, piv):
        # analytical force solution
        r = p2 - p1
        r_abs = np.linalg.norm(r)
        r_unit = r / r_abs
        F_ana = (
            3
            * magpy.mu_0
            / (4 * np.pi * r_abs**4)
            * (
                np.cross(np.cross(r_unit, m1), m2)
                + np.cross(np.cross(r_unit, m2), m1)
                - 2 * r_unit * np.dot(m1, m2)
                + 5 * r_unit * (np.dot(np.cross(r_unit, m1), np.cross(r_unit, m2)))
            )
        )
        B = src.getB(p2)
        T_ana = np.cross(m2, B) + np.cross(p2 - piv, F_ana)
        return F_ana, T_ana

    # random numbers
    m1 = (
        np.array(
            [(0.976, 4.304, 2.055), (1.432, 0.352, 2.345), (-1.234, 4.362, -4.765)]
        )
        * 1e3
    )
    m2 = (
        np.array(
            [(0.878, -1.527, 2.918), (3.142, -2.863, 1.742), (0.591, 3.218, -2.457)]
        )
        * 1e3
    )
    p1 = np.array(
        [(-1.248, 7.835, 9.273), (2.164, -3.521, 4.896), (6.382, 1.947, -2.135)]
    )
    p2 = np.array(
        [(-2.331, 5.835, 0.578), (1.829, -4.267, 7.093), (-0.516, 2.748, 3.921)]
    )
    piv = np.array(
        [(0.727, 5.152, 5.363), (-1.635, 2.418, 8.091), (4.572, -3.864, 1.247)]
    )
    rot = R.from_rotvec([(55, -126, 222), (43, 165, -32), (94, 2, -23)], degrees=True)
    for i in range(3):
        m2_rot = rot[i].apply(m2[i])

        src1 = magpy.misc.Dipole(position=p1[i], moment=m1[i])
        src2 = magpy.misc.Dipole(position=[p1[i]] * 4, moment=m1[i])

        tgt1 = magpy.misc.Dipole(position=[p2[i]] * 4, moment=m2[i]).rotate(rot[i])
        tgt2 = magpy.misc.Dipole(position=p2[i], moment=m2_rot)
        tgt3 = magpy.current.Polyline(
            current=1,
            vertices=[(-0.1, 0, 0), (0.1, 0, 0)],
            position=[(2, 2, 2), (3, 3, 3), (4, 4, 4)],
            meshing=10,
        )
        tgt4 = magpy.current.Polyline(
            current=1, vertices=[(-0.1, 0, 0), (0.1, 0, 0)], meshing=10
        )
        tgt5 = magpy.magnet.Cylinder(
            dimension=(0.2, 0.2), polarization=(2, 3, 1), meshing=7
        )

        F0, T0 = FTana(p1[i], p2[i], m1[i], m2_rot, src1, piv[i])

        F, T = getFT(
            [src1, src1], [tgt2, tgt2, tgt4, tgt2], pivot=piv[i], squeeze=False
        )  # no path
        assert F.shape == (2, 1, 4, 3)
        assert np.amax(abs(F[:, :, :2] - F0)) / np.linalg.norm(F0) < 1e-9
        assert np.amax(abs(T[:, :, :2] - T0)) / np.linalg.norm(T0) < 1e-9

        F, T = getFT(
            [src1, src1], [tgt1, tgt2, tgt4, tgt2], pivot=piv[i], squeeze=False
        )  # tgt_path, no src_path
        assert F.shape == (2, 4, 4, 3)
        assert np.amax(abs(F[:, :, :2] - F0)) / np.linalg.norm(F0) < 1e-9
        assert np.amax(abs(T[:, :, :2] - T0)) / np.linalg.norm(T0) < 1e-9

        F, T = getFT(
            [src2, src1], [tgt1, tgt2, tgt2, tgt5, tgt3, tgt4, tgt2, tgt5], pivot=piv[i]
        )  # tgt_path + src_path
        assert F.shape == (2, 4, 8, 3)
        assert np.amax(abs(F[:, :, :3] - F0)) / np.linalg.norm(F0) < 1e-9
        assert np.amax(abs(T[:, :, :3] - T0)) / np.linalg.norm(T0) < 1e-9


def test_force_path7a():
    """
    CORE TEST 180° ROTATION bug
    source without path
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=(0, 0, 5),
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT(src, dip, pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT(src, sqloop, pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT(src, cube, pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    assert abs(F1[0, 2] + F1[1, 2]) < 1e-14  # dipole
    assert abs(F2[0, 2] + F2[1, 2]) < 1e-14  # loop
    assert abs(F3[0, 2] + F3[1, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=1) / np.linalg.norm(F1 + F2, axis=1)
    assert max(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=1) / np.linalg.norm(F2 + F3, axis=1)
    assert max(errF23) < 0.0002


def test_force_path7b():
    """
    CORE TEST 180° ROTATION bug
    2 sources without path
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=(0, 0, 5),
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT([src, src], dip, pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT([src, src], sqloop, pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT([src, src], cube, pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for i in range(2):
        assert abs(F1[i, 0, 2] + F1[i, 1, 2]) < 1e-14  # dipole
        assert abs(F2[i, 0, 2] + F2[i, 1, 2]) < 1e-14  # loop
        assert abs(F3[i, 0, 2] + F3[i, 1, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=2) / np.linalg.norm(F1 + F2, axis=2)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=2) / np.linalg.norm(F2 + F3, axis=2)
    assert np.amax(errF23) < 0.0002


def test_force_path7c():
    """
    CORE TEST 180° ROTATION bug
    source with path
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180, 0, 180, 0, 180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=[(0, 0, 5), (0, 0, 5), (0, 0, 6), (0, 0, 6), (0, 0, 7), (0, 0, 7)],
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT(src, dip, pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT(src, sqloop, pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT(src, cube, pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for i in range(3):
        assert abs(F1[2 * i, 2] + F1[2 * i + 1, 2]) < 1e-14  # dipole
        assert abs(F2[2 * i, 2] + F2[2 * i + 1, 2]) < 1e-14  # loop
        assert abs(F3[2 * i, 2] + F3[2 * i + 1, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=1) / np.linalg.norm(F1 + F2, axis=1)
    assert max(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=1) / np.linalg.norm(F2 + F3, axis=1)
    assert max(errF23) < 0.0002


def test_force_path7d():
    """
    CORE TEST 180° ROTATION bug
    2 sources with path
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180, 0, 180, 0, 180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)
    angs2 = [180, 0, 180, 0, 180]
    for obj in [cube, dip, sqloop]:
        obj.reset_path()
        obj.rotate_from_angax(angs2, "x")
    src = magpy.magnet.Sphere(
        position=[(0, 0, 5), (0, 0, 5), (0, 0, 6), (0, 0, 6), (0, 0, 7), (0, 0, 7)],
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT([src, src], dip, pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT([src, src], sqloop, pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT([src, src], cube, pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for j in range(2):
        for i in range(3):
            assert abs(F1[j, 2 * i, 2] + F1[j, 2 * i + 1, 2]) < 1e-14  # dipole
            assert abs(F2[j, 2 * i, 2] + F2[j, 2 * i + 1, 2]) < 1e-14  # loop
            assert abs(F3[j, 2 * i, 2] + F3[j, 2 * i + 1, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=2) / np.linalg.norm(F1 + F2, axis=2)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=2) / np.linalg.norm(F2 + F3, axis=2)
    assert np.amax(errF23) < 0.0002


def test_force_path7e():
    """
    CORE TEST 180° ROTATION bug
    source without path 2 targets
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=(0, 0, 5),
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT(src, [dip, dip], pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT(src, [sqloop, sqloop], pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT(src, [cube, cube], pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for i in range(2):
        assert abs(F1[0, i, 2] + F1[1, i, 2]) < 1e-14  # dipole
        assert abs(F2[0, i, 2] + F2[1, i, 2]) < 1e-14  # loop
        assert abs(F3[0, i, 2] + F3[1, i, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=2) / np.linalg.norm(F1 + F2, axis=2)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=2) / np.linalg.norm(F2 + F3, axis=2)
    assert np.amax(errF23) < 0.0002


def test_force_path7f():
    """
    CORE TEST 180° ROTATION bug
    2 sources without path 2 targets
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=(0, 0, 5),
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT([src, src], [dip, dip], pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT([src, src], [sqloop, sqloop], pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT([src, src], [cube, cube], pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for j in range(2):
        for i in range(2):
            assert abs(F1[i, 0, j, 2] + F1[i, 1, j, 2]) < 1e-14  # dipole
            assert abs(F2[i, 0, j, 2] + F2[i, 1, j, 2]) < 1e-14  # loop
            assert abs(F3[i, 0, j, 2] + F3[i, 1, j, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=3) / np.linalg.norm(F1 + F2, axis=3)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=3) / np.linalg.norm(F2 + F3, axis=3)
    assert np.amax(errF23) < 0.0002


def test_force_path7g():
    """
    CORE TEST 180° ROTATION bug
    source with path 2 targets
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180, 0, 180, 0, 180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    src = magpy.magnet.Sphere(
        position=[(0, 0, 5), (0, 0, 5), (0, 0, 6), (0, 0, 6), (0, 0, 7), (0, 0, 7)],
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT(src, [dip, dip], pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT(src, [sqloop, sqloop], pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT(src, [cube, cube], pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for j in range(2):
        for i in range(3):
            assert abs(F1[2 * i, j, 2] + F1[2 * i + 1, j, 2]) < 1e-14  # dipole
            assert abs(F2[2 * i, j, 2] + F2[2 * i + 1, j, 2]) < 1e-14  # loop
            assert abs(F3[2 * i, j, 2] + F3[2 * i + 1, j, 2]) < 1e-12  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=2) / np.linalg.norm(F1 + F2, axis=2)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=2) / np.linalg.norm(F2 + F3, axis=2)
    assert np.amax(errF23) < 0.0002


def test_force_path7h():
    """
    CORE TEST 180° ROTATION bug
    2 sources with path 2 targets
    """
    a = 0.2
    b = 0.3
    h = 0.1
    pz = 100
    angles = [180, 0, 180, 0, 180]
    axis = [1, 0, 0]
    anch = (0, 0, 0)

    # magnet
    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
        meshing=(12, 18, 6),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # moment
    mom = a * b * h * cube.magnetization
    dip = magpy.misc.Dipole(moment=mom).rotate_from_angax(
        angles, axis=axis, anchor=anch
    )

    # current
    i0 = np.linalg.norm(mom) / (a * b)
    sqloop = magpy.current.Polyline(
        vertices=[
            (-a / 2, -b / 2, 0),
            (a / 2, -b / 2, 0),
            (a / 2, b / 2, 0),
            (-a / 2, b / 2, 0),
            (-a / 2, -b / 2, 0),
        ],
        current=i0,
        meshing=40,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)
    angs2 = [180, 0, 180, 0, 180]
    for obj in [cube, dip, sqloop]:
        obj.reset_path()
        obj.rotate_from_angax(angs2, "x")
    src = magpy.magnet.Sphere(
        position=[(0, 0, 5), (0, 0, 5), (0, 0, 6), (0, 0, 6), (0, 0, 7), (0, 0, 7)],
        diameter=0.23,
        polarization=(0, 0, 100),
    )

    F1, _T1 = magpy.getFT([src, src], [dip, dip], pivot=(0, 0, 0))
    F2, _T2 = magpy.getFT([src, src], [sqloop, sqloop], pivot=(0, 0, 0))
    F3, _T3 = magpy.getFT([src, src], [cube, cube], pivot=(0, 0, 0))

    # 180° rotated must give same values negative
    for k in range(2):
        for j in range(2):
            for i in range(3):
                assert (
                    abs(F1[j, 2 * i, k, 2] + F1[j, 2 * i + 1, k, 2]) < 1e-14
                )  # dipole
                assert abs(F2[j, 2 * i, k, 2] + F2[j, 2 * i + 1, k, 2]) < 1e-14  # loop
                assert (
                    abs(F3[j, 2 * i, k, 2] + F3[j, 2 * i + 1, k, 2]) < 1e-12
                )  # current

    # dipole approximation vs loop
    errF12 = np.linalg.norm(F1 - F2, axis=3) / np.linalg.norm(F1 + F2, axis=3)
    assert np.amax(errF12) < 0.002

    # equivalence loop and cube
    errF23 = np.linalg.norm(F2 - F3, axis=3) / np.linalg.norm(F2 + F3, axis=3)
    assert np.amax(errF23) < 0.0002


#############################################################################
#############################################################################
### BACK-FORWARD TESTS ######################################################
#############################################################################
#############################################################################


def test_force_backforward_dipole_circle():
    """
    test backward and forward force on dipole in circle
    test meshing convergence
    """
    loop = magpy.current.Circle(
        diameter=5,
        current=1e6,
        position=(0, 0, 0),
    ).rotate_from_angax(
        [10, 20, 55, 70, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1e3, 0, 0),
        position=np.linspace((-0.5, -0.4, -0.3), (0.3, 0.4, -0.2), 10),
    )

    F0, T0 = getFT(loop, dip, pivot=(0, 0, 0))

    for meshing, err in zip([120, 360, 1080], [1e-3, 1e-4, 1e-5], strict=False):
        loop.meshing = meshing
        F1, T1 = getFT(dip, loop, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_polyline():
    """
    test backward and forward force on dipole and closed polyline
    test meshing convergence
    """
    vertices = ((-3, -3, -3), (3, 3, -3), (3, 3, 2), (-2, -1, 3), (-3, -3, -3))
    loop = magpy.current.Polyline(
        vertices=vertices,
        current=1e6,
        position=(0, 0, 0),
    ).rotate_from_angax(
        [10, 20, 55, 70, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1e3, 0, 0),
        position=np.linspace((-0.5, -0.4, -0.3), (0.3, 0.4, -0.2), 10),
    )

    F0, T0 = getFT(loop, dip, pivot=(0, 0, 0))

    for meshing, err in zip([120, 360, 1080], [1e-3, 1e-4, 1e-5], strict=False):
        loop.meshing = meshing
        F1, T1 = getFT(dip, loop, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_sphere():
    """
    test backward and forward force on dipole and sphere
    test meshing convergence
    """
    sphere = magpy.magnet.Sphere(
        diameter=2.2,
        polarization=(1.1, 2.1, -3.2),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.2, 0.3, 0.1),
    )
    dip = magpy.misc.Dipole(
        moment=(1e3, -1e3, 2e3),
        position=np.linspace((-0.5, -0.4, -1.3), (0.3, 0.4, -1.2), 10),
    )

    # SPECIAL CASE - FORCE ON SPHERE = FORCE ON DIPOLE MOMENT

    F0, T0 = getFT(sphere, dip, pivot=(0, 0, 0))
    F1, T1 = getFT(dip, sphere, pivot=(0, 0, 0))

    errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
    assert errF < 1e-6, f"Force mismatch: {errF}"
    errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
    assert errT < 1e-6, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_cuboid():
    """
    test backward and forward force on dipole and cuboid
    test meshing convergence
    """
    cube = magpy.magnet.Cuboid(
        dimension=(3, 2, 1),
        polarization=(1.2, 2.3, -3.1),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1.3e3, -1.1e3, 2.2e3),
        position=np.linspace((-5, -0.4, -1.3), (3, 1.4, -1.2), 10),
    )

    F0, T0 = getFT(cube, dip, pivot=(0, 0, 0))

    for meshing, err in zip([5, 120, 360], [1e-1, 1e-2, 1e-3], strict=False):
        cube.meshing = meshing
        F1, T1 = getFT(dip, cube, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err * 2, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_cylinder():
    """
    test backward and forward force on dipole and cylinder
    test meshing convergence
    """
    cylinder = magpy.magnet.Cylinder(
        dimension=(2, 1),
        polarization=(1.2, 2.3, -3.1),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1.3e3, -1.1e3, 2.2e3),
        position=np.linspace((-5, -0.4, -1.3), (3, 1.4, -1.2), 10),
    )

    F0, T0 = getFT(cylinder, dip, pivot=(0, 0, 0))

    for meshing, err in zip([1, 20, 360], [0.3, 1e-1, 1e-2], strict=False):
        cylinder.meshing = meshing
        F1, T1 = getFT(dip, cylinder, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err * 2, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_cylinderSegment():
    """
    test backward and forward force on dipole and cylinder
    test meshing convergence
    """
    cylseg = magpy.magnet.CylinderSegment(
        dimension=(1, 2, 1, 45, 123),
        polarization=(1.2, 2.3, -3.1),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1.3e3, -1.1e3, 2.2e3),
        position=np.linspace((-5, -0.4, -1.3), (3, 1.4, -1.2), 10),
    )

    F0, T0 = getFT(cylseg, dip, pivot=(0, 0, 0))

    for meshing, err in zip([1, 20, 360], [0.6, 1e-1, 1e-2], strict=False):
        cylseg.meshing = meshing
        F1, T1 = getFT(dip, cylseg, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err, f"Torque mismatch: {errT}"


def test_force_backforward_Dipole_Tetrahedron():
    """
    test backward and forward force on dipole and tetrahedron
    test meshing convergence
    """
    verts = [(0, 0, -0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    tetra = magpy.magnet.Tetrahedron(
        vertices=verts,
        polarization=(1.2, 2.3, -3.1),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1.3e3, -1.1e3, 2.2e3),
        position=np.linspace((-5, -0.4, -1.3), (3, 1.4, -1.2), 10),
    )

    F0, T0 = getFT(tetra, dip, pivot=(0, 0, 0))

    for meshing, err in zip([1, 20, 300], [0.08, 0.1, 0.07], strict=False):
        tetra.meshing = meshing
        F1, T1 = getFT(dip, tetra, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err * 2.1, f"Torque mismatch: {errT}"


def test_force_backforward_Dipole_Trimesh():
    """
    test backward and forward force on dipole and TriangularMesh
    test meshing convergence
    """
    verts = [(0, 0, -0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    trimesh = magpy.magnet.TriangularMesh(
        vertices=verts,
        faces=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        polarization=(1.2, 2.3, -3.1),
    ).rotate_from_angax(
        [11, 24.3, 55.2, 76, 20, 10, 15, 20, -123.1234, 1234],
        axis=(1, 2, -3),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1.3e3, -1.1e3, 2.2e3),
        position=np.linspace((-5, -0.4, -1.3), (3, 1.4, -1.2), 10),
    )

    F0, T0 = getFT(trimesh, dip, pivot=(0, 0, 0))

    for meshing, err in zip(
        [1, 60, 100], [0.08, 0.1, 0.01], strict=False
    ):  # careful, 100 is a sweet spot
        trimesh.meshing = meshing
        F1, T1 = getFT(dip, trimesh, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < err, f"Force mismatch: {errF}"
        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < err * 2.1, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_tristrip():
    """
    test backward and forward force on dipole in triangle strip
    test meshing convergence
    """

    strip = magpy.current.TriangleStrip(
        current=1e6,
        vertices=(
            (-3, -3, -3),
            (-3, -3, 3),
            (3.3, -3, -3),
            (3, -3.3, 3),
            (0, 5.2, -3),
            (0, 5, 3.2),
            (-3, -3, -3),
            (-3, -3, 3),
        ),
    ).rotate_from_angax(
        [10, 20, 55, -20, 180, -123.1234],
        axis=(1.1, 2.1, -3.2),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1e3, 0, 0),
        position=np.linspace((-0.5, -0.4, -0.3), (0.3, 0.4, -0.2), 6),
    )

    F0, T0 = getFT(strip, dip, pivot=(0, 0, 0))

    for meshing, errf, errt in zip(
        [500, 2000, 5000], [0.08, 0.007, 0.0007], [0.003, 0.0004, 7e-5], strict=False
    ):
        strip.meshing = meshing
        F1, T1 = getFT(dip, strip, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < errf, f"Force mismatch: {errF}"

        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < errt, f"Torque mismatch: {errT}"


def test_force_backforward_dipole_trisheet():
    """
    test backward and forward force on dipole in triangle sheet
    test meshing convergence
    """

    strip = magpy.current.TriangleSheet(
        current_densities=[(1e6, 0, 0)] * 2
        + [(0, 1e6, 0)] * 2
        + [(-1e6, 0, 0)] * 2
        + [(0, -1e6, 0)] * 2,
        vertices=(
            (-3, -3, -3),
            (-3, -3, 3),
            (3, -3, -3),
            (3, -3, 3),
            (3, 3, -3),
            (3, 3, 3),
            (-3, 3, -3),
            (-3, 3, 3),
            (-3, -3, -3),
            (-3, -3, 3),
        ),
        faces=(
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
            (4, 5, 6),
            (5, 6, 7),
            (6, 7, 0),
            (7, 0, 1),
        ),
    ).rotate_from_angax(
        [10, 20, 55, -20, 180, -123.1234],
        axis=(1.1, 2.1, -3.2),
        anchor=(0.1, 0.2, 0.3),
    )
    dip = magpy.misc.Dipole(
        moment=(1e3, 0, 0),
        position=np.linspace((-0.5, -0.4, -0.3), (0.3, 0.4, -0.2), 6),
    )

    F0, T0 = getFT(strip, dip, pivot=(0, 0, 0))

    for meshing, errf, errt in zip(
        [50, 300, 2000], [0.35, 0.02, 0.003], [0.035, 0.003, 0.0004], strict=False
    ):
        strip.meshing = meshing
        F1, T1 = getFT(dip, strip, pivot=(0, 0, 0))

        errF = np.max(np.linalg.norm(F1 + F0, axis=1) / np.linalg.norm(F1 - F0, axis=1))
        assert errF < errf, f"Force mismatch: {errF}"

        errT = np.max(np.linalg.norm(T1 + T0, axis=1) / np.linalg.norm(T1 - T0, axis=1))
        assert errT < errt, f"Torque mismatch: {errT}"


#################################################################################
#################################################################################
### FEM #########################################################################
#################################################################################
#################################################################################


def test_force_ANSYS_cube_cube():
    """
    compare to ANSYS values using EXP04
    """
    dat = np.array(
        (
            (
                1,
                7,
                2,
                377.854600016549,
                -188.542528864873,
                -92.8990855112117,
                -313.999320817873,
                141.297867367292,
                -298.077476393508,
                2.79328118297203,
            ),
            (
                1,
                7,
                4,
                222.97324592523,
                -196.942454862682,
                -45.7293146685391,
                -94.0189749520763,
                103.019653363118,
                -499.3139545593,
                14.0657284644094,
            ),
            (
                1,
                10,
                2,
                109.392835475382,
                -40.734672622913,
                -19.9251847187105,
                -99.5513230126722,
                43.6991462807681,
                -97.0400812701753,
                0.963515112458678,
            ),
            (
                1,
                10,
                4,
                82.5409955281411,
                -57.0911591058395,
                -13.7888753933175,
                -57.9955378465955,
                37.2506186840298,
                -167.207747894386,
                1.88952437302015,
            ),
            (
                3,
                7,
                2,
                268.044882092576,
                -122.351045384398,
                -181.533246425662,
                -154.673724179409,
                370.592326952291,
                -258.784762668333,
                4.13894019162516,
            ),
            (
                3,
                7,
                4,
                164.713887772005,
                -128.609022516067,
                -95.5090438696884,
                -38.319795035053,
                268.387524559781,
                -363.857009255895,
                3.85642340519037,
            ),
            (
                3,
                10,
                2,
                87.2315925006503,
                -30.1457939660204,
                -48.080701791827,
                -66.2482297994683,
                142.329658703303,
                -76.1110023199789,
                -5.72426828151665,
            ),
            (
                3,
                10,
                4,
                66.3151258655567,
                -44.4688881017354,
                -32.4675166815565,
                -36.9604419628989,
                104.515519563477,
                -150.862082328095,
                3.51331256342416,
            ),
        )
    )
    dat[:, (0, 1, 2)] = dat[:, (2, 0, 1)]  # correct bad xyz-order

    tgt_pos = dat[:, (0, 1, 2)]
    F_fe = dat[:, (4, 5, 6)]
    T_fe = dat[:, (7, 8, 9)]
    gen = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5, 5, 5),
    )
    tgt = magpy.magnet.Cuboid(
        dimension=(0.3, 0.3, 0.3),
        polarization=(0, 0, 1),
    )
    tgt.meshing = (10, 10, 10)

    for i, poz in enumerate(tgt_pos):
        tgt.position = poz
        F, T = getFT(gen, tgt)
        T *= -1  # bad sign in original implementation

        errF = np.linalg.norm(F - F_fe[i]) / np.linalg.norm(F)
        assert errF < 0.04
        errT = np.linalg.norm(T - T_fe[i]) / np.linalg.norm(T)
        assert errT < 0.25


def test_force_ANSYS_loop_loop():
    """
    compare to ANSYS loop-loop computation
    Warning: ANSYS computes force very inaccurately and torque is completely off
    """
    data = (
        (
            0.5,
            0.25,
            -500,
            -0.378990675610971,
            1.16471535508468,
            -3.93840492731175,
            0.316391999895969,
            -0.323117059194582,
            0.554105980303591,
        ),
        (
            0.5,
            0.25,
            500,
            0.52319608703649,
            0.477119683892785,
            -3.95525312585684,
            -0.344823150943224,
            -0.32983037884051,
            0.548719111194436,
        ),
        (
            0.5,
            1.75,
            -500,
            -0.247318906984941,
            -0.475750180468637,
            2.20298358334411,
            -0.112567180482201,
            0.0876390943086132,
            0.40936448187346,
        ),
        (
            0.5,
            1.75,
            500,
            0.271711794735348,
            -0.229983825017913,
            2.51566260990816,
            0.091544957851064,
            0.0899073023482688,
            0.413415626972563,
        ),
        (
            1.5,
            0.25,
            -500,
            0.938566291200263,
            -2.02436288094611,
            -7.4080821527953,
            -0.618507366424736,
            1.99422271236167,
            3.65138404864739,
        ),
        (
            1.5,
            0.25,
            500,
            -0.00216325664572926,
            -2.1084068976844,
            -6.54170362000025,
            0.5995369738223,
            2.00706353179035,
            3.67068375878204,
        ),
        (
            1.5,
            0.5,
            500,
            -0.523860907171043,
            -1.99575639650359,
            -0.940337877715457,
            0.429197197716147,
            1.28142571497163,
            1.8011336159446,
        ),
        (
            1.5,
            1.75,
            -500,
            -0.328463925710092,
            -0.738926572407658,
            3.6506362987988,
            -0.0893135956299432,
            0.26486293279763,
            0.263853801870998,
        ),
        (
            1.5,
            1.75,
            500,
            -0.143145542837849,
            -0.277054173217136,
            2.16655477926914,
            0.0843493800576269,
            0.246120004205157,
            0.234717784641934,
        ),
    )

    i_squ = 0.5
    i_circ = 10

    verts1 = (
        np.array(
            (
                (0.5, 0.5, 0),
                (-0.5, 0.5, 0),
                (-0.5, -0.5, 0),
                (0.5, -0.5, 0),
                (0.5, 0.5, 0),
            )
        )
        * 1e-3
    )
    sloop = magpy.current.Polyline(
        vertices=verts1,
        current=i_squ,
    )
    sloop.meshing = 100
    ts = np.linspace(0, 2 * np.pi, 100)
    verts2 = 1.975 * np.array([(np.cos(t), np.sin(t), 0) for t in ts]) * 1e-3
    cloop = magpy.current.Polyline(vertices=verts2, current=i_circ)
    cloop.meshing = 3

    for d in data:
        c1y, c1z, c1x = d[:3]
        pos = np.array((c1x * 1e-3, c1y, c1z)) * 1e-3
        cloop.position = pos

        # fem force
        F2 = d[6:9]

        # analytical force
        F3, _ = getFT(sources=cloop, targets=sloop)
        F3 *= 1e6

        err = np.linalg.norm(F2 - F3) / np.linalg.norm(F3)
        assert err < 0.2


def test_force_ANSYS_loop_magnet():
    """
    compare to FEM solution
    """
    # "yy [mm]","zz [mm]","xx [um]","F_magnet.Force_mag [mNewton]","F_magnet.Force_x [mNewton]","F_magnet.Force_y [mNewton]","F_magnet.Force_z [mNewton]","F_square.Force_mag [mNewton]","F_square.Force_x [mNewton]","F_square.Force_y [mNewton]","F_square.Force_z [mNewton]","Fv_magnet.Force_mag [mNewton]","Fv_magnet.Force_x [mNewton]","Fv_magnet.Force_y [mNewton]","Fv_magnet.Force_z [mNewton]"
    dataF = np.array(
        (
            (
                0.8,
                0.1,
                200,
                6.73621798696924e-15,
                -3.584546668289e-15,
                4.7801359239701e-16,
                -5.68323507839592e-15,
                16.6586082371456,
                -11.831409840104,
                4.03959186599999,
                11.0094807847751,
                16.6805051117026,
                11.8433422379711,
                -4.05649710352454,
                -11.0235804829885,
            ),
            (
                0.8,
                0.1,
                800,
                1.92251067162549e-15,
                4.23204435527338e-16,
                1.17475208106657e-15,
                -1.46181491177701e-15,
                13.3809780256917,
                4.44706302385542,
                5.81654806197601,
                11.2000880366462,
                13.4481224103919,
                -4.41614382881405,
                -5.86900194663829,
                -11.2651891328316,
            ),
            (
                1,
                0.1,
                800,
                0,
                0,
                0,
                0,
                10.5206977015748,
                3.18905970265248,
                6.23300504192764,
                7.85268275738579,
                10.5186835277485,
                -3.19801296156644,
                -6.2332692626082,
                -7.84613092896142,
            ),
            (
                2,
                0.1,
                800,
                0,
                0,
                0,
                0,
                0.85304709445094,
                -0.123455805806032,
                0.800549927287909,
                -0.267521631430614,
                0.821752082865582,
                0.0413733953267691,
                -0.783525909803599,
                0.244237336456775,
            ),
            (
                0.8,
                0.8,
                200,
                0,
                0,
                0,
                0,
                2.84620865578413,
                -2.56083873331669,
                0.556365961005322,
                1.11061497002341,
                2.86935053705004,
                2.60797676976126,
                -0.418458292630111,
                -1.12094706841316,
            ),
            (
                0.8,
                0.8,
                800,
                0,
                0,
                0,
                0,
                2.54654428882465,
                -0.0827582901825677,
                0.976845061909694,
                2.35027926114625,
                2.56309182955053,
                0.160281418778475,
                -1.03704319103324,
                -2.33843772921894,
            ),
        )
    )
    # "yy [mm]","zz [mm]","xx [um]","Tvx_m.Torque [uNewtonMeter]","Tvy_m.Torque [uNewtonMeter]","Tvz_m.Torque [uNewtonMeter]","Tx_square.Torque [uNewtonMeter]","Ty_square.Torque [uNewtonMeter]","Tz_square.Torque [uNewtonMeter]"
    dataT = np.array(
        (
            (
                0.8,
                0.1,
                200,
                -0.152752678323172,
                2.53687184879894,
                0.31545404743791,
                0.140313023228343,
                -2.53736199874947,
                -0.30517463426644,
            ),
            (
                0.8,
                0.1,
                800,
                0.959480198170481,
                2.89789848899664,
                -0.435258330080741,
                -0.870512145889583,
                -2.87366675400084,
                0.430657575370342,
            ),
            (
                1,
                0.1,
                800,
                0.800553299865047,
                2.17870450804352,
                -0.381791156481078,
                -0.81345605829445,
                -2.17610385228856,
                0.398898899795887,
            ),
            (
                2,
                0.1,
                800,
                0.853704644034881,
                0.427380408643971,
                -0.0124517565704488,
                -0.888547449837403,
                -0.41747637388676,
                0.0181451681407211,
            ),
            (
                0.8,
                0.8,
                200,
                0.0940961486719588,
                1.24970685094319,
                0.00346834797401525,
                -0.118237549519446,
                -1.18243794063886,
                -0.0161649360104749,
            ),
            (
                0.8,
                0.8,
                800,
                0.340487764979492,
                0.756762736556279,
                -0.0413327662068715,
                -0.346866763509663,
                -0.707475285454173,
                0.00988830084537582,
            ),
        )
    )

    verts1 = (
        np.array(
            (
                (0.5, 0.5, 0),
                (-0.5, 0.5, 0),
                (-0.5, -0.5, 0),
                (0.5, -0.5, 0),
                (0.5, 0.5, 0),
            )
        )
        * 1e-3
    )
    loop = magpy.current.Polyline(
        vertices=verts1,
        current=50,
    )
    magnet = magpy.magnet.Cuboid(
        dimension=np.array((1, 2, 1)) * 1e-3,
        polarization=(1, 0, 0),
    )

    for dat, dat2 in zip(dataF, dataT, strict=False):
        c1y, c1z, c1x = dat[:3]
        pos = np.array((c1x * 1e-3, c1y, c1z)) * 1e-3
        pos[2] += 0.5 * 1e-3
        magnet.position = pos

        c1yb, c1zb, c1xb = dat2[:3]
        pos2 = np.array((c1xb * 1e-3, c1yb, c1zb)) * 1e-3
        pos2[2] += 0.5 * 1e-3
        np.testing.assert_allclose(pos, pos2)

        F2 = dat[8:11]
        F1 = dat[12:15]
        T2 = dat2[3:6]
        T1 = dat2[6:9]

        loop.meshing = 1000
        magnet.meshing = (10, 20, 10)
        F3, T3 = getFT(sources=loop, targets=magnet, pivot=(0, 0, 0))
        T3 *= -1  # bad sign at initial test design
        F4, T4 = getFT(sources=magnet, targets=loop, pivot=(0, 0, 0))
        T4 *= -1  # bad sign at initial test design
        F3 *= 1e3
        F4 *= 1e3
        T3 *= 1e6
        T4 *= 1e6

        err = np.linalg.norm(F2 + F3) / np.linalg.norm(F3)
        assert err < 0.15
        err = np.linalg.norm(F1 + F4) / np.linalg.norm(F4)
        assert err < 0.15
        err = np.linalg.norm(T2 + T3) / np.linalg.norm(T3)
        assert err < 0.15
        err = np.linalg.norm(T1 + T4) / np.linalg.norm(T4)
        assert err < 0.15


def test_force_ANSYS_magnet_current_close():
    """current loop close to magnet"""

    magnet = magpy.magnet.Cuboid(
        dimension=np.array((0.5, 10, 0.3)) * 1e-3,
        polarization=(1, 0, 0),
    )
    magnet.meshing = (5, 50, 3)

    # wire spit up into 4 parts
    d = 0.025  # put PolyLine in center of crosssection
    t = 0.01  # put PolyLine in center of crosssection
    verts1 = (
        np.array(
            (
                (-0.25 + d, -4 + d, t),
                (0.25 - d, -4 + d, t),
                (0.25 - d, 4 - d, t),
                (-0.25 + d, 4 - d, t),
                (-0.25 + d, -4 + d, t),
            )
        )
        * 1e-3
    )
    discr = 10 * 1e3  # wire discretizations per meter
    wires = []
    for i in range(4):
        wire = magpy.current.Polyline(vertices=(verts1[i : i + 2]))
        mw = int(discr * np.linalg.norm(verts1[i] - verts1[i + 1])) + 1
        wire.meshing = mw
        wires.append(wire)

    # "I_square [mA]","yy [mm]","zz [mm]","xx [um]","F_square.Force_x [uNewton]","F_square.Force_y [uNewton]","F_square.Force_z [uNewton]","Fv_magnet.Force_x [uNewton]","Fv_magnet.Force_y [uNewton]","Fv_magnet.Force_z [uNewton]"
    datF = np.array(
        (
            (
                50,
                0,
                0.2,
                200,
                -38.697482833418,
                0.0002242350531848,
                59.4310879419995,
                29.0564031184439,
                7.11803882106958,
                -47.9368393911027,
            ),
            (
                50,
                0,
                0.2,
                500,
                31.1552670207915,
                -0.00142810245237877,
                29.8624423913996,
                -65.3924604890968,
                11.4789019916509,
                -51.8408330694864,
            ),
            (
                50,
                0,
                0.2,
                800,
                15.1908535165256,
                -3.87787107323596e-05,
                -2.59112312011276,
                -10.2503856180475,
                3.77071833140656,
                -40.2905673214654,
            ),
            (
                50,
                1,
                0.2,
                200,
                -38.2166912173904,
                0.74775728098968,
                59.1285115576116,
                -7.1275613537107,
                -8.16434162283945,
                -70.8158765538541,
            ),
            (
                50,
                1,
                0.2,
                500,
                30.9248391472397,
                0.711934423270261,
                29.7711197385536,
                -47.0335059106064,
                5.30602983129916,
                -94.6243138266417,
            ),
            (
                50,
                1,
                0.2,
                800,
                14.9818522746782,
                0.285580482211302,
                -2.48002525728516,
                -17.4778153030685,
                14.3839498468311,
                -33.3455008748144,
            ),
            (
                50,
                0,
                0.5,
                200,
                -15.4440842393698,
                0.000630278271938781,
                13.8040620063476,
                -29.3586587495271,
                -7.04552102088018,
                -31.9675710200005,
            ),
            (
                50,
                0,
                0.5,
                500,
                2.40460818037774,
                -0.000453637988892101,
                14.3771835667471,
                59.7897254642478,
                -7.72738474941987,
                -5.30574925396906,
            ),
            (
                50,
                0,
                0.5,
                800,
                6.6367471620929,
                0.000255164078938144,
                4.82949919020895,
                -18.5474946200134,
                1.33369577647253,
                -10.881515270821,
            ),
            (
                50,
                1,
                0.5,
                200,
                -15.145226144758,
                0.235812935580012,
                13.6277157811128,
                -1.85120493391665,
                -2.66991565009376,
                -10.0791303759911,
            ),
            (
                50,
                1,
                0.5,
                500,
                2.40322610295551,
                0.343272487193449,
                14.182570171465,
                42.5947990318086,
                8.34168525083406,
                -3.27399171150132,
            ),
            (
                50,
                1,
                0.5,
                800,
                6.52589219503979,
                0.243370384679762,
                4.7572297397883,
                -42.1269990492652,
                -7.52176458860419,
                -4.45812096051993,
            ),
        )
    )
    # "I_square [mA]","yy [mm]","zz [mm]","xx [um]","Tvx_m.Torque [nNewtonMeter]","Tvy_m.Torque [nNewtonMeter]","Tvz_m.Torque [nNewtonMeter]","Tx_square.Torque [nNewtonMeter]","Ty_square.Torque [nNewtonMeter]","Tz_square.Torque [nNewtonMeter]"
    datT = np.array(
        (
            (
                50,
                0,
                0.2,
                200,
                100.376931034983,
                1.29132310544713,
                -44.5742433626561,
                0.0135554469175028,
                -12.259497183979,
                0.0234822598655818,
            ),
            (
                50,
                0,
                0.2,
                500,
                -35.3674218480089,
                15.2488447927969,
                16.5356714838072,
                0.0128574072989242,
                -2.11323611976223,
                0.0140856152834011,
            ),
            (
                50,
                0,
                0.2,
                800,
                78.1145191288726,
                28.9160470934922,
                42.7406950893594,
                -0.00270261511070544,
                3.82101753288969,
                0.001268285244331,
            ),
            (
                50,
                1,
                0.2,
                200,
                253.962525714332,
                -3.58528490354542,
                -18.3631031181225,
                0.906354335079891,
                -11.983108298508,
                1.88954778000799,
            ),
            (
                50,
                1,
                0.2,
                500,
                4.46861420652312,
                17.1480742529462,
                29.6735385079683,
                0.0283925165409689,
                -2.05547893566587,
                -0.837502175937236,
            ),
            (
                50,
                1,
                0.2,
                800,
                -134.966632069798,
                26.1191706299256,
                -3.58074585018869,
                -0.728913411271276,
                3.76373308619741,
                -0.751039454165596,
            ),
            (
                50,
                0,
                0.5,
                200,
                -23.9024071346421,
                -17.6356253739313,
                -183.322034423475,
                0.00115987401080014,
                -6.23683839052518,
                0.00221598446713591,
            ),
            (
                50,
                0,
                0.5,
                500,
                2.78526886521961,
                31.7179118567721,
                -24.2255502578008,
                0.00550707289213479,
                -2.67769582918341,
                -0.00468607713562535,
            ),
            (
                50,
                0,
                0.5,
                800,
                -40.0407569434816,
                3.37798020460892,
                -39.931893857586,
                -0.00396662587435437,
                0.350027477836724,
                -0.00218198771623613,
            ),
            (
                50,
                1,
                0.5,
                200,
                -2.0307023796287,
                0.583836236430798,
                75.3662429442374,
                0.542371195596806,
                -6.0801890938976,
                1.19222997481852,
            ),
            (
                50,
                1,
                0.5,
                500,
                23.698487998103,
                18.7804224066191,
                51.5633199828762,
                0.51802382216635,
                -2.60537736999987,
                0.0549590129589455,
            ),
            (
                50,
                1,
                0.5,
                800,
                145.274705178714,
                -19.4722759246531,
                6.63502773049354,
                0.0223391263649002,
                0.348069100595905,
                -0.392873453263144,
            ),
        )
    )

    for d, t in zip(datF, datT, strict=False):
        i0 = d[0] * 1e-3  # A
        pos = np.array((d[3] * 1e-3, d[1], d[2])) * 1e-3
        f2 = np.array((d[4], d[5], d[6])) * 1e-6
        # f1 = np.array((d[7], d[8], d[9])) * 1e-6 # TODO check if necessary

        # t1 = np.array((t[4], t[5], t[6])) * 1e-9 # TODO check if necessary
        t2 = np.array((t[7], t[8], t[9])) * 1e-9

        for wire in wires:
            wire.current = i0
        magnet.position = pos + np.array((0, 0, 0.15)) * 1e-3

        F1, _ = np.sum(getFT(wires, magnet, pivot=(0, 0, 0)), axis=1)
        F2, T2 = np.sum(getFT(magnet, wires, pivot=(0, 0, 0)), axis=1)
        T2 *= -1  # bad sign at initial test design

        assert np.linalg.norm(F1 + F2) / np.linalg.norm(F1) < 1e-3
        assert np.linalg.norm(f2 - F2) / np.linalg.norm(F2) < 1e-2
        assert np.linalg.norm(t2 + T2) / np.linalg.norm(T2) < 0.1


#######################################################################
#######################################################################
### MISC TESTS ########################################################
#######################################################################
#######################################################################


def test_force_obj_rotations1():
    """
    test if rotated currents give the same result
    """
    s1 = magpy.magnet.Sphere(diameter=1, polarization=(1, 2, 3), position=(0, 0, -1))

    verts1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    verts2 = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 0, 0)]

    c1 = magpy.current.Polyline(
        vertices=verts1,
        current=1,
    )
    c1.meshing = 15

    c2 = magpy.current.Polyline(
        vertices=verts2,
        current=1,
    )
    c2.meshing = 15
    c2.rotate_from_angax(-90, "x")

    F, T = magpy.getFT(s1, [c1, c2], pivot=(0, 0, 0))

    errF = max(abs(F[0] - F[1]) / np.linalg.norm(F[0] + F[1]) * 2)
    errT = max(abs(T[0] - T[1]) / np.linalg.norm(T[0] + T[1]) * 2)

    assert errF < 1e-14, f"Force mismatch: {errF}"
    assert errT < 1e-14, f"Torque mismatch: {errT}"


def test_force_obj_rotations2():
    """
    test if dipole with orientation gives same result as
    rotated magnetic moment
    """
    mm, md = np.array((0.976, 4.304, 2.055)), np.array((0.878, -1.527, 2.918))
    pm, pd = np.array((-1.248, 7.835, 9.273)), np.array((-2.331, 5.835, 0.578))

    magnet = magpy.magnet.Cuboid(position=pm, dimension=(1, 2, 3), polarization=mm)

    r = R.from_euler("xyz", (25, 65, 150), degrees=True)
    dipole1 = magpy.misc.Dipole(position=pd, moment=md, orientation=r)
    dipole2 = magpy.misc.Dipole(position=pd, moment=r.apply(md))

    F, T = magpy.getFT(magnet, [dipole1, dipole2], pivot=(0, 0, 0))

    errF = max(abs(F[0] - F[1]) / np.linalg.norm(F[0] + F[1]) * 2)
    errT = max(abs(T[0] - T[1]) / np.linalg.norm(T[0] + T[1]) * 2)

    assert errF < 1e-14, f"Force mismatch: {errF}"
    assert errT < 1e-14, f"Torque mismatch: {errT}"


def test_force_2sources():
    """
    test force with two sources
    """
    src1 = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(1, 1, 1),
    )
    src2 = src1.copy(polarization=(0, 0, 2))

    tgt1 = magpy.magnet.Cuboid(
        dimension=(0.3, 0.3, 0.3),
        polarization=(0, 0, 1),
        position=(0, 1, 3),
        meshing=27,
    )
    tgt2 = tgt1.copy(
        polarization=(0, 0, 2),
    )
    F, T = getFT([src1, src2], [tgt1, tgt2], eps=1e-6, pivot=(3, 3, 3))

    assert np.allclose(F[1, 1], F[0, 0] * 4)
    assert np.allclose(F[1, 1], F[0, 1] * 2)
    assert np.allclose(F[1, 1], F[1, 0] * 2)

    assert np.allclose(T[1, 1], T[0, 0] * 4)
    assert np.allclose(T[1, 1], T[0, 1] * 2)
    assert np.allclose(T[1, 1], T[1, 0] * 2)


def test_force_meshing_validation():
    """Test meshing inputs"""

    # standard objects
    objects = [
        magpy.magnet.Cylinder(dimension=(3, 3), polarization=(0, 0, 1)),
        magpy.magnet.CylinderSegment(
            dimension=(1, 2, 3, 0, 360), polarization=(1, 2, 3)
        ),
        magpy.magnet.Tetrahedron(
            vertices=[(1, 1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1)],
            polarization=(0.111, 0.222, 0.333),
        ),
        magpy.current.Polyline(vertices=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], current=1),
        magpy.magnet.Cuboid(dimension=(3, 3, 3), polarization=(0, 0, 1)),
    ]
    for obj in objects:
        with pytest.raises(ValueError):  # noqa: PT011
            obj.meshing = "bad"

        with pytest.raises(ValueError):  # noqa: PT011
            obj.meshing = -1

        obj.meshing = 1
        assert obj.meshing == 1, f"Failed for {obj} with meshing=1"

        obj.meshing = 100
        assert obj.meshing == 100, f"Failed for {obj} with meshing=100"

    # circle
    obj = magpy.current.Circle(diameter=3, current=1)
    with pytest.raises(ValueError):  # noqa: PT011
        obj.meshing = "bad"

    with pytest.raises(ValueError):  # noqa: PT011
        obj.meshing = 3

    obj.meshing = 4
    assert obj.meshing == 4, f"Failed for {obj} with meshing=4"

    obj.meshing = 100
    assert obj.meshing == 100, f"Failed for {obj} with meshing=100"

    cube = magpy.magnet.Cuboid(dimension=(3, 3, 3), polarization=(0, 0, 1))
    cube.meshing = (10, 10, 10)
    assert np.all(cube.meshing == (10, 10, 10))

    # polyline
    poly = magpy.current.Polyline(vertices=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], current=1)
    poly.meshing = 1

    with pytest.warns(UserWarning):  # noqa: PT030
        getFT(cube, poly)


def test_centroid():
    """Test centroid calculation"""
    circ = magpy.current.Circle(diameter=1, current=1)
    np.testing.assert_allclose(circ.centroid, (0, 0, 0))
    np.testing.assert_allclose(circ._centroid, [(0, 0, 0)])

    poly = magpy.current.Polyline(
        vertices=[(0, 0, 0), (1, 1, 1)], current=1, position=(2, 2, 2)
    )
    np.testing.assert_allclose(poly.centroid, (2.5, 2.5, 2.5))
    np.testing.assert_allclose(poly._centroid, [(2.5, 2.5, 2.5)])

    poly = magpy.current.Polyline(
        vertices=[(0, 0, 0), (1, 1, 1)], current=1, position=[(2, 2, 2), (3, 3, 3)]
    )
    np.testing.assert_allclose(poly.centroid, [(2.5, 2.5, 2.5), (3.5, 3.5, 3.5)])
    np.testing.assert_allclose(poly._centroid, [(2.5, 2.5, 2.5), (3.5, 3.5, 3.5)])

    cube = magpy.magnet.Cuboid(
        dimension=(1, 1, 1), polarization=(1, 0, 0), position=(1, 2, 3)
    )
    np.testing.assert_allclose(cube.centroid, (1, 2, 3))
    np.testing.assert_allclose(cube._centroid, [(1, 2, 3)])

    cyl = magpy.magnet.Cylinder(
        dimension=(1, 1), polarization=(1, 0, 0), position=(1, 2, 3)
    )
    np.testing.assert_allclose(cyl.centroid, (1, 2, 3))
    np.testing.assert_allclose(cyl._centroid, [(1, 2, 3)])

    seg = magpy.magnet.CylinderSegment(
        dimension=(1, 2, 3, 0, 90),
        polarization=(1, 0, 0),
        position=[(1, 2, 3), (4, 5, 6)],
    )
    np.testing.assert_allclose(seg.centroid, seg._barycenter)
    np.testing.assert_allclose(seg._centroid, seg._barycenter)

    sph = magpy.magnet.Sphere(diameter=1, polarization=(1, 0, 0), position=(3, 2, 3))
    np.testing.assert_allclose(sph.centroid, (3, 2, 3))
    np.testing.assert_allclose(sph._centroid, [(3, 2, 3)])


########################################################################################
########################################################################################
### EQUIVALENT TESTS ###################################################################
########################################################################################
########################################################################################


def test_force_equivalent_dipole_Circle():
    """dipole moment equivalence"""
    src = magpy.magnet.Cuboid(
        dimension=(1, 2, 3),
        polarization=(3, 2, -1),
        position=(-0.6, -1.1, -1.6),
    )
    r0 = 0.0123
    i0 = 12345
    loop = magpy.current.Circle(diameter=2 * r0, current=i0, meshing=4)

    # moment = surface * current
    mom = r0**2 * np.pi * i0 * np.array((0, 0, 1))
    dip = magpy.misc.Dipole(moment=mom)

    F, T = getFT(src, [loop, dip], pivot=(0.21, 0.12, 0.33))
    errF = np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[0] + F[1])
    errT = np.linalg.norm(T[0] - T[1]) / np.linalg.norm(T[0] + T[1])
    assert errF < 1e-3
    assert errT < 1e-3


def test_force_equivalent_dipole_Polyline():
    """dipole moment equivalence"""
    src = magpy.magnet.Cuboid(
        dimension=(1, 2, 3),
        polarization=(3, 2, -1),
        position=(-0.6, -1.1, -1.6),
    )
    verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0)]
    verts = np.array(verts) * 1e-3
    i0 = 12345
    loop = magpy.current.Polyline(vertices=verts, meshing=4, current=i0)

    # moment = surface * current
    mom = 4 * 1e-6 * i0 * np.array((0, 0, 1))
    dip = magpy.misc.Dipole(moment=mom)

    F, T = getFT(src, [loop, dip], pivot=(0.21, 0.12, 0.33))
    errF = np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[0] + F[1])
    errT = np.linalg.norm(T[0] - T[1]) / np.linalg.norm(T[0] + T[1])
    assert errF < 1e-5
    assert errT < 1e-5


def test_force_equivalent_dipole_Magnets():
    """dipole moment equivalence"""
    src = magpy.magnet.Cuboid(
        dimension=(1, 2, 3),
        polarization=(3, 2, -1),
        position=(-0.6, -1.1, -1.6),
    )
    cube = magpy.magnet.Cuboid(
        dimension=np.array((2.23, 1.2, 3)) * 1e-3,
        polarization=(2, 2.34, 1),
        meshing=(2, 2, 2),
    )
    cyl = magpy.magnet.Cylinder(
        dimension=(1.23e-3, 1.51e-3),
        polarization=(1, 2.2, 3.1),
        meshing=5,
    )
    cyls = magpy.magnet.CylinderSegment(
        dimension=(0.23e-3, 0.51e-3, 0.2e-3, -33, 128.1),
        polarization=(1, 2.2, 3.1),
        meshing=10,
    )
    tetra = magpy.magnet.Tetrahedron(
        vertices=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]) * 1e-3,
        polarization=(1, 2.2, 3.1),
        meshing=10,
    )
    trimesh = magpy.magnet.TriangularMesh(
        vertices=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]) * 1e-4,
        faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        polarization=(1, 2.2, 3.1),
        meshing=10,
    )
    sph = magpy.magnet.Sphere(
        diameter=1.1e-3,
        polarization=(1, 2.2, 3.1),
    )
    magnets = [cube, cyl, cyls, tetra, trimesh, sph]
    for magnet in magnets:
        # moment = volume * magnetization
        mom = magnet.volume * magnet.magnetization
        dip = magpy.misc.Dipole(moment=mom, position=magnet.centroid)

        F, T = getFT(src, [magnet, dip], pivot=(0.21, 0.12, 0.33))
        errF = np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[0] + F[1])
        errT = np.linalg.norm(T[0] - T[1]) / np.linalg.norm(T[0] + T[1])
        assert errF < 1e-4
        assert errT < 1e-4


def test_force_equivalent_trisheet_polyline():
    """compare narrow tri sheet with straight polyline"""
    cube = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        position=(4, 5, 4),
        polarization=(1, 2, 3),
    )

    i0 = 100
    cd = np.array((25, 0, 5))
    cd = cd / np.linalg.norm(cd) * i0 / 0.02
    angles = [45.123, -90, 180, 1, 2, -3, 225]

    cs = magpy.current.TriangleSheet(
        vertices=((-10, -0.01, -3), (15, -0.01, 2), (-10, 0.01, -3), (15, 0.01, 2)),
        faces=((0, 1, 2), (1, 2, 3)),
        current_densities=[cd, cd],
        meshing=1000,
    ).rotate_from_angax(angles, axis=(1, 2, -1))

    line = magpy.current.Polyline(
        current=i0,
        vertices=((-10, 0, -3), (15, 0, 2)),
        meshing=100,
    ).rotate_from_angax(angles, axis=(1, 2, -1))

    f, t = magpy.getFT(cube, [cs, line])

    errf = np.linalg.norm(f[:, 0] - f[:, 1], axis=1) / np.linalg.norm(
        f[:, 0] + f[:, 1], axis=1
    )
    assert max(errf) < 1e-4

    errt = np.linalg.norm(t[:, 0] - t[:, 1], axis=1) / np.linalg.norm(
        t[:, 0] + t[:, 1], axis=1
    )
    assert max(errt) < 1e-3


def test_force_equivalent_tristrip_polyline():
    """compare narrow tri strip with straight polyline"""
    src = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        position=(4, 5, 4),
        polarization=(1, 2, 3),
    )

    i0 = 123.123

    strip = magpy.current.TriangleStrip(
        vertices=((-10, -0.01, -3), (-10, 0.01, -3), (15, -0.01, 2), (15, 0.01, 2)),
        current=i0,
        meshing=1000,
    ).rotate_from_angax([1, 2, 3, 4], axis=(1, 2, -1))

    line = magpy.current.Polyline(
        current=i0,
        vertices=((-10, 0, -3), (15, 0, 2)),
        meshing=100,
    ).rotate_from_angax([1, 2, 3, 4], axis=(1, 2, -1))

    f, t = magpy.getFT(src, [strip, line])

    errf = np.linalg.norm(f[:, 0] - f[:, 1], axis=1) / np.linalg.norm(
        f[:, 0] + f[:, 1], axis=1
    )
    assert max(errf) < 1e-4

    errt = np.linalg.norm(t[:, 0] - t[:, 1], axis=1) / np.linalg.norm(
        t[:, 0] + t[:, 1], axis=1
    )
    assert max(errt) < 1e-3


def test_force_equivalent_Cuboid_Strip_Sheet():
    """
    tests cube VS strip VS sheet
    test strip/sheet meshing convergence
    """
    a = 1.123
    b = a * 3
    h = a * 2
    pz = 1.5543e4
    mom = pz * a * b * h
    i0 = mom / (a * b) / magpy.mu_0
    cd = i0 / h
    angles = [1, 45, 180, 181.123, -1234]
    axis = [1.22, -1.1, 1.23]
    anch = [0.4, 0.2, 0.3]

    cube = magpy.magnet.Cuboid(
        dimension=(a, b, h),
        polarization=(0, 0, pz),
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    verts = [
        (-a / 2, -b / 2, -h / 2),
        (-a / 2, -b / 2, h / 2),
        (a / 2, -b / 2, -h / 2),
        (a / 2, -b / 2, h / 2),
        (a / 2, b / 2, -h / 2),
        (a / 2, b / 2, h / 2),
        (-a / 2, b / 2, -h / 2),
        (-a / 2, b / 2, h / 2),
        (-a / 2, -b / 2, -h / 2),
        (-a / 2, -b / 2, h / 2),
    ]

    strip = magpy.current.TriangleStrip(
        vertices=verts,
        current=i0,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    sheet = magpy.current.TriangleSheet(
        vertices=verts,
        faces=[(i, i + 1, i + 2) for i in range(8)],
        current_densities=[(cd, 0, 0)] * 2
        + [(0, cd, 0)] * 2
        + [(-cd, 0, 0)] * 2
        + [(0, -cd, 0)] * 2,
    ).rotate_from_angax(angles, axis=axis, anchor=anch)

    # check if all generate the same magentic field
    B1 = cube.getB((1, 2, 3))
    B2 = strip.getB((1, 2, 3))
    B3 = sheet.getB((1, 2, 3))

    errB12 = np.linalg.norm(B1 - B2, axis=1) / np.linalg.norm(B1 + B2, axis=1)
    errB13 = np.linalg.norm(B1 - B3, axis=1) / np.linalg.norm(B1 + B3, axis=1)

    assert max(errB12) < 1e-12
    assert max(errB13) < 1e-12

    # check if all generate the same force/torque
    src = magpy.magnet.Sphere(
        position=(3, 4, -3),
        diameter=1.23,
        polarization=(3, 2, 2.123),
    )

    nn = 5
    cube.meshing = np.array((nn, 3 * nn, 2 * nn))
    strip.meshing = 20
    sheet.meshing = 20

    F1, T1 = magpy.getFT(src, cube, pivot=(0, 0, 0))
    F2, T2 = magpy.getFT(src, strip, pivot=(0, 0, 0))
    F3, T3 = magpy.getFT(src, sheet, pivot=(0, 0, 0))

    errF12 = np.linalg.norm(F1 - F2, axis=1) / np.linalg.norm(F1 + F2, axis=1)
    errF13 = np.linalg.norm(F1 - F3, axis=1) / np.linalg.norm(F1 + F3, axis=1)
    errF23 = np.linalg.norm(F2 - F3, axis=1) / np.linalg.norm(F2 + F3, axis=1)
    assert max(errF12) < 0.06
    assert max(errF13) < 0.06
    assert max(errF23) < 1e-12

    errT12 = np.linalg.norm(T1 - T2, axis=1) / np.linalg.norm(T1 + T2, axis=1)
    errT13 = np.linalg.norm(T1 - T3, axis=1) / np.linalg.norm(T1 + T3, axis=1)
    errT23 = np.linalg.norm(T2 - T3, axis=1) / np.linalg.norm(T2 + T3, axis=1)
    assert max(errT12) < 0.2
    assert max(errT13) < 0.2
    assert max(errT23) < 1e-12

    # check meshing convergence
    strip.meshing = 200
    sheet.meshing = 200
    F1, T1 = magpy.getFT(src, cube, pivot=(0, 0, 0))
    F2, T2 = magpy.getFT(src, strip, pivot=(0, 0, 0))
    F3, T3 = magpy.getFT(src, sheet, pivot=(0, 0, 0))

    errF12 = np.linalg.norm(F1 - F2, axis=1) / np.linalg.norm(F1 + F2, axis=1)
    errF13 = np.linalg.norm(F1 - F3, axis=1) / np.linalg.norm(F1 + F3, axis=1)
    errF23 = np.linalg.norm(F2 - F3, axis=1) / np.linalg.norm(F2 + F3, axis=1)
    assert max(errF12) < 0.01
    assert max(errF13) < 0.01
    assert max(errF23) < 1e-14

    errT12 = np.linalg.norm(T1 - T2, axis=1) / np.linalg.norm(T1 + T2, axis=1)
    errT13 = np.linalg.norm(T1 - T3, axis=1) / np.linalg.norm(T1 + T3, axis=1)
    errT23 = np.linalg.norm(T2 - T3, axis=1) / np.linalg.norm(T2 + T3, axis=1)
    assert max(errT12) < 0.006
    assert max(errT13) < 0.006
    assert max(errT23) < 1e-12

    strip.meshing = 2000
    sheet.meshing = 2000
    F1, T1 = magpy.getFT(src, cube, pivot=(0, 0, 0))
    F2, T2 = magpy.getFT(src, strip, pivot=(0, 0, 0))
    F3, T3 = magpy.getFT(src, sheet, pivot=(0, 0, 0))

    errF12 = np.linalg.norm(F1 - F2, axis=1) / np.linalg.norm(F1 + F2, axis=1)
    errF13 = np.linalg.norm(F1 - F3, axis=1) / np.linalg.norm(F1 + F3, axis=1)
    errF23 = np.linalg.norm(F2 - F3, axis=1) / np.linalg.norm(F2 + F3, axis=1)
    assert max(errF12) < 0.002
    assert max(errF13) < 0.002
    assert max(errF23) < 1e-14

    errT12 = np.linalg.norm(T1 - T2, axis=1) / np.linalg.norm(T1 + T2, axis=1)
    errT13 = np.linalg.norm(T1 - T3, axis=1) / np.linalg.norm(T1 + T3, axis=1)
    errT23 = np.linalg.norm(T2 - T3, axis=1) / np.linalg.norm(T2 + T3, axis=1)
    assert max(errT12) < 0.0008
    assert max(errT13) < 0.0008
    assert max(errT23) < 1e-12
