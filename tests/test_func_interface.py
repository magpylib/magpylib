import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.fields.field_BH_circle import _BHJM_circle
from magpylib._src.fields.field_BHfunc import _getBH_func
from magpylib.func import (
    circle_field,
    cuboid_field,
    cylinder_field,
    cylinder_segment_field,
    dipole_field,
    polyline_field,
    sphere_field,
    tetrahedron_field,
    triangle_charge_field,
    triangle_current_field,
)


def test_getBHfunc_bad_inputs1():
    """test bad orientation input"""
    dic = {
        "position": (0, 0, 0),
        "orientation": 1,
        "observers": (0, 0, 0),
        "diameter": 3.123,
        "current": 1.123,
    }
    with pytest.raises(TypeError):
        _getBH_func(_BHJM_circle, "B", dic, True, {})


def test_getBHfunc_bad_inputs2():
    """test bad orientation input"""
    dic = {
        "position": (0, 0, 0),
        "orientation": None,
        "observers": (0, 0, 0),
        "diameter": "woot",
        "current": 1.123,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        _getBH_func(_BHJM_circle, "B", dic, True, {})


def test_getBHfunc_bad_inputs3():
    """test bad orientation input"""
    dic = {
        "position": (0, 0, 0),
        "orientation": None,
        "observers": (0, 0, 0),
        "diameter": [(1, 2, 3), (1, 2, 3)],
        "current": 1.123,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        _getBH_func(_BHJM_circle, "B", dic, True, {})


def test_getBHfunc_bad_inputs4():
    """test bad orientation input"""
    dic = {
        "position": (0, 0, 0),
        "orientation": None,
        "observers": (0, 0, 0, 0),
        "diameter": (1, 2, 3),
        "current": 1.123,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        _getBH_func(_BHJM_circle, "B", dic, True, {})


def test_getBHfunc_bad_inputs5():
    """test bad orientation input"""
    dic = {
        "position": (0, 0, 0),
        "orientation": None,
        "observers": (0, 0),
        "diameter": (1, 2, 3),
        "current": 1.123,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        _getBH_func(_BHJM_circle, "B", dic, True, {})


def test_func_circle():
    """test if Circle implementation gives correct output"""
    B = circle_field("B", (0, 0, 0), 2, 1)
    Btest = np.array([0, 0, 0.6283185307179586 * 1e-6])
    np.testing.assert_allclose(B, Btest)

    H = circle_field("H", (0, 0, 0), 2, 1)
    Htest = np.array([0, 0, 0.6283185307179586 * 10 / 4 / np.pi])
    np.testing.assert_allclose(H, Htest)


def test_func_squeeze():
    """test if squeeze works"""
    B1 = circle_field("B", (0, 0, 0), 2, 1)
    B2 = circle_field("B", [(0, 0, 0)], 2, 1)
    B3 = circle_field("B", [(0, 0, 0)], 2, 1, squeeze=False)
    B4 = circle_field("B", [(0, 0, 0)] * 2, 2, 1)

    assert B1.ndim == 1
    assert B2.ndim == 1
    assert B3.ndim == 2
    assert B4.ndim == 2


def test_func_polyline1():
    """test getBHv with Polyline"""
    H = polyline_field(
        "H",
        observers=[(1, 1, 1), (1, 2, 3), (2, 2, 2)],
        currents=1,
        segments_start=(0, 0, 0),
        segments_end=[(0, 0, 0), (2, 2, 2), (2, 2, 2)],
    )
    x = (
        np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
        * 10
        / 4
        / np.pi
    )
    np.testing.assert_allclose(x, H, rtol=1e-05, atol=1e-08)


def test_func_polyline2():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6

    # z-line on x=1
    B1 = polyline_field(
        "B",
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
    )
    expected = np.array([0, -x, 0])
    np.testing.assert_allclose(B1, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline3():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # move z-line to x=-1
    B2 = polyline_field(
        "B",
        positions=(-2, 0, 0),
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
    )
    np.testing.assert_allclose(B2, np.array([0, x, 0]), rtol=1e-05, atol=1e-08)


def test_func_polyline4():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("z", 90, degrees=True)
    B3 = polyline_field(
        "B",
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot,
    )
    expected = np.array([x, 0, 0])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline5():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("x", 90, degrees=True)
    B3 = polyline_field(
        "B",
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot,
    )
    expected = np.array([0, 0, -x])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline6():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("y", 90, degrees=True)
    B3 = polyline_field(
        "B",
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot,
    )
    expected = np.array([0, -x, 0])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)


def test_func_cuboid():
    """test field wrapper functions"""
    n = 25
    pos_obs = np.array([1, 2, 2])
    mag = [
        [111, 222, 333],
    ] * n
    dim = [3, 3, 3]
    pos = np.array([0, 0, 0])
    rot = R.from_quat([(t, 0.2, 0.3, 0.4) for t in np.linspace(0, 0.1, n)])

    B1 = cuboid_field("B", pos_obs, dim, mag, pos, rot)

    B2 = []
    for i in range(n):
        pm = magpy.magnet.Cuboid(
            polarization=mag[i], dimension=dim, position=pos, orientation=rot[i]
        )
        B2 += [pm.getB(pos_obs)]
    B2 = np.array(B2)
    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_cylinder1():
    """cylinder test"""
    pos_obs = (11, 2, 2)
    mag = [111, 222, 333]
    dim = [3, 3]

    pm = magpy.magnet.Cylinder(polarization=mag, dimension=dim)
    pm.move(np.linspace((0.5, 0, 0), (7.5, 0, 0), 15), start=-1)
    pm.rotate_from_angax(np.linspace(0, 666, 25), "y", anchor=0)
    pm.move([(0, x, 0) for x in np.linspace(0, 5, 5)], start=-1)
    B2 = pm.getB(pos_obs)

    pos = pm.position
    rot = pm.orientation

    B1 = cylinder_field("B", pos_obs, dim, mag, pos, rot)

    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_cylinder2():
    """test cylinder func"""
    pos_obs = (11, 2, 2)
    mag = [111, 222, 333]
    dim = [3, 3]
    pos = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (5, 5, 5)]

    B1 = cylinder_field("B", pos_obs, dim, mag, pos)

    pm = magpy.magnet.Cylinder(polarization=mag, dimension=dim, position=pos)
    B2 = magpy.getB([pm], pos_obs)

    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_cylinder3():
    """test func cylinder"""
    pos_obs = (11, 2, 2)
    mag = [111, 222, 333]
    dim = [3, 3]

    H1 = cylinder_field("H", pos_obs, dim, mag)

    pm = magpy.magnet.Cylinder(polarization=mag, dimension=dim)
    H2 = pm.getH(pos_obs)

    np.testing.assert_allclose(H1, H2, rtol=1e-12, atol=1e-12)


def test_func_cylinder_segment_FEM():
    """test against FEM"""
    ts = np.linspace(0, 2, 31)
    obsp = [(t, t, t) for t in ts]

    Bfem = np.array(
        [
            (-0.0300346254954609, -0.00567085248536589, -0.0980899423563197),
            (-0.0283398999697276, -0.00136726650574628, -0.10058277210005),
            (-0.0279636086648847, 0.00191033319772333, -0.102068667474779),
            (-0.0287959403346942, 0.00385627171155148, -0.102086609934239),
            (-0.0298064414078247, 0.00502298395545467, -0.101395051504575),
            (-0.0309138327020785, 0.00585315159763698, -0.0994210978208941),
            (-0.0304478836262897, 0.00637062970240076, -0.0956959733446996),
            (-0.0294756102340511, 0.00796586777139283, -0.0909716586168481),
            (-0.0257014555198541, 0.00901347002514088, -0.0839378050637996),
            (-0.0203392379411272, 0.0113401710780434, -0.0758447872526493),
            (-0.0141186721748514, 0.014275060463367, -0.0666447793887049),
            (-0.00715638330645336, 0.0169990957749629, -0.0567988806666027),
            (-0.000315107745706201, 0.0196025044167515, -0.0471345331233655),
            (0.00570680487262037, 0.0216935664564627, -0.0379802748006986),
            (0.0106937560983821, 0.0229598553802506, -0.029816827145783),
            (0.0147153251512036, 0.0237740278061223, -0.0226247514391129),
            (0.0173457909761498, 0.0240321714861875, -0.0167312828159773),
            (0.0193755103218335, 0.023674091804632, -0.0119446813034152),
            (0.0204291390948416, 0.0230735973599725, -0.00805340729977855),
            (0.0207908036651642, 0.0221875600164857, -0.00496582571560478),
            (0.020692112773328, 0.0211419193131436, -0.00269563642259977),
            (0.0202607525969918, 0.0199897027578393, -0.000891130303443818),
            (0.0195698099586468, 0.0187793271229261, 0.000332964123866357),
            (0.0187342589014612, 0.0175395229794614, 0.00128198337775133),
            (0.0178090320514157, 0.0163998590430951, 0.00196979345612218),
            (0.0168069297247124, 0.0152418998801328, 0.00243910426847474),
            (0.0158127817011691, 0.0141524929704775, 0.00274664013462767),
            (0.0148149313600427, 0.013148844940711, 0.00293212192295656),
            (0.013878964772737, 0.0121841676914905, 0.00302995618189322),
            (0.0129803941608119, 0.0113011353152514, 0.00305232762136824),
            (0.0121250819870128, 0.0104894041620816, 0.00303690098080925),
        ]
    )

    # compare against FEM
    dim = ((1, 2, 1, 90, 360),)
    pol = (np.array((1, 2, 3)) * 1000 / np.sqrt(14),)
    pos = ((0, 0, 0.5),)
    B = cylinder_segment_field("B", obsp, dim, pol, pos)
    err = np.linalg.norm(B - Bfem * 1000, axis=1) / np.linalg.norm(B, axis=1)
    assert np.amax(err) < 0.01


def test_func_cylinder_variations():
    """compare multiple solid-cylinder solutions against each other"""
    # combine multiple slices to one big Cylinder
    B1 = cylinder_segment_field(
        field="B",
        observers=(1, 2, 3),
        dimensions=[(0, 1, 2, 20, 120), (0, 1, 2, 120, 220), (0, 1, 2, 220, 380)],
        polarizations=(22, 33, 44),
    )
    B1 = np.sum(B1, axis=0)

    # one big cylinder
    B2 = cylinder_segment_field(
        field="B",
        observers=(1, 2, 3),
        dimensions=(0, 1, 2, 0, 360),
        polarizations=(22, 33, 44),
    )

    # compute with solid cylinder code
    B3 = cylinder_field(
        field="B",
        observers=(1, 2, 3),
        dimensions=(2, 2),
        polarizations=(22, 33, 44),
    )

    np.testing.assert_allclose(B1, B2)
    np.testing.assert_allclose(B1, B3)


def test_func_cylinder4():
    """test field wrapper functions"""
    pos_obs = (11, 2, 2)
    mag = [111, 222, 333]
    dim = [3, 3]

    pm = magpy.magnet.Cylinder(polarization=mag, dimension=dim)
    pm.move(np.linspace((0.5, 0, 0), (7.5, 0, 0), 15))
    pm.rotate_from_angax(np.linspace(0, 666, 25), "y", anchor=0)
    pm.move([(0, x, 0) for x in np.linspace(0, 5, 5)])
    B2 = pm.getB(pos_obs)

    pos = pm.position
    rot = pm.orientation

    B1 = cylinder_field(
        field="B",
        observers=pos_obs,
        dimensions=dim,
        polarizations=mag,
        positions=pos,
        orientations=rot,
    )

    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_sphere1():
    """func sphere test"""
    pos_obs = (1, 2, 2)
    mag = [[111, 222, 333], [22, 2, 2], [22, -33, -44]]
    dim = 3
    H1 = sphere_field("H", pos_obs, dim, mag)

    H2 = []
    for i in range(3):
        pm = magpy.magnet.Sphere(polarization=mag[i], diameter=dim)
        H2 += [magpy.getH([pm], pos_obs)]
    H2 = np.array(H2)

    np.testing.assert_allclose(H1, H2, rtol=1e-12, atol=1e-12)


def test_func_sphere2():
    """func sphere test"""
    n = 25
    pos_obs = np.array([1, 2, 2])
    mag = [
        [111, 222, 333],
    ] * n
    dim = 3
    pos = np.array([0, 0, 0])
    rot = R.from_quat([(t, 0.2, 0.3, 0.4) for t in np.linspace(0, 0.1, n)])

    B1 = sphere_field("B", pos_obs, dim, mag, pos, rot)

    B2 = []
    for i in range(n):
        pm = magpy.magnet.Sphere(
            polarization=mag[i], diameter=dim, position=pos, orientation=rot[i]
        )
        B2 += [pm.getB(pos_obs)]
    B2 = np.array(B2)
    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_dipole():
    """test if Dipole implementation gives correct output"""
    B = dipole_field("B", (1, 1, 1), (1, 2, 3))
    Btest = np.array([9.62250449e-08, 7.69800359e-08, 5.77350269e-08])
    np.testing.assert_allclose(B, Btest)

    H = dipole_field("H", (1, 1, 1), (1, 2, 3))
    Htest = np.array([0.07657346, 0.06125877, 0.04594407])
    np.testing.assert_allclose(H, Htest, rtol=1e-05, atol=1e-08)


def test_func_triangle_charge():
    """test func triangle"""
    tria = magpy.misc.Triangle(
        vertices=[(0, 0, 0), (1, 0, 0.5), (0, 1, -0.5)],
        polarization=(10, 10, 10),
        position=(1, 2, 3),
    ).rotate_from_angax([45, 46, 147, 148], "x", anchor=(-1, 2, 3))
    obs = (0.3, 0.1, 0.2)
    B1 = tria.getB(obs)
    B2 = triangle_charge_field(
        "B", obs, tria.vertices, tria.polarization, tria.position, tria.orientation
    )
    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_triangle_current():
    """test func triangle current"""
    tria = magpy.current.TriangleSheet(
        vertices=[(0, 0, 0), (1, 0, 0.5), (0, 1, -0.5)],
        faces=(0, 1, 2),
        current_densities=(10, 10, 10),
        position=(1, 2, 3),
    ).rotate_from_angax([45, 46, 147, 148], "x", anchor=(-1, 2, 3))
    obs = (0.3, 0.1, 0.2)
    B1 = tria.getB(obs)

    vert = tria.vertices[tria.faces]
    B2 = triangle_current_field(
        "B", obs, vert, tria.current_densities, tria.position, tria.orientation
    )
    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_func_tetrahedron():
    """test func tetrahedron"""
    tetra = magpy.magnet.Tetrahedron(
        vertices=[(0, 0, 0), (1, 0, 0.5), (0, 1, -0.5), (0.1, 0, 1)],
        polarization=(10, 10, 10),
        position=(1, 2, 3),
    ).rotate_from_angax([45, 46, 147, 148], "x", anchor=(-1, 2, 3))
    obs = (0.3, 0.1, 0.2)
    B1 = tetra.getB(obs)
    B2 = tetrahedron_field(
        "B", obs, tetra.vertices, tetra.polarization, tetra.position, tetra.orientation
    )
    np.testing.assert_allclose(B1, B2, rtol=1e-12, atol=1e-12)
