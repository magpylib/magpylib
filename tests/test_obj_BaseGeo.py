import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput

# pylint: disable=no-member


def test_BaseGeo_basics():
    """fundamental usage test"""

    ptest = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0.67545246, -0.6675014, -0.21692852],
        ]
    )

    otest = np.array(
        [
            [0, 0, 0],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0, 0, 0],
            [0.20990649, 0.41981298, 0.62971947],
            [0, 0, 0],
            [0.59199676, 0.44281248, 0.48074693],
        ]
    )

    poss, rots = [], []

    bgeo = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        orientation=None,
    )
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.position = (1, 2, 3)
    bgeo.orientation = R.from_rotvec((0.1, 0.2, 0.3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.move((-1, -2, -3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((-0.1, -0.2, -0.3))
    bgeo.rotate(rotation=rot, start=-1)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(angle=45, axis=(1, 2, 3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(-np.pi / 4, (1, 2, 3), degrees=False)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((0.1, 0.2, 0.3))
    bgeo.rotate(rot, anchor=(3, 2, 1), start=-1)
    bgeo.rotate_from_angax(33, (3, 2, 1), anchor=0, start=-1)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    poss = np.array(poss)
    rots = np.array(rots)

    # avoid generating different zeros in macos CI tests (atol=1e-6)
    np.testing.assert_allclose(
        poss,
        ptest,
        atol=1e-6,
        err_msg="test_BaseGeo bad position",
    )
    np.testing.assert_allclose(
        rots,
        otest,
        atol=1e-6,
        err_msg="test_BaseGeo bad orientation",
    )


def test_rotate_vs_rotate_from():
    """testing rotate vs rotate_from_angax"""
    roz = [
        (0.1, 0.2, 0.3),
        (-0.1, -0.1, -0.1),
        (0.2, 0, 0),
        (0.3, 0, 0),
        (0, 0, 0.4),
        (0, -0.2, 0),
    ]

    bg1 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(3, 4, 5),
        orientation=R.from_quat((0, 0, 0, 1)),
    )
    for ro in roz:
        rroz = R.from_rotvec((ro,))
        bg1.rotate(rotation=rroz, anchor=(-3, -2, 1))
    pos1 = bg1.position
    ori1 = bg1.orientation.as_quat()

    bg2 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(3, 4, 5),
        orientation=R.from_quat((0, 0, 0, 1)),
    )
    angs = np.linalg.norm(roz, axis=1)
    for ang, ax in zip(angs, roz, strict=False):
        bg2.rotate_from_angax(angle=[ang], degrees=False, axis=ax, anchor=(-3, -2, 1))
    pos2 = bg2.position
    ori2 = bg2.orientation.as_quat()

    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)


def test_BaseGeo_reset_path():
    """testing reset path"""
    # pylint: disable=protected-access
    bg = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        orientation=R.from_quat((0, 0, 0, 1)),
    )
    bg.move([(1, 1, 1)] * 11)

    assert len(bg._position) == 12, "bad path generation"

    bg.reset_path()
    assert len(bg._position) == 1, "bad path reset"


def test_BaseGeo_anchor_None():
    """testing rotation with None anchor"""
    pos = np.array([1, 2, 3])
    bg = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=pos,
        orientation=R.from_quat((0, 0, 0, 1)),
    )
    bg.rotate(R.from_rotvec([(0.1, 0.2, 0.3), (0.2, 0.4, 0.6)]))

    pos3 = np.array([pos] * 3)
    rot3 = np.array([(0, 0, 0), (0.1, 0.2, 0.3), (0.2, 0.4, 0.6)])
    np.testing.assert_allclose(
        bg.position,
        pos3,
        err_msg="None rotation changed position",
    )
    np.testing.assert_allclose(
        bg.orientation.as_rotvec(),
        rot3,
        err_msg="None rotation did not adjust rot",
    )


def evall(obj):
    """return pos and orient of object"""
    # pylint: disable=protected-access
    pp = obj._position
    rr = obj._orientation.as_quat()
    rr = np.array([r / max(r) for r in rr])
    return (pp, rr)


def test_attach():
    """test attach functionality"""
    bg = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=[0, 0, 0],
        orientation=R.from_rotvec((0, 0, 0)),
    )
    rot_obj = R.from_rotvec([(x, 0, 0) for x in np.linspace(0, 10, 11)])
    bg.rotate(rot_obj, start=-1)

    bg2 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=[0, 0, 0],
        orientation=R.from_rotvec((0, 0, 0)),
    )
    roto = R.from_rotvec(((1, 0, 0),))
    for _ in range(10):
        bg2.rotate(roto)

    np.testing.assert_allclose(
        bg.position,
        bg2.position,
        err_msg="attach p",
    )
    np.testing.assert_allclose(
        bg.orientation.as_quat(),
        bg2.orientation.as_quat(),
        err_msg="attach o",
    )


def test_path_functionality1():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    b1, b2, b3, b4, b5 = pos0
    c1, c2, c3 = inpath
    q1, q2, q3, q4, q5 = np.array(
        [(1, 0, 0, 1), (1, 0, 0, 0.5), (1, 0, 0, 0.25), (1, 0, 0, 0.2), (1, 0, 0, 0.1)]
    )

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        )
    )
    P = np.array([b1, b2, b3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=0)
    )
    P = np.array([b1 + c1, b2 + c2, b3 + c3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=1)
    )
    P = np.array([b1, b2 + c1, b3 + c2, b4 + c3, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=2)
    )
    P = np.array([b1, b2, b3 + c1, b4 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)


def test_path_functionality2():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    b1, b2, b3, b4, b5 = pos0
    c1, c2, c3 = inpath
    q1, q2, q3, q4, q5 = np.array(
        [(1, 0, 0, 1), (1, 0, 0, 0.5), (1, 0, 0, 0.25), (1, 0, 0, 0.2), (1, 0, 0, 0.1)]
    )

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=3)
    )
    P = np.array([b1, b2, b3, b4 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=4)
    )
    P = np.array([b1, b2, b3, b4, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=5)
    )
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=5)
    )
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)

    pos, ori = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath)
    )
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    np.testing.assert_allclose(pos, P)
    np.testing.assert_allclose(ori, Q)


def test_path_functionality3():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    pos1, ori1 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=4)
    )
    pos2, ori2 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=-1)
    )
    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)

    pos1, ori1 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=3)
    )
    pos2, ori2 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=-2)
    )
    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)

    pos1, ori1 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=2)
    )
    pos2, ori2 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=-3)
    )
    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)

    pos1, ori1 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=1)
    )
    pos2, ori2 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=-4)
    )
    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)

    pos1, ori1 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=0)
    )
    pos2, ori2 = evall(
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=pos0, orientation=rot0
        ).move(inpath, start=-5)
    )
    np.testing.assert_allclose(pos1, pos2)
    np.testing.assert_allclose(ori1, ori2)


def test_scipy_from_methods():
    """test all rotation methods inspired from scipy implemented in BaseTransform"""

    def cube():
        return magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 1, 1))

    angs_deg = np.linspace(0, 360, 10)
    angs = np.deg2rad(angs_deg)
    rot = R.from_rotvec((np.array([[0, 0, 1]] * 10).T * angs).T)
    anchor = (1, 2, 3)
    cube0 = cube().rotate(rot, anchor=anchor)

    from_rotvec = cube().rotate_from_rotvec(
        rot.as_rotvec(degrees=True), anchor=anchor, degrees=True
    )
    np.testing.assert_allclose(
        cube0.position,
        from_rotvec.position,
        rtol=1e-05,
        atol=1e-08,
        err_msg="from_rotvec failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_rotvec.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )

    from_angax = cube().rotate_from_angax(angs_deg, "z", anchor=anchor, degrees=True)
    np.testing.assert_allclose(
        cube0.position,
        from_angax.position,
        err_msg="from_angax failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_angax.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )

    from_euler = cube().rotate_from_euler(
        angs_deg[:, np.newaxis], "z", anchor=anchor, degrees=True
    )
    np.testing.assert_allclose(
        cube0.position,
        from_euler.position,
        rtol=1e-05,
        atol=1e-08,
        err_msg="from_euler failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_euler.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )

    from_matrix = cube().rotate_from_matrix(rot.as_matrix(), anchor=anchor)
    np.testing.assert_allclose(
        cube0.position,
        from_matrix.position,
        rtol=1e-05,
        atol=1e-08,
        err_msg="from_matrix failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_matrix.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )

    from_mrp = cube().rotate_from_mrp(rot.as_mrp(), anchor=anchor)
    np.testing.assert_allclose(
        cube0.position,
        from_mrp.position,
        rtol=1e-05,
        atol=1e-08,
        err_msg="from_mrp failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_mrp.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )

    from_quat = cube().rotate_from_quat(rot.as_quat(), anchor=anchor)
    np.testing.assert_allclose(
        cube0.position,
        from_quat.position,
        rtol=1e-05,
        atol=1e-08,
        err_msg="from_quat failed on position",
    )
    np.testing.assert_allclose(
        cube0.orientation.as_rotvec(),
        from_quat.orientation.as_rotvec(),
        err_msg="from_rotvec failed on orientation",
    )


def test_style():
    """test when setting wrong style class"""
    bg = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        orientation=None,
    )
    bg.style = {"color": "red"}
    bg.style = {"label": "mylabel"}
    assert bg.style.color == "red"
    assert bg.style.label == "mylabel"
    with pytest.raises(TypeError, match=r"Input .* must be"):
        bg.style = "wrong class"


def test_kwargs():
    """test kwargs inputs, only relevant for styles"""
    bg = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        orientation=None,
        style={"label": "label_01"},
        style_label="label_02",
    )
    assert bg.style.label == "label_02"

    with pytest.raises(TypeError):
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1),
            polarization=(0, 0, 1),
            position=(0, 0, 0),
            orientation=None,
            styl_label="label_02",
        )


def test_copy():
    """test copying object"""
    bg1 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(0, 0, 0),
        orientation=None,
        style_label="label1",
    )  # has style
    bg2 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(1, 2, 3),
        orientation=None,
    )  # has no style
    bg3 = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=(4, 6, 8),
        style_color="blue",
    )  # has style but label is None
    bg1c = bg1.copy()
    bg2c = bg2.copy(position=(10, 0, 0), style={"color": "red"}, style_color="orange")
    bg3c = bg3.copy()

    # original object should not be affected"
    np.testing.assert_allclose(bg1.position, (0, 0, 0))
    np.testing.assert_allclose(bg2.position, (1, 2, 3))

    # check if label suffix iterated correctly
    assert bg1c.style.label == "label2"
    assert bg2c.style.label is None
    assert bg3c.style.label == "Cuboid_01"

    # check if style is passed correctly
    assert bg2c.style.color == "orange"


def test_copy_parents():
    """make sure that parents are not copied"""
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    x3 = magpy.Sensor()

    c = x1 + x2 + x3

    y = x1.copy()

    assert x1.parent.parent == c
    assert y.parent is None


def test_copy_order():
    """Make sure copying objects of a collection does not affect order of children (#530)"""

    thing1 = magpy.magnet.Cuboid(style_label="t1")
    thing2 = magpy.magnet.Cuboid(style_label="t2")
    thing3 = magpy.magnet.Cuboid(style_label="t3")
    coll = magpy.Collection(thing1, thing2, thing3)

    desc_before = coll.describe(format="label", return_string=True)

    thing1.copy()

    desc_after = coll.describe(format="label", return_string=True)

    assert desc_after == desc_before


def match_string_up_to_id(s1, s2, /):
    """Checks if 2 inputs (first is list, second string) match as long as id= follows a
    series of numbers"""
    patt = "id=[0-9]*[0-9]", "id=Regex"
    assert re.sub(*patt, "\n".join(s1)).split("\n") == re.sub(*patt, s2).split("\n")


def test_describe_with_label():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    # pylint: disable=protected-access
    x = magpy.magnet.Cuboid(style_label="x1")

    # describe calls print by default -> no return value
    desc = x.describe()
    assert desc is None

    # describe string
    test = [
        "Cuboid(id=2743358534352, label='x1')",
        "  • path length: 1",
        "  • parent: None",
        "  • path properties: ",
        "    • polarization: None T",
        "    • magnetization: None A/m",
        "    • position: [0. 0. 0.] m",
        "    • orientation: [0. 0. 0.] deg",
        "    • dimension: None m",
        "  • volume: 0 m³",
        "  • centroid: [0. 0. 0.]",
        "  • dipole_moment: [0. 0. 0.]",
        "  • meshing: None",
    ]
    match_string_up_to_id(test, x.describe(return_string=True))

    # describe html string
    test_html = ("<pre>" + "\n".join(test) + "</pre>").split("\n")
    match_string_up_to_id(test_html, x._repr_html_().replace("<br>", "\n"))


def test_describe_with_parent():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    x = magpy.magnet.Cuboid(style_label="x1")
    magpy.Collection(x)  # add parent
    test = [
        "Cuboid(id=1687262797456, label='x1')",
        "  • path length: 1",
        "  • parent: Collection(id=1687262859280)",
        "  • path properties: ",
        "    • polarization: None T",
        "    • magnetization: None A/m",
        "    • position: [0. 0. 0.] m",
        "    • orientation: [0. 0. 0.] deg",
        "    • dimension: None m",
        "  • volume: 0 m³",
        "  • centroid: [0. 0. 0.]",
        "  • dipole_moment: [0. 0. 0.]",
        "  • meshing: None",
    ]
    match_string_up_to_id(test, x.describe(return_string=True))


def test_describe_with_path():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    x = magpy.Sensor(position=[(1, 2, 3)] * 3)
    test = [
        "Sensor(id=2743359152656)",
        "  • path length: 3",
        "  • parent: None",
        "  • path properties: ",
        "    • position: [1. 2. 3.] m",
        "    • orientation: [0. 0. 0.] deg",
        "  • centroid: [[1. 2. 3.]  [1. 2. 3.]  [1. 2. 3.]]",
        "  • handedness: right",
        "  • pixel: None",
    ]
    match_string_up_to_id(test, x.describe(return_string=True))


def test_describe_with_exclude_None():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    x = magpy.Sensor()
    test = [
        "Sensor(id=140534160166976)",
        "  • path length: 1",
        "  • parent: None",
        "  • path properties: ",
        "    • position: [0. 0. 0.] m",
        "    • orientation: [0. 0. 0.] deg",
        "  • centroid: [0. 0. 0.]",
        "  • handedness: right",
        "  • pixel: None",
        (
            "  • style: SensorStyle(size=None, sizemode=None,"
            " pixel=Pixel(size=1, sizemode=None, color=None, symbol=None,"
            " field=PixelField(source=None, colormap=None, shownull=None, symbol=None,"
            " sizescaling=None, sizemin=None, colorscaling=None)),"
            " arrows=ArrowCS(x=ArrowSingle(show=True, color=None),"
            " y=ArrowSingle(show=True, color=None), z=ArrowSingle(show=True, color=None)),"
            " label=None, description=Description(text=None, show=None),"
            " legend=Legend(text=None, show=None), color=None, opacity=None,"
            " path=Path(show=None, marker=Marker(size=None, color=None, symbol=None),"
            " line=Line(style=None, color=None, width=None), frames=None, numbering=None),"
            " model3d=Model3d(showdefault=True, data=()))"
        ),
    ]
    match_string_up_to_id(test, x.describe(exclude=None, return_string=True))


def test_describe_with_many_pixels():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    x = magpy.Sensor(pixel=[[[(1, 2, 3)] * 5] * 5] * 3, handedness="left")
    test = [
        "Sensor(id=1687262996944)",
        "  • path length: 1",
        "  • parent: None",
        "  • path properties: ",
        "    • position: [0. 0. 0.] m",
        "    • orientation: [0. 0. 0.] deg",
        "  • centroid: [1. 2. 3.]",
        "  • handedness: left",
        "  • pixel: 75 (3x5x5)",
    ]
    match_string_up_to_id(test, x.describe(return_string=True))


def test_describe_with_triangularmesh():
    """testing describe method"""
    # print("test = [\n    " + '",\n    '.join(f'"{s}' for s in desc.split("\n")) + '",\n]')
    points = [
        (-1, -1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (0, 0, 2),
    ]
    x = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=(0, 0, 1),
        points=points,
        check_selfintersecting="skip",
    )
    test = [
        "TriangularMesh(id=1687257413648)",
        "  • path length: 1",
        "  • parent: None",
        "  • path properties: ",
        "    • polarization: [0. 0. 1.] T",
        "    • magnetization: [     0.         0.    795774.716] A/m",
        "    • position: [0. 0. 0.] m",
        "    • orientation: [0. 0. 0.] deg",
        "    • vertices: [[-1. -1.  0.]  [-1.  1.  0.]  [ 1. -1.  0.]  [ 1.  1.  0.]  [ 0.  0.  2.]] m",
        "  • volume: 2.67 m³",
        "  • centroid: [0.    0.    0.461]",
        "  • dipole_moment: [      0.          0.    2122065.908]",
        "  • faces: [[4 0 2]  [4 1 0]  [3 4 2]  [3 1 4]  [3 0 1]  [3 2 0]]",
        "  • mesh: shape(6, 3, 3)",
        "  • meshing: None",
        "  • status_disconnected: False",
        "  • status_disconnected_data: 1 part",
        "  • status_open: False",
        "  • status_open_data: []",
        "  • status_reoriented: True",
        "  • status_selfintersecting: None",
        "  • status_selfintersecting_data: None",
    ]

    match_string_up_to_id(test, x.describe(return_string=True))


def test_unset_describe():
    """test describe completely unset objects"""
    objs = [
        magpy.magnet.Cuboid(),
        magpy.magnet.Cylinder(),
        magpy.magnet.CylinderSegment(),
        magpy.magnet.Sphere(),
        magpy.magnet.Tetrahedron(),
        # magpy.magnet.TriangularMesh(), not possible yet
        magpy.misc.Triangle(),
        magpy.misc.Dipole(),
        magpy.current.Polyline(),
        magpy.current.Circle(),
    ]

    for o in objs:
        o.describe()


def test_orientation_setter_repositions_children_when_prior_orientation_is_none():
    """When _orientation is None at the time the orientation setter is called
    (e.g. after an internal reset), children must still be repositioned.
    The guard `if old_oriQ is not None` was silently skipping the child loop."""
    child = magpy.magnet.Cuboid(
        dimension=(0.02, 0.02, 0.02),
        magnetization=(0, 0, 1e6),
        position=(0.05, 0, 0),
    )
    coll = magpy.Collection(child)
    object.__setattr__(coll, "_orientation", None)
    coll.orientation = R.from_euler("z", 90, degrees=True)
    np.testing.assert_allclose(child.position, [0, 0.05, 0], atol=1e-10)


def test_orientation_setter_none_treated_as_identity_rotation():
    """With _orientation=None treated as identity, the delta applied to children
    must equal the new orientation (new * identity.inv() = new)."""
    rot = R.from_euler("z", 45, degrees=True)
    pos_initial = np.array([0.05, 0.0, 0.0])
    pos_expected = rot.apply(pos_initial)
    child = magpy.magnet.Cuboid(
        dimension=(0.01, 0.01, 0.01),
        magnetization=(0, 0, 1e6),
        position=pos_initial,
    )
    coll = magpy.Collection(child)
    object.__setattr__(coll, "_orientation", None)
    coll.orientation = rot
    np.testing.assert_allclose(child.position, pos_expected, atol=1e-10)


def test_is_initializing_reset_after_failed_init():
    """_is_initializing must be False after __init__ raises, even via __new__.

    If a setter raises during __init__ (e.g. bad dimension), the try/finally
    around the _is_initializing block ensures the flag is cleared.  Without it,
    any subsequent position/orientation setter call on the partially-constructed
    object would skip eager geometric sync, silently leaving path lengths
    inconsistent.
    """

    obj = magpy.magnet.Cuboid.__new__(magpy.magnet.Cuboid)
    with pytest.raises(MagpylibBadUserInput):
        obj.__init__(polarization=(0, 0, 1), dimension="bad")

    # Flag must be False regardless of whether __init__ completed
    assert not getattr(obj, "_is_initializing", True), (
        "_is_initializing stuck at True after failed __init__"
    )

    # With the flag cleared, manually bring the object to a valid geometric state
    # and verify the position setter correctly syncs orientation length.
    obj._position = np.zeros((1, 3))
    obj._orientation = R.from_quat([[0, 0, 0, 1]])  # single identity
    long_path = np.array([(0, 0, i) for i in range(5)], dtype=float)
    obj.position = long_path
    assert len(obj._orientation) == 5, (
        "orientation path not synced to new position length — "
        "_is_initializing was True when it shouldn't be"
    )


def test_squeeze_path_property_helper():
    """_squeeze_path_property collapses only the path axis, never feature axes."""
    f = magpy.magnet.Cuboid._squeeze_path_property

    assert f(None) is None

    # scalar property (p,) -> scalar for len 1, full path otherwise
    np.testing.assert_array_equal(f(np.array([5.0])), 5.0)
    np.testing.assert_array_equal(f(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])

    # vector property (p, k)
    assert f(np.zeros((1, 3))).shape == (3,)
    assert f(np.zeros((4, 3))).shape == (4, 3)

    # feature axis of length 1 must be preserved (the np.squeeze trap)
    assert f(np.zeros((1, 1, 3))).shape == (1, 3)
    assert f(np.zeros((2, 1, 3))).shape == (2, 1, 3)

    # scipy Rotation: single unchanged, array len-1 -> single, len>1 -> full
    single = R.from_rotvec((0.1, 0.2, 0.3))
    assert f(single) is single
    arr1 = R.from_rotvec([(0.1, 0.2, 0.3)])
    assert f(arr1).single
    arr2 = R.from_rotvec([(0.1, 0.2, 0.3), (0.2, 0.3, 0.4)])
    assert len(f(arr2)) == 2


def test_squeeze_path_property_matches_np_squeeze_on_safe_shapes():
    """For the shapes the old getters used (feature dims >= 2), the helper must
    reproduce np.squeeze exactly — proving the consistency refactor is
    behavior-preserving."""
    f = magpy.magnet.Cuboid._squeeze_path_property
    for shape in [(1, 2), (1, 3), (1, 5), (3, 3), (1, 3, 3), (4, 3), (1, 4, 3)]:
        arr = np.arange(np.prod(shape), dtype=float).reshape(shape)
        np.testing.assert_array_equal(f(arr), np.squeeze(arr))


@pytest.mark.parametrize(
    ("obj", "attr"),
    [
        (
            magpy.magnet.Cylinder(
                magnetization=[(0, 0, 1e6), (0, 0, 2e6)],
                dimension=[(1, 1), (2, 2), (3, 3)],
            ),
            "dipole_moment",
        ),
        (
            magpy.magnet.Sphere(
                magnetization=[(0, 0, 1e6), (0, 0, 2e6)], diameter=[1, 2, 3]
            ),
            "dipole_moment",
        ),
        (
            magpy.magnet.CylinderSegment(
                magnetization=(0, 0, 1e6),
                dimension=[(1, 2, 1, 0, 90), (1, 2, 1, 0, 180), (1, 2, 1, 0, 270)],
                position=[(0, 0, 0), (0, 0, 1)],
            ),
            "dipole_moment",
        ),
        (
            magpy.magnet.CylinderSegment(
                magnetization=(0, 0, 1e6),
                dimension=[(1, 2, 1, 0, 90), (1, 2, 1, 0, 180), (1, 2, 1, 0, 270)],
                position=[(0, 0, 0), (0, 0, 1)],
            ),
            "centroid",
        ),
        (
            magpy.current.Circle(diameter=[1, 2, 3], current=[1, 2]),
            "dipole_moment",
        ),
        (
            magpy.magnet.Tetrahedron(
                magnetization=[(0, 0, 1e6), (0, 0, 2e6)],
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                position=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            ),
            "centroid",
        ),
    ],
)
def test_computed_getters_mismatched_path_lengths(obj, attr):
    """Computed getters must edge-pad path properties to a common length.

    Regression test: under lazy storage a property (e.g. magnetization) may be
    stored with a different path length than the geometry it multiplies. These
    getters combined the raw arrays and raised a numpy broadcasting error while
    getB on the same object worked via JIT padding.
    """
    result = np.asarray(getattr(obj, attr))  # must not raise
    assert result.ndim == 2  # (p, 3), padded to the common max path length


def test_path_properties_aggregate_over_all_bases():
    """`_path_properties` must collect from every base, not just the nearest.

    Concrete classes mix several bases (e.g. Cuboid(BaseMagnet, BaseTarget,
    BaseVolume, BaseDipoleMoment)). An earlier version stopped at the first
    base declaring `_path_properties`, so a property declared on any later
    base was silently dropped -- no error, just missing path padding.
    """

    class PathMixin:
        _path_properties = ("mixin_prop",)

    class Probe(magpy.magnet.Cuboid, PathMixin):
        _path_properties = ("own_prop",)

    assert "mixin_prop" in Probe._path_properties
    assert "own_prop" in Probe._path_properties
    # inherited ones survive, in MRO order, without duplicates
    assert Probe._path_properties[:5] == (
        "position",
        "orientation",
        "polarization",
        "magnetization",
        "dimension",
    )
    assert len(set(Probe._path_properties)) == len(Probe._path_properties)


def test_path_properties_unchanged_for_builtin_classes():
    """Pin the aggregated tuples so the MRO walk cannot reorder them."""
    assert magpy.magnet.Cuboid._path_properties == (
        "position",
        "orientation",
        "polarization",
        "magnetization",
        "dimension",
    )
    assert magpy.current.Polyline._path_properties == (
        "position",
        "orientation",
        "current",
        "vertices",
    )
    assert magpy.misc.Dipole._path_properties == ("position", "orientation", "moment")
