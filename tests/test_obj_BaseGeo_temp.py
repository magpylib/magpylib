import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest
import magpylib as magpy

# position and orientation paths are tiled up at init

# An operation applied to a collection is individually
# applied to BaseGeo and to each child.

# BaseGeo is mostly used for tracking the Collection

# pylint: disable=no-member


def validate_pos_orient(obj, ppath, opath_as_rotvec):
    """ test object position and orientation"""
    sp = obj.position
    so = obj.orientation
    ppath = np.array(ppath)
    opath = R.from_rotvec(opath_as_rotvec)
    assert ppath.shape == sp.shape, (
        "position shapes do not match"
        f"\n object has {sp.shape} instead of {ppath.shape}"
    )
    assert opath.as_rotvec().shape == so.as_rotvec().shape, (
        "orientation as_rotvec shapes do not match"
        f"\n object has {so.as_rotvec().shape} instead of {opath.as_rotvec().shape}"
    )
    assert np.allclose(sp, ppath), (
        f"position validation failed with ({sp})" f"\n expected {ppath}"
    )
    assert np.allclose(so.as_matrix(), opath.as_matrix()), (
        f"orientation validation failed with ({so.as_rotvec()})"
        f"\n expected {opath_as_rotvec}"
    )


###############################################################################
# INIT OBJECT TESTING

# at initialization position and orientation to have similar shape (N,3)
# - if inputs are (N,3) and (3,) then the (3,) is tiled up to (N,3)
# - if inputs are (N,3) and (M,3) then the smaller one is padded up
# - None orientation input is interpreted as (0,0,0) rotvec / (0,0,0,1) quat


def get_init_pos_orient_test_data():
    """init_position, init_orientation_rotvec, expected_position, expected_orientation_rotvec"""
    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [(1, 2, 3), (1, 1, 1)]
    o0 = None
    o1 = (0, 0, 0.1)
    o2 = [(0, 0, 0.1)]
    o3 = [(0, 0, 0.1), (0, 0, 0.2)]
    o4 = [(0, 0, 0.1), (0, 0, 0.2), (0, 0, 0.3)]

    init_test_data = [
        [p0, o0, p0, (0, 0, 0)],
        [p0, o1, p0, o1],
        [p0, o2, p0, o1],
        [p0, o3, (p0, p0), o3],
        [p1, o0, p0, (0, 0, 0)],
        [p1, o1, p0, o1],
        [p1, o2, p0, o1],
        [p1, o3, (p0, p0), o3],
        [p2, o0, p2, [(0, 0, 0)] * 2],
        [p2, o1, p2, [o1] * 2],
        [p2, o2, p2, [o1] * 2],
        [p2, o3, p2, o3],
        [p2, o4, p2 + [(1, 1, 1)], o4],  # uneven paths
    ]
    return init_test_data


@pytest.mark.parametrize(
    "init_position, init_orientation_rotvec, expected_position, expected_orientation_rotvec",
    get_init_pos_orient_test_data(),
    ids=[f"{ind+1:02d}" for ind, t in enumerate(get_init_pos_orient_test_data())],
)
def test_init_pos_orient(
    init_position,
    init_orientation_rotvec,
    expected_position,
    expected_orientation_rotvec,
):
    """test position and orientation initialization"""
    # print(init_position, init_orientation_rotvec, expected_position, expected_orientation_rotvec)
    if init_orientation_rotvec is None:
        init_orientation_rotvec = (0, 0, 0)
    src = magpy.magnet.Cuboid(
        (1, 0, 0),
        (1, 1, 1),
        position=init_position,
        orientation=R.from_rotvec(init_orientation_rotvec),
    )
    validate_pos_orient(src, expected_position, expected_orientation_rotvec)


############################################################################
# SETTER OBJECT TESTING

# object position setting
# - when position or orientation are set by user, the other attribute is not touched
#   advantage: the user is not aware of this happening and possibly wants to access
#              what he thinks is in the orientation path. Any auto-tiling requires
#              a complicated set of rules
# -> possible extended auto-tiling functionality - possibly v4.1 ?:
#    - if one is set to shape (N,3) and the other is (3,) then the other is tiled up
#    - what happens if one is set to (N,3) the other is still (M,3) for M<N and M>N ?

# when setting a position the orientation remains unchanged


def get_setter_pos_orient_test_data():
    """get test data as dictionaries for position and orientation setter testing"""
    return [
        dict(
            init_position=[(1, 1, 1), (2, 2, 2)],
            init_orientation_rotvec=None,
            setter_position=(0, 0, 1),
            setter_orientation_rotvec="not_set",
            expected_position=(0, 0, 1),
            expected_orientation_rotvec=[(0, 0, 0), (0, 0, 0)],
        ),
        dict(
            init_position=[(1, 1, 1), (2, 2, 2)],
            init_orientation_rotvec=None,
            setter_position="not_set",
            setter_orientation_rotvec=(0, 0, 0.1),
            expected_position=[(1, 1, 1), (2, 2, 2)],
            expected_orientation_rotvec=(0, 0, 0.1),
        ),
    ]


@pytest.mark.parametrize(
    (
        "init_position",
        "init_orientation_rotvec",
        "setter_position",
        "setter_orientation_rotvec",
        "expected_position",
        "expected_orientation_rotvec",
    ),
    [d.values() for d in get_setter_pos_orient_test_data()],
)
def test_setter_pos_orient(
    init_position,
    init_orientation_rotvec,
    setter_position,
    setter_orientation_rotvec,
    expected_position,
    expected_orientation_rotvec,
):
    """testing position and orientation setters"""
    # print (locals())
    if init_orientation_rotvec is None:
        init_orientation_rotvec = (0, 0, 0)
    src = magpy.magnet.Cuboid(
        (1, 0, 0),
        (1, 1, 1),
        position=init_position,
        orientation=R.from_rotvec(init_orientation_rotvec),
    )
    if setter_position != "not_set":
        src.position = setter_position
    if setter_orientation_rotvec != "not_set":
        src.orientation = R.from_rotvec(setter_orientation_rotvec)
    validate_pos_orient(src, expected_position, expected_orientation_rotvec)


def get_anchor_pos_orient_test_data():
    """get test data as dictionaries for multi anchor testing"""
    data = [
        dict(
            description="scalar path - scalar anchor",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=90,
            axis="z",
            anchor=0,
            start="auto",
            expected_position=(0, 0, 0),
            expected_orientation_rotvec=(0, 0, np.pi / 2),
        ),
        dict(
            description="vector path 1 - scalar anchor",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=[90],
            axis="z",
            anchor=(1, 0, 0),
            start="auto",
            expected_position=[(0, 0, 0), (1, -1, 0)],
            expected_orientation_rotvec=[(0, 0, 0), (0, 0, np.pi / 2)],
        ),
        dict(
            description="vector path 2 - scalar anchor",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=[90, 270],
            axis="z",
            anchor=(1, 0, 0),
            start="auto",
            expected_position=[(0, 0, 0), (1, -1, 0), (1, 1, 0)],
            expected_orientation_rotvec=[
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, -np.pi / 2),
            ],
        ),
        dict(
            description="scalar path - vector anchor 1",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=90,
            axis="z",
            anchor=[(1, 0, 0)],
            start="auto",
            expected_position=[(0, 0, 0), (1, -1, 0)],
            expected_orientation_rotvec=[(0, 0, 0), (0, 0, np.pi / 2)],
        ),
        dict(
            description="scalar path - vector anchor 2",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=90,
            axis="z",
            anchor=[(1, 0, 0), (2, 0, 0)],
            start="auto",
            expected_position=[(0, 0, 0), (1, -1, 0), (2, -2, 0)],
            expected_orientation_rotvec=[
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, np.pi / 2),
            ],
        ),
        dict(
            description="vector path 2 - vector anchor 2",
            init_position=(0, 0, 0),
            init_orientation_rotvec=None,
            angle=[90, 270],
            axis="z",
            anchor=[(1, 0, 0), (2, 0, 0)],
            start="auto",
            expected_position=[(0, 0, 0), (1, -1, 0), (2, 2, 0)],
            expected_orientation_rotvec=[
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, -np.pi / 2),
            ],
        ),
        dict(
            description="vector path 2 - vector anchor 2 - path 2 - start=0",
            init_position=[(0, 0, 0), (2, 1, 0)],
            init_orientation_rotvec=None,
            angle=[90, 270],
            axis="z",
            anchor=[(1, 0, 0), (2, 0, 0)],
            start=0,
            expected_position=[(1, -1, 0), (3, 0, 0)],
            expected_orientation_rotvec=[(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
        ),
        dict(
            description="init crazy, path 2, anchor 3, start middle",
            init_position=[(0, 0, 0), (2, 0, 0)],
            init_orientation_rotvec=(0, 0, 0.1),
            angle=[90, 270],
            axis="z",
            anchor=[(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            start=1,
            expected_position=[(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)],
            expected_orientation_rotvec=[
                (0, 0, 0.1),
                (0, 0, np.pi / 2 + 0.1),
                (0, 0, -np.pi / 2 + 0.1),
                (0, 0, -np.pi / 2 + 0.1),
            ],
        ),
        dict(
            description="init crazy, path 2, anchor 3, start before",
            init_position=[(0, 0, 0), (2, 0, 0)],
            init_orientation_rotvec=(0, 0, 0.1),
            angle=[90, 270],
            axis="z",
            anchor=[(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            start=-4,
            expected_position=[(1, -1, 0), (2, 2, 0), (3, 3, 0), (2, 0, 0)],
            expected_orientation_rotvec=[
                (0, 0, 0.1 + np.pi / 2),
                (0, 0, 0.1 - np.pi / 2),
                (0, 0, 0.1 - np.pi / 2),
                (0, 0, 0.1),
            ],
        ),
    ]
    return data


@pytest.mark.parametrize(
    (
        "description",
        "init_position",
        "init_orientation_rotvec",
        "angle",
        "axis",
        "anchor",
        "start",
        "expected_position",
        "expected_orientation_rotvec",
    ),
    [d.values() for d in get_anchor_pos_orient_test_data()],
    ids=[d["description"] for d in get_anchor_pos_orient_test_data()],
)
def test_anchor_pos_orient(
    description,
    init_position,
    init_orientation_rotvec,
    angle,
    axis,
    anchor,
    start,
    expected_position,
    expected_orientation_rotvec,
):
    """testing multi anchors"""
    print(description)
    # print(locals())
    if init_orientation_rotvec is None:
        init_orientation_rotvec = (0, 0, 0)
    src = magpy.magnet.Cuboid(
        (1, 0, 0),
        (1, 1, 1),
        position=init_position,
        orientation=R.from_rotvec(init_orientation_rotvec),
    )
    src.rotate_from_angax(angle, axis, anchor, start)
    validate_pos_orient(src, expected_position, expected_orientation_rotvec)


############################################################################
# COLLECTION - COMPOUND BEHAVIOR TESTS


def test_compound_00():
    """init Collection should not change source pos and ori"""
    src = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 2, 3), (2, 3, 4)])
    validate_pos_orient(src, [(1, 2, 3), (2, 3, 4)], [(0, 0, 0)] * 2)
    col = magpy.Collection(src, position=[(1, 1, 1)])
    validate_pos_orient(src, [(1, 2, 3), (2, 3, 4)], [(0, 0, 0)] * 2)
    print(col)


# test: An operation move() or rotate() applied to a Collection is
# individually applied to BaseGeo and to each child:


def test_compound_01():
    """very sensible Compound behavior with rotation anchor"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 1, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (-1, -1, -1))
    col = magpy.Collection(s1, s2)
    col.move((0, 0, 1))
    validate_pos_orient(s1, (1, 1, 2), (0, 0, 0))
    validate_pos_orient(s2, (-1, -1, 0), (0, 0, 0))
    validate_pos_orient(col, (0, 0, 1), (0, 0, 0))
    col.move([(0, 0, 1)])
    validate_pos_orient(s1, [(1, 1, 2), (1, 1, 3)], [(0, 0, 0)] * 2)
    validate_pos_orient(s2, [(-1, -1, 0), (-1, -1, 1)], [(0, 0, 0)] * 2)
    validate_pos_orient(col, [(0, 0, 1), (0, 0, 2)], [(0, 0, 0)] * 2)
    col.rotate_from_rotvec((0, 0, np.pi / 2), anchor=0)
    validate_pos_orient(s1, [(-1, 1, 2), (-1, 1, 3)], [(0, 0, np.pi / 2)] * 2)
    validate_pos_orient(s2, [(1, -1, 0), (1, -1, 1)], [(0, 0, np.pi / 2)] * 2)
    validate_pos_orient(col, [(0, 0, 1), (0, 0, 2)], [(0, 0, np.pi / 2)] * 2)


def test_compound_02():
    """very sensible Compound behavior with vector anchor"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (-1, 0, -1))
    col = magpy.Collection(s1, s2, position=(3, 0, 3))
    col.rotate_from_rotvec((0, 0, np.pi / 2), anchor=[(1, 0, 0), (2, 0, 0)])
    validate_pos_orient(
        s1,
        [(1, 0, 1), (1, 0, 1), (2, -1, 1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        s2,
        [(-1, 0, -1), (1, -2, -1), (2, -3, -1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(3, 0, 3), (1, 2, 3), (2, 1, 3)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, np.pi / 2)],
    )


def test_compound_03():
    """very sensible Compound behavior with vector path and anchor and start=0"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(3, 0, 0), (1, 0, 0)])
    s2 = magpy.magnet.Cuboid(
        (1, 0, 0),
        (1, 1, 1),
        [(2, 0, 2), (2, 0, 2)],
        R.from_rotvec([(0, 0, -0.1), (0, 0, -0.2)]),
    )
    col = magpy.Collection(s1, s2, position=[(3, 0, 2), (3, 0, 3)])
    col.rotate_from_rotvec(
        [(0, 0, np.pi / 2), (0, 0, 3 * np.pi / 2)],
        anchor=[(1, 0, 0), (2, 0, 0)],
        start=0,
    )
    validate_pos_orient(
        s1, [(1, 2, 0), (2, 1, 0)], [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)]
    )
    validate_pos_orient(
        s2, [(1, 1, 2), (2, 0, 2)], [(0, 0, np.pi / 2 - 0.1), (0, 0, -np.pi / 2 - 0.2)]
    )
    validate_pos_orient(
        col, [(1, 2, 2), (2, -1, 3)], [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)]
    )


def test_compound_04():
    """nonsensical but correct Collection behavior when col and children
    all have different path formats"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), position=(1, 1, 1))
    s2 = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), orientation=R.from_rotvec([(0, 0, -0.1), (0, 0, -0.2)])
    )
    col = magpy.Collection(s1, s2, position=[(1, 2, 3), (1, 3, 4)])
    col.rotate_from_angax(90, "z", anchor=(1, 0, 0))
    validate_pos_orient(s1, (0, 0, 1), (0, 0, np.pi / 2))
    validate_pos_orient(
        s2, [(1, -1, 0)] * 2, [(0, 0, np.pi / 2 - 0.1), (0, 0, np.pi / 2 - 0.2)]
    )
    validate_pos_orient(col, [(-1, 0, 3), (-2, 0, 4)], [(0, 0, np.pi / 2)] * 2)


def test_compound_05():
    """nonsensical but correct Collection behavior with vector anchor"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), position=(1, 0, 1))
    s2 = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), orientation=R.from_rotvec([(0, 0, -0.1), (0, 0, -0.2)])
    )
    col = magpy.Collection(s1, s2, position=[(3, 0, 3), (4, 0, 4)])
    col.rotate_from_angax(90, "z", anchor=[(1, 0, 0), (2, 0, 0)])
    validate_pos_orient(
        s1,
        [(1, 0, 1), (1, 0, 1), (2, -1, 1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        s2,
        [(0, 0, 0), (0, 0, 0), (1, -1, 0), (2, -2, 0)],
        [(0, 0, -0.1), (0, 0, -0.2), (0, 0, np.pi / 2 - 0.2), (0, 0, np.pi / 2 - 0.2)],
    )
    validate_pos_orient(
        col,
        [(3, 0, 3), (4, 0, 4), (1, 3, 4), (2, 2, 4)],
        [(0, 0, 0), (0, 0, 0), (0, 0, np.pi / 2), (0, 0, np.pi / 2)],
    )


def test_compound_06():
    """Compound rotation (anchor=None), scalar input, scalar pos"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (0, -1, -1))
    col = magpy.Collection(s1, s2)
    col.rotate_from_angax(90, "z")
    validate_pos_orient(s1, (0, 1, 1), (0, 0, np.pi / 2))
    validate_pos_orient(s2, (1, 0, -1), (0, 0, np.pi / 2))
    validate_pos_orient(col, (0, 0, 0), (0, 0, np.pi / 2))


def test_compound_07():
    """Compound rotation (anchor=None), scalar input, vector pos, start=auto"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((0, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z")
    validate_pos_orient(
        s1, [(0, 1, 0), (1, 1, 0)], [(0, 0, np.pi / 2), (0, 0, np.pi / 2)]
    )
    validate_pos_orient(
        s2, [(0, -1, 0), (1, -3, 0)], [(0, 0, np.pi / 2), (0, 0, np.pi / 2)]
    )
    validate_pos_orient(
        col, [(0, 0, 0), (1, 0, 0)], [(0, 0, np.pi / 2), (0, 0, np.pi / 2)]
    )


def test_compound_08():
    """Compound rotation (anchor=None), scalar input, vector pos, start=1"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((0, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=1)
    validate_pos_orient(s1, [(1, 0, 0), (1, 1, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(s2, [(-1, 0, 0), (1, -3, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(col, [(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])


def test_compound_09():
    """Compound rotation (anchor=None), scalar input, vector pos, start=-1"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((0, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=-1)
    validate_pos_orient(s1, [(1, 0, 0), (1, 1, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(s2, [(-1, 0, 0), (1, -3, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(col, [(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])


def test_compound_10():
    """Compound rotation (anchor=None), scalar input, vector pos, start->pad before"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((2, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=-4)
    validate_pos_orient(
        s1, [(2, -1, 0), (2, -1, 0), (2, -1, 0), (1, 1, 0)], [(0, 0, np.pi / 2)] * 4,
    )
    validate_pos_orient(
        s2, [(2, -3, 0), (2, -3, 0), (2, -3, 0), (1, -3, 0)], [(0, 0, np.pi / 2)] * 4,
    )
    validate_pos_orient(
        col, [(2, 0, 0), (2, 0, 0), (2, 0, 0), (1, 0, 0)], [(0, 0, np.pi / 2)] * 4,
    )


def test_compound_11():
    """Compound rotation (anchor=None), scalar input, vector pos, start->pad behind"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((2, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=3)
    validate_pos_orient(
        s1,
        [(1, 0, 0), (2, 0, 0), (2, 0, 0), (1, 1, 0)],
        [(0, 0, 0)] * 3 + [(0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        s2,
        [(-1, 0, 0), (-2, 0, 0), (-2, 0, 0), (1, -3, 0)],
        [(0, 0, 0)] * 3 + [(0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(2, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)],
        [(0, 0, 0)] * 3 + [(0, 0, np.pi / 2)],
    )


def test_compound_12():
    """Compound rotation (anchor=None), vector input, simple pos, start=auto"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (0, -1, -1))
    col = magpy.Collection(s1, s2)
    col.rotate_from_angax([90, -90], "z")
    validate_pos_orient(
        s1,
        [(1, 0, 1), (0, 1, 1), (0, -1, 1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )
    validate_pos_orient(
        s2,
        [(0, -1, -1), (1, 0, -1), (-1, 0, -1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )


def test_compound_13():
    """Compound rotation (anchor=None), vector input, vector pos, start=1"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (0, -1, -1))
    col = magpy.Collection(s1, s2)
    col.rotate_from_angax([90, -90], "z")
    col.rotate_from_angax([-90, 180], "z", start=1)
    validate_pos_orient(
        s1, [(1, 0, 1), (1, 0, 1), (0, 1, 1)], [(0, 0, 0), (0, 0, 0), (0, 0, np.pi / 2)]
    )
    validate_pos_orient(
        s2,
        [(0, -1, -1), (0, -1, -1), (1, 0, -1)],
        [(0, 0, 0), (0, 0, 0), (0, 0, np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (0, 0, np.pi / 2)],
    )


def test_compound_14():
    """Compound rotation (anchor=None), vector input, vector pos, start=1, pad_behind"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (0, -1, -1))
    col = magpy.Collection(s1, s2)
    col.rotate_from_angax([90, -90], "z")
    col.rotate_from_angax([-90, 180], "z", start=1)
    col.rotate_from_angax([90, 180, -90], "z", start=1)
    validate_pos_orient(
        s1,
        [(1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 0, 1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )
    validate_pos_orient(
        s2,
        [(0, -1, -1), (1, 0, -1), (-1, 0, -1), (0, -1, -1)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )
    validate_pos_orient(
        col,
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )


def test_compound_15():
    """Compound rotation (anchor=None), vector input, simple pos, start=-3, pad_before"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (-1, 0, -1))
    col = magpy.Collection(s1, s2, position=(2, 0, 0))
    col.rotate_from_angax([90, -90], "z", start=-3)
    validate_pos_orient(
        s1,
        [(2, -1, 1), (2, 1, 1), (1, 0, 1)],
        [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )
    validate_pos_orient(
        s2,
        [(2, -3, -1), (2, 3, -1), (-1, 0, -1)],
        [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )
    validate_pos_orient(
        col,
        [(2, 0, 0), (2, 0, 0), (2, 0, 0)],
        [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, 0)],
    )


def test_compound_16():
    """Compound rotation (anchor=None), vector input, vector pos, start=-3,
    pad_before AND pad_behind"""
    s1 = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), orientation=R.from_rotvec([(0, 0, 0.1), (0, 0, 0.2)])
    )
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), position=[(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=[(1, 0, 0), (0, 0, 0)])
    col.rotate_from_angax([90, -90, 90, -90], "z", start=-3)
    validate_pos_orient(
        s1,
        [(1, -1, 0), (1, 1, 0), (0, 0, 0), (0, 0, 0)],
        [
            (0, 0, 0.1 + np.pi / 2),
            (0, 0, 0.1 - np.pi / 2),
            (0, 0, 0.2 + np.pi / 2),
            (0, 0, 0.2 - np.pi / 2),
        ],
    )
    validate_pos_orient(
        s2,
        [(1, -2, 0), (1, 2, 0), (0, -2, 0), (0, 2, 0)],
        [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(1, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, np.pi / 2), (0, 0, -np.pi / 2), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )


def test_compound_17():
    """CRAZY Compound rotation (anchor=None) with messy path formats"""
    s1 = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), orientation=R.from_rotvec([(0, 0, 0.1), (0, 0, 0.2)])
    )
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), position=(-1, 0, 0))
    col = magpy.Collection(s1, s2, position=[(1, 0, 0), (0, 0, 0), (3, 0, 3)])
    col.rotate_from_angax([90, -90], "z", start="auto")
    validate_pos_orient(
        s1,
        [(0, 0, 0), (0, 0, 0), (3, -3, 0), (3, 3, 0)],
        [(0, 0, 0.1), (0, 0, 0.2), (0, 0, 0.2 + np.pi / 2), (0, 0, 0.2 - np.pi / 2)],
    )
    validate_pos_orient(
        s2,
        [(-1, 0, 0), (3, -4, 0), (3, 4, 0)],
        [(0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )
    validate_pos_orient(
        col,
        [(1, 0, 0), (0, 0, 0), (3, 0, 3), (3, 0, 3), (3, 0, 3)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
    )
