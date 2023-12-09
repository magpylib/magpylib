import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy

###############################################################################
###############################################################################
# NEW BASE GEO TESTS FROM v4


def validate_pos_orient(obj, ppath, opath_as_rotvec):
    """test position (ppath) and orientation (opath) of BaseGeo object (obj)"""
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
###############################################################################
# BASEGEO POS/ORI INIT TESTING
# at initialization position and orientation must have similar shape (N,3)
# - if inputs are (N,3) and (3,) then the (3,) is tiled up to (N,3)
# - if inputs are (N,3) and (M,3) then the smaller one is padded up
# - None orientation input is interpreted as (0,0,0) rotvec == (0,0,0,1) quat


def get_init_pos_orient_test_data():
    """
    returns data for object init testing

    init_position, init_orientation_rotvec, expected_position, expected_orientation_rotvec
    """
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
def test_BaseGeo_init(
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
############################################################################
# BASEGEO POS/ORI SETTER TESTING
# when pos/ori is set then ori/pos is edge-padded / end-sliced to similar shape.


def get_data_object_setter(inp):
    """
    returns data for object setter tests

    init_pos, init_ori, test_pos, test_ori
    """
    # test positions
    p1 = (1, 2, 3)
    p3 = [(2, 3, 4), (3, 4, 5), (4, 5, 6)]

    # test orientations
    o1 = (0.1, 0.2, 0.3)
    o3 = [(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5)]

    # init states
    P1 = (1, 1, 1)
    O1 = (0.1, 0.1, 0.1)
    P2 = [(2, 2, 2), (3, 3, 3)]
    O2 = [(0.2, 0.2, 0.2), (0.3, 0.3, 0.3)]
    P3 = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
    O3 = [(0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4)]
    P4 = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
    O4 = [(0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5, 0.5)]

    test_data_pos = [
        # position init, orientation init, set/test position, test orientation
        (P1, O1, p1, O1),
        (P1, O1, p3, [O1] * 3),  # edge-pad
        (P2, O2, p1, O2[1]),  # end-slice
        (P2, O2, p3, O2 + [(0.3, 0.3, 0.3)]),  # edge-pad
        (P3, O3, p1, O3[2]),  # end-slice
        (P3, O3, p3, O3),
        (P4, O4, p1, O4[3]),  # end-slice
        (P4, O4, p3, O4[1:]),  # end-slice
    ]

    test_data_ori = [
        # position init, orientation init, set/test position, test orientation
        (P1, O1, P1, o1),
        (P1, O1, [P1] * 3, o3),  # edge-pad
        (P2, O2, P2[1], o1),  # end-slice
        (P2, O2, P2 + [P2[1]], o3),  # edge-pad
        (P3, O3, P3[-1], o1),  # end-slice
        (P3, O3, P3, o3),
        (P4, O4, P4[-1], o1),  # end-slice
        (P4, O4, P4[1:], o3),  # end-slice
    ]
    if inp == "pos":
        return test_data_pos
    return test_data_ori


@pytest.mark.parametrize(
    "init_pos, init_ori, test_pos, test_ori",
    get_data_object_setter("pos"),
    ids=[f"{ind+1:02d}" for ind, _ in enumerate(get_data_object_setter("pos"))],
)
def test_BaseGeo_setting_position(
    init_pos,
    init_ori,
    test_pos,
    test_ori,
):
    """test position and orientation initialization"""
    src = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), init_pos, R.from_rotvec(init_ori))
    src.position = test_pos
    validate_pos_orient(src, test_pos, test_ori)


@pytest.mark.parametrize(
    "init_pos, init_ori, test_pos, test_ori",
    get_data_object_setter("ori"),
    ids=[f"{ind+1:02d}" for ind, _ in enumerate(get_data_object_setter("ori"))],
)
def test_BaseGeo_setting_orientation(
    init_pos,
    init_ori,
    test_pos,
    test_ori,
):
    """test position and orientation initialization"""
    src = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), init_pos, R.from_rotvec(init_ori))
    src.orientation = R.from_rotvec(test_ori)
    validate_pos_orient(src, test_pos, test_ori)


###############################################################################
###############################################################################
# BASEGEO MULTI-ANCHOR ROTATION TESTING


def get_data_BaseGeo_multianchor_rotation():
    """get test data as dictionaries for multi anchor testing"""
    data = [
        {
            "description": "scalar path - scalar anchor",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": 90,
            "axis": "z",
            "anchor": 0,
            "start": "auto",
            "expected_position": (0, 0, 0),
            "expected_orientation_rotvec": (0, 0, np.pi / 2),
        },
        {
            "description": "vector path 1 - scalar anchor",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": [90],
            "axis": "z",
            "anchor": (1, 0, 0),
            "start": "auto",
            "expected_position": [(0, 0, 0), (1, -1, 0)],
            "expected_orientation_rotvec": [(0, 0, 0), (0, 0, np.pi / 2)],
        },
        {
            "description": "vector path 2 - scalar anchor",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": [90, 270],
            "axis": "z",
            "anchor": (1, 0, 0),
            "start": "auto",
            "expected_position": [(0, 0, 0), (1, -1, 0), (1, 1, 0)],
            "expected_orientation_rotvec": [
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, -np.pi / 2),
            ],
        },
        {
            "description": "scalar path - vector anchor 1",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": 90,
            "axis": "z",
            "anchor": [(1, 0, 0)],
            "start": "auto",
            "expected_position": [(0, 0, 0), (1, -1, 0)],
            "expected_orientation_rotvec": [(0, 0, 0), (0, 0, np.pi / 2)],
        },
        {
            "description": "scalar path - vector anchor 2",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": 90,
            "axis": "z",
            "anchor": [(1, 0, 0), (2, 0, 0)],
            "start": "auto",
            "expected_position": [(0, 0, 0), (1, -1, 0), (2, -2, 0)],
            "expected_orientation_rotvec": [
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, np.pi / 2),
            ],
        },
        {
            "description": "vector path 2 - vector anchor 2",
            "init_position": (0, 0, 0),
            "init_orientation_rotvec": None,
            "angle": [90, 270],
            "axis": "z",
            "anchor": [(1, 0, 0), (2, 0, 0)],
            "start": "auto",
            "expected_position": [(0, 0, 0), (1, -1, 0), (2, 2, 0)],
            "expected_orientation_rotvec": [
                (0, 0, 0),
                (0, 0, np.pi / 2),
                (0, 0, -np.pi / 2),
            ],
        },
        {
            "description": "vector path 2 - vector anchor 2 - path 2 - start=0",
            "init_position": [(0, 0, 0), (2, 1, 0)],
            "init_orientation_rotvec": None,
            "angle": [90, 270],
            "axis": "z",
            "anchor": [(1, 0, 0), (2, 0, 0)],
            "start": 0,
            "expected_position": [(1, -1, 0), (3, 0, 0)],
            "expected_orientation_rotvec": [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
        },
        {
            "description": "init crazy, path 2, anchor 3, start middle",
            "init_position": [(0, 0, 0), (2, 0, 0)],
            "init_orientation_rotvec": (0, 0, 0.1),
            "angle": [90, 270],
            "axis": "z",
            "anchor": [(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            "start": 1,
            "expected_position": [(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)],
            "expected_orientation_rotvec": [
                (0, 0, 0.1),
                (0, 0, np.pi / 2 + 0.1),
                (0, 0, -np.pi / 2 + 0.1),
                (0, 0, -np.pi / 2 + 0.1),
            ],
        },
        {
            "description": "init crazy, path 2, anchor 3, start before",
            "init_position": [(0, 0, 0), (2, 0, 0)],
            "init_orientation_rotvec": (0, 0, 0.1),
            "angle": [90, 270],
            "axis": "z",
            "anchor": [(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            "start": -4,
            "expected_position": [(1, -1, 0), (2, 2, 0), (3, 3, 0), (2, 0, 0)],
            "expected_orientation_rotvec": [
                (0, 0, 0.1 + np.pi / 2),
                (0, 0, 0.1 - np.pi / 2),
                (0, 0, 0.1 - np.pi / 2),
                (0, 0, 0.1),
            ],
        },
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
    [d.values() for d in get_data_BaseGeo_multianchor_rotation()],
    ids=[d["description"] for d in get_data_BaseGeo_multianchor_rotation()],
)
def test_BaseGeo_multianchor_rotation(
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
    """testing BaseGeo multi anchor rotations"""
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
