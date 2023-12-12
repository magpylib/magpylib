import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy


###############################################################################
###############################################################################
# NEW COLLECTION POS/ORI TESTS FROM v4


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


############################################################################
############################################################################
# COLLECTION POS/ORI SETTER TESTING
# when setting pos/ori of a collection, the children retain their original
# relative position and orientation in the Collection


def get_data_collection_position_setter():
    """
    returns data for collection setter tests
    Args:
    col_pos_init, col_ori_init, src_pos_init, src_ori_init
    col_pos_test, col_ori_test, src_pos_test, src_ori_test
    """
    data_pos = [
        [
            (1, 2, 3),
            (0.1, 0.2, 0.3),
            (1, 1, 1),
            (0, 0, -0.1),
            (3, 2, 1),
            (0.1, 0.2, 0.3),
            (3, 1, -1),
            (0, 0, -0.1),
        ],
        [
            [(1, 2, 3), (2, 3, 4)],
            [(0, 0, 0)] * 2,
            [(1, 1, 1), (2, 2, 2)],
            [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2)],
            (4, 5, 6),
            (0, 0, 0),
            (4, 4, 4),
            (0.2, 0.2, 0.2),
        ],
        [
            [(1, 2, 3), (2, 3, 4)],
            [(0, 0, 0)] * 2,
            [(1, 1, 1), (2, 2, 2)],
            [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2)],
            [(4, 5, 6), (5, 6, 7), (6, 7, 8)],
            [(0, 0, 0)] * 3,
            [(4, 4, 4), (5, 5, 5), (6, 6, 6)],
            [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)],
        ],
        [
            (1, 2, 3),
            (0, 0, 0),
            [(1, 1, 1), (2, 2, 2)],
            [(0.1, 0.1, 0.1)],
            [(4, 5, 6), (5, 6, 7), (6, 7, 8)],
            [(0, 0, 0)] * 3,
            [(4, 4, 4), (6, 6, 6), (7, 7, 7)],
            [(0.1, 0.1, 0.1)] * 3,
        ],
    ]
    return data_pos


@pytest.mark.parametrize(
    """col_pos_init, col_ori_init, src_pos_init, src_ori_init,
    col_pos_test, col_ori_test, src_pos_test, src_ori_test""",
    get_data_collection_position_setter(),
    ids=[f"{ind+1:02d}" for ind, _ in enumerate(get_data_collection_position_setter())],
)
def test_Collection_setting_position(
    col_pos_init,
    col_ori_init,
    src_pos_init,
    src_ori_init,
    col_pos_test,
    col_ori_test,
    src_pos_test,
    src_ori_test,
):
    """Test position and orientation setters on Collection"""
    src = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), src_pos_init, R.from_rotvec(src_ori_init)
    )
    col = magpy.Collection(
        src, position=col_pos_init, orientation=R.from_rotvec(col_ori_init)
    )
    col.position = col_pos_test
    validate_pos_orient(col, col_pos_test, col_ori_test)
    validate_pos_orient(src, src_pos_test, src_ori_test)


def get_data_collection_orientation_setter():
    """
    returns data for collection setter tests
    Args:
    col_pos_init, col_ori_init, src_pos_init, src_ori_init
    col_pos_test, col_ori_test, src_pos_test, src_ori_test
    """
    data_ori = [
        # col orientation setter simple
        [
            (1, 0, 3),
            (0, 0, np.pi / 4),
            (2, 0, 1),
            (0, 0, 0.1),
            (1, 0, 3),
            (0, 0, -np.pi / 4),
            (1, -1, 1),
            (0, 0, -np.pi / 2 + 0.1),
        ],
        # collection orientation setter with path
        [
            [(1, 0, 3), (2, 0, 3)],
            [(0, 0, 0)] * 2,
            [(2, 0, 1), (1, 0, 1)],
            [(0, 0, 0)] * 2,
            [(1, 0, 3), (2, 0, 3)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
            [(1, 1, 1), (2, 1, 1)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
        ],
        # collection orientation setter slice test
        [
            [(1, 0, 3), (2, 0, 3), (3, 0, 3)],
            [(0, 0, 0)] * 3,
            [(2, 0, 1), (1, 0, 1), (0, 0, 1)],
            (0, 0, 0),
            [(2, 0, 3), (3, 0, 3)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
            [(2, -1, 1), (3, 3, 1)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
        ],
        # collection orientation setter pad test
        [
            (3, 0, 3),
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 0),
            [(3, 0, 3)] * 2,
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
            [(3, -3, 1), (3, 3, 1)],
            [(0, 0, np.pi / 2), (0, 0, -np.pi / 2)],
        ],
        # crazy collection test with different path formats
        [
            [(0, 0, 0), (-1, 0, 0)],
            [(0, 0, 0)] * 2,
            (0, 0, 0),
            (0, 0, 0.1),
            (-1, 0, 0),
            (0, 0, np.pi / 2),
            (-1, 1, 0),
            (0, 0, np.pi / 2 + 0.1),
        ],
        # crazy collection test with different path formats pt2
        [
            [(0, 0, 0), (-1, 0, 0)],
            [(0, 0, 0)] * 2,
            [(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            [(0, 0, 0)] * 3,
            (-1, 0, 0),
            (0, 0, np.pi / 2),
            (-1, 4, 0),
            (0, 0, np.pi / 2),
        ],
    ]
    return data_ori


@pytest.mark.parametrize(
    """col_pos_init, col_ori_init, src_pos_init, src_ori_init, col_pos_test,
    col_ori_test, src_pos_test, src_ori_test""",
    get_data_collection_orientation_setter(),
    ids=[
        f"{ind+1:02d}" for ind, _ in enumerate(get_data_collection_orientation_setter())
    ],
)
def test_Collection_setting_orientation(
    col_pos_init,
    col_ori_init,
    src_pos_init,
    src_ori_init,
    col_pos_test,
    col_ori_test,
    src_pos_test,
    src_ori_test,
):
    """test_Collection_setting_orientation"""
    src = magpy.magnet.Cuboid(
        (1, 0, 0), (1, 1, 1), src_pos_init, R.from_rotvec(src_ori_init)
    )
    col = magpy.Collection(
        src, position=col_pos_init, orientation=R.from_rotvec(col_ori_init)
    )
    col.orientation = R.from_rotvec(col_ori_test)
    validate_pos_orient(col, col_pos_test, col_ori_test)
    validate_pos_orient(src, src_pos_test, src_ori_test)


def test_Collection_setter():
    """
    general col position and orientation setter testing
    """
    # graphical test: is the Collection moving/rotating as a whole ?
    # col0 = magpy.Collection()
    # for poz,roz in zip(
    #     [(0,0,0), (0,0,5), (5,0,0), (5,0,5), (10,0,0), (10,0,5)],
    #     [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,2,3), (-2,-1,3)]
    #     ):
    #     col = magpy.Collection()
    #     for i,color in enumerate(['r', 'orange', 'gold', 'green', 'cyan']):
    #         src = magpy.magnet.Cuboid((1,0,0), (.5,.5,.5), (1,0,0), style_color=color)
    #         src.rotate_from_angax(72*i, 'z', (0,0,0))
    #         col = col + src
    #     base = magpy.Sensor()
    #     col.position = poz
    #     col.orientation = R.from_rotvec(roz)
    #     base.position = poz
    #     base.orientation = R.from_rotvec(roz)
    #     col0 = col0 + col + base
    # magpy.show(*col0)
    POS = []
    ORI = []
    for poz, roz in zip(
        [(0, 0, 0), (0, 0, 5), (5, 0, 0), (5, 0, 5), (10, 0, 0), (10, 0, 5)],
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (-2, -1, 3)],
    ):
        col = magpy.Collection()
        for i in range(5):
            src = magpy.magnet.Cuboid((1, 0, 0), (0.5, 0.5, 0.5), (1, 0, 0))
            src.rotate_from_angax(72 * i, "z", (0, 0, 0))
            col.add(src)
        col.position = poz
        col.orientation = R.from_rotvec(roz)

        POS += [[src.position for src in col]]
        ORI += [[src.orientation.as_rotvec() for src in col]]

    test_POS, test_ORI = np.load("tests/testdata/testdata_Collection_setter.npy")

    assert np.allclose(POS, test_POS)
    assert np.allclose(ORI, test_ORI)


############################################################################
############################################################################
# COLLECTION MOTION TESTS
# An operation move() or rotate() applied to a Collection is
# individually applied to BaseGeo and to each child:


def test_compound_motion_00():
    """init Collection should not change source pos and ori"""
    src = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 2, 3), (2, 3, 4)])
    validate_pos_orient(src, [(1, 2, 3), (2, 3, 4)], [(0, 0, 0)] * 2)
    col = magpy.Collection(src, position=[(1, 1, 1)])
    validate_pos_orient(src, [(1, 2, 3), (2, 3, 4)], [(0, 0, 0)] * 2)
    print(col)


def test_compound_motion_01():
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
    col.rotate_from_rotvec((0, 0, np.pi / 2), anchor=0, degrees=False)
    validate_pos_orient(s1, [(-1, 1, 2), (-1, 1, 3)], [(0, 0, np.pi / 2)] * 2)
    validate_pos_orient(s2, [(1, -1, 0), (1, -1, 1)], [(0, 0, np.pi / 2)] * 2)
    validate_pos_orient(col, [(0, 0, 1), (0, 0, 2)], [(0, 0, np.pi / 2)] * 2)


def test_compound_motion_02():
    """very sensible Compound behavior with vector anchor"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (-1, 0, -1))
    col = magpy.Collection(s1, s2, position=(3, 0, 3))
    col.rotate_from_rotvec(
        (0, 0, np.pi / 2), anchor=[(1, 0, 0), (2, 0, 0)], degrees=False
    )
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


def test_compound_motion_03():
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
        degrees=False,
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


def test_compound_motion_04():
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


def test_compound_motion_05():
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


def test_compound_motion_06():
    """Compound rotation (anchor=None), scalar input, scalar pos"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (1, 0, 1))
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), (0, -1, -1))
    col = magpy.Collection(s1, s2)
    col.rotate_from_angax(90, "z")
    validate_pos_orient(s1, (0, 1, 1), (0, 0, np.pi / 2))
    validate_pos_orient(s2, (1, 0, -1), (0, 0, np.pi / 2))
    validate_pos_orient(col, (0, 0, 0), (0, 0, np.pi / 2))


def test_compound_motion_07():
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


def test_compound_motion_08():
    """Compound rotation (anchor=None), scalar input, vector pos, start=1"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((0, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=1)
    validate_pos_orient(s1, [(1, 0, 0), (1, 1, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(s2, [(-1, 0, 0), (1, -3, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(col, [(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])


def test_compound_motion_09():
    """Compound rotation (anchor=None), scalar input, vector pos, start=-1"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((0, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=-1)
    validate_pos_orient(s1, [(1, 0, 0), (1, 1, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(s2, [(-1, 0, 0), (1, -3, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])
    validate_pos_orient(col, [(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (0, 0, np.pi / 2)])


def test_compound_motion_10():
    """Compound rotation (anchor=None), scalar input, vector pos, start->pad before"""
    s1 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(1, 0, 0), (2, 0, 0)])
    s2 = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1), [(-1, 0, 0), (-2, 0, 0)])
    col = magpy.Collection(s1, s2, position=((2, 0, 0), (1, 0, 0)))
    col.rotate_from_angax(90, "z", start=-4)
    validate_pos_orient(
        s1,
        [(2, -1, 0), (2, -1, 0), (2, -1, 0), (1, 1, 0)],
        [(0, 0, np.pi / 2)] * 4,
    )
    validate_pos_orient(
        s2,
        [(2, -3, 0), (2, -3, 0), (2, -3, 0), (1, -3, 0)],
        [(0, 0, np.pi / 2)] * 4,
    )
    validate_pos_orient(
        col,
        [(2, 0, 0), (2, 0, 0), (2, 0, 0), (1, 0, 0)],
        [(0, 0, np.pi / 2)] * 4,
    )


def test_compound_motion_11():
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


def test_compound_motion_12():
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


def test_compound_motion_13():
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


def test_compound_motion_14():
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


def test_compound_motion_15():
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


def test_compound_motion_16():
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


def test_compound_motion_17():
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
