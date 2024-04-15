import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.utility import add_iteration_suffix
from magpylib._src.utility import check_duplicates
from magpylib._src.utility import filter_objects


def test_duplicates():
    """test duplicate elimination and sorting"""
    pm1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    src_list = [pm1, pm2, pm1]
    src_list_new = check_duplicates(src_list)
    assert src_list_new == [pm1, pm2], "duplicate elimination failed"


def test_filter_objects():
    """tests elimination of unwanted types"""
    pm1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    sens = magpy.Sensor()
    src_list = [pm1, pm2, sens]
    list_new = filter_objects(src_list, allow="sources")
    assert list_new == [pm1, pm2], "Failed to eliminate sensor"


def test_format_getBH_class_inputs():
    """special case testing of different input formats"""
    possis = [3, 3, 3]
    sens = magpy.Sensor(position=(3, 3, 3))
    pm1 = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    col = pm1 + pm2

    B1 = pm1.getB(possis)
    B2 = pm1.getB(sens)
    np.testing.assert_allclose(B1, B2, err_msg="pos_obs should give same as sens")

    B3 = pm1.getB(sens, sens)
    B4 = pm1.getB([sens, sens])
    B44 = pm1.getB((sens, sens))
    np.testing.assert_allclose(
        B3,
        B4,
        err_msg="sens,sens should give same as [sens,sens]",
    )
    np.testing.assert_allclose(
        B3,
        B44,
        err_msg="sens,sens should give same as (sens,sens)",
    )

    B1 = sens.getH(pm1) * 4
    B2 = sens.getH(pm1, pm2, col, sumup=True)
    B3 = sens.getH([col]) * 2
    B4 = sens.getH([col, pm1, pm2], sumup=True)

    np.testing.assert_allclose(
        B1,
        B2,
        err_msg="src,src should give same as [src,src]",
    )
    np.testing.assert_allclose(
        B1,
        B3,
        err_msg="src should give same as [src]",
    )
    np.testing.assert_allclose(
        B1,
        B4,
        err_msg="src,src should give same as [src,src]",
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("col", "col_01"),
        ("col_", "col_01"),
        ("col1", "col2"),
        ("col_02", "col_03"),
    ],
)
def test_add_iteration_suffix(name, expected):
    """check if iteration suffix works correctly"""
    assert add_iteration_suffix(name) == expected
