import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.utility import (
    LENGTH_UNITS,
    add_iteration_suffix,
    check_duplicates,
    filter_objects,
    get_unit_factor,
)


def test_duplicates():
    """test duplicate elimination and sorting"""
    pm1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    src_list = [pm1, pm2, pm1]
    with pytest.warns(UserWarning, match=r"Eliminating duplicate"):
        src_list_new = check_duplicates(src_list)
    assert src_list_new == [pm1, pm2], "duplicate elimination failed"


def test_filter_objects():
    """tests elimination of unwanted types"""
    pm1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    sens = magpy.Sensor()
    src_list = [pm1, pm2, sens]
    with pytest.warns(UserWarning, match=r"Cannot add Sensor.* to Collection"):
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
        err_msg="sens, sens should give same as [sens, sens]",
    )
    np.testing.assert_allclose(
        B3,
        B44,
        err_msg="sens, sens should give same as (sens, sens)",
    )

    B1 = sens.getH(pm1) * 4
    B2 = sens.getH(pm1, pm2, col, sumup=True)
    B3 = sens.getH([col]) * 2
    B4 = sens.getH([col, pm1, pm2], sumup=True)

    np.testing.assert_allclose(
        B1,
        B2,
        err_msg="src, src should give same as [src, src]",
    )
    np.testing.assert_allclose(
        B1,
        B3,
        err_msg="src should give same as [src]",
    )
    np.testing.assert_allclose(
        B1,
        B4,
        err_msg="src, src should give same as [src, src]",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
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


def test_length_units_matches_the_converter():
    """`LENGTH_UNITS` must list exactly what `get_unit_factor` accepts.

    The tuple feeds the `magpy.defaults.display.units.length` choice set, so a
    unit advertised there but rejected by the converter would pass validation
    and then raise at draw time.
    """
    for unit in LENGTH_UNITS:
        assert get_unit_factor(unit, target_unit="m") > 0

    # and nothing outside it gets through
    for bad in ("dam", "inch", "mT", "", "e"):
        with pytest.raises(ValueError, match="must be one of"):
            get_unit_factor(bad, target_unit="m")
