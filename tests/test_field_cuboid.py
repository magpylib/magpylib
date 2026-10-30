"""Cuboid fields close to edges must retain their finite physical values."""

from itertools import product

import numpy as np
import pytest

import magpylib as magpy


def test_cuboid_reported_mesh_cell():
    """The cell and observer from #1058 must not poison a collection's field."""
    cell = magpy.magnet.Cuboid(
        polarization=(0.25, 0, 0),
        dimension=(0.004545454545454546, 0.005, 0.005),
        position=(0.0022727272727272665, -0.0024999999999999996, -0.0075),
    )
    observer = (1.4414090374556621e-18, 0.0, -0.008453260901015891)
    # Surface-charge integrals on the six faces, evaluated at 90 decimal digits.
    expected = (0.17127053255816985, -1.3281449790135584, 0.012027218266562925)
    np.testing.assert_allclose(cell.getB(observer), expected, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("axis", range(3))
def test_cuboid_near_edges(axis):
    """Check all edge directions/octants, including finite but inaccurate fields."""
    dimension = np.roll([2.0, 4.0, 6.0], axis)
    polarization = np.roll([0.25, -0.17, 0.36], axis)
    observers = np.roll(
        [
            [1.000001, 1.999999, 0.7],
            [1.000000001, 1.999999999, 0.7],
            [1.000000000001, 1.999999999999, 0.7],
        ],
        axis,
        axis=1,
    )
    # Independent rectangular-face surface-charge integrals, evaluated with
    # mpmath at 90 decimal digits using the exact binary-float inputs above.
    expected = np.roll(
        [
            [-0.31342848854100824, 0.5910594099306999, -0.02750340127638744],
            [-0.5003270398543082, 0.865910261406929, -0.027503403088643803],
            [-0.6872232145655537, 1.140757577199482, -0.02750340309045605],
        ],
        axis,
        axis=1,
    )
    for signs in product([-1, 1], repeat=3):
        cell = magpy.magnet.Cuboid(
            dimension=dimension, polarization=polarization * signs
        )
        np.testing.assert_allclose(
            cell.getB(observers * signs), expected * signs, rtol=2e-13, atol=1e-14
        )


def test_cuboid_subdivision_near_shared_edge():
    """Internal surface charges must cancel when a uniformly polarized bar is split."""
    polarization = (0.25, -0.17, 0.36)
    whole = magpy.magnet.Cuboid(dimension=(2, 2, 2), polarization=polarization)
    cells = [
        magpy.magnet.Cuboid(
            dimension=(1, 1, 2), position=(x, y, 0), polarization=polarization
        )
        for x, y in product([-0.5, 0.5], repeat=2)
    ]
    observers = np.array([[1e-12, 2e-12, z] for z in np.linspace(-0.9, 0.9, 19)])
    np.testing.assert_allclose(
        magpy.getB(cells, observers, sumup=True),
        whole.getB(observers),
        rtol=2e-13,
        atol=1e-14,
    )
