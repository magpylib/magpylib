import numpy as np
import pytest

import magpylib as magpy
from magpylib.func import cylinder_segment_field


@pytest.mark.parametrize("turns", [-1, 1, 2])
@pytest.mark.parametrize("angles", [(-350, -170, 9), (40, 85, 130)])
def test_section_angles_wrap(angles, turns):
    """a section shifted by full turns has the field of its two halves"""
    phi1, phi_mid, phi2 = angles
    obs = np.array(
        [(r * np.cos(a), r * np.sin(a), 0.1) for r in (0.1, 0.4) for a in range(-3, 4)]
    )
    pol = (0.3, -0.5, 0.8)
    halves = [
        magpy.magnet.CylinderSegment(polarization=pol, dimension=(0.2, 0.6, 1, *ang))
        for ang in ((phi1, phi_mid), (phi_mid, phi2))
    ]
    shift = 360 * turns
    magnet = magpy.magnet.CylinderSegment(
        polarization=pol, dimension=(0.2, 0.6, 1, phi1 + shift, phi2 + shift)
    )
    np.testing.assert_allclose(
        magnet.getB(obs), sum(half.getB(obs) for half in halves), atol=1e-9
    )
    np.testing.assert_array_equal(
        magnet.getJ(obs), sum(half.getJ(obs) for half in halves)
    )


def test_section_angles_wrap_in_a_batch():
    """rows of one call follow their own angles"""
    obs = np.array([[0.25, 0.01, 0.2]] * 3)
    dims = np.array([(0, 0.5, 1, 10, 369), (0, 0.5, 1, -350, 9), (0, 0.5, 1, 40, 130)])
    B = cylinder_segment_field(
        "B", observers=obs, dimensions=dims, polarizations=(0, 0, 1)
    )
    for row, dim in zip(B, dims, strict=True):
        single = cylinder_segment_field(
            "B", observers=obs[0], dimensions=dim, polarizations=(0, 0, 1)
        )
        np.testing.assert_allclose(row, single, atol=1e-12)


@pytest.mark.parametrize(
    ("dimension", "observer"),
    [
        ((0, 0.5, 1, 90, 360), (0.25, 0, 0.2)),  # face at 360
        ((0, 0.5, 1, -360, -90), (0.25, -0.0, 0.2)),  # face at -360
        (
            (0.2, 0.6, 1, 370, 400),
            (0.4 * np.cos(np.deg2rad(10)), 0.4 * np.sin(np.deg2rad(10)), 0.2),
        ),  # face past 360
        ((0, 0.5, 1, 10, 90), (0, 0, 0.5)),  # axis on the top, NaN before
        ((0, 0.5, 1, 90, 360), (1e-17, 1e-17, 0.2)),  # axis within rounding
    ],
)
def test_section_surfaces_give_zero(dimension, observer):
    """faces and the axis of a section with r1=0 are surfaces, where B and H are 0"""
    magnet = magpy.magnet.CylinderSegment(polarization=(0, 0, 1), dimension=dimension)
    np.testing.assert_array_equal(magnet.getB(observer), 0)
    np.testing.assert_array_equal(magnet.getH(observer), 0)


def test_section_J_on_a_surface_ignores_other_observers():
    """J on a face is the same alone and next to points off the surface"""
    magnet = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 1), dimension=(0, 0.5, 1, 0, 270)
    )
    face = (0.25, 0, 0.2)
    np.testing.assert_array_equal(magnet.getJ(face), magnet.getJ([face, (9, 9, 9)])[0])


def test_section_full_turn_with_rounding_is_accepted():
    """phi2 - phi1 rounds to just over 360 for this full turn"""
    phi1 = 152.2
    assert (phi1 + 360) - phi1 > 360
    magnet = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 1), dimension=(0.2, 0.6, 1, phi1, phi1 + 360)
    )
    np.testing.assert_array_equal(magnet.getJ((0.4, 0.1, 0)), (0, 0, 1))


def test_section_span_over_a_turn_raises():
    """magpy.func does not validate dimension up front"""
    with pytest.raises(ValueError, match="phi2 - phi1 <= 360"):
        cylinder_segment_field(
            "B",
            observers=(0, 0, 0),
            dimensions=(0.2, 0.6, 1, -100, 270),
            polarizations=(0, 0, 1),
        )


def test_repr():
    """test __repr__"""
    pm2 = magpy.magnet.CylinderSegment(
        polarization=(1, 2, 3), dimension=(1, 2, 3, 0, 90)
    )
    assert repr(pm2)[:15] == "CylinderSegment", "CylinderSegment repr failed"


def test_centroid():
    """test if centroid is computed correctly"""
    cs = magpy.magnet.CylinderSegment(
        polarization=(100, 0, 0), dimension=(1, 2, 1, 85, 170)
    )

    expected_centroid_squeezed = np.array([-0.86248133, 1.12400755, 0.0])
    np.testing.assert_allclose(cs.centroid, expected_centroid_squeezed)

    cs.rotate_from_angax([76 * i for i in range(5)], "x", anchor=(0, 0, 5), start=0)

    expected_centroid_path = np.array(
        [
            [-0.86248133, 1.12400755, 0.0],
            [-0.86248133, 5.12340067, 4.88101025],
            [-0.86248133, 1.35491805, 9.94242755],
            [-0.86248133, -4.46783198, 7.51035264],
            [-0.86248133, -3.51665082, 1.27219099],
        ]
    )
    np.testing.assert_allclose(cs.centroid, expected_centroid_path)


def test_CylinderSegment_volume():
    """Test CylinderSegment volume calculation."""
    segment = magpy.magnet.CylinderSegment(
        dimension=(1.0, 2.0, 3.0, 0, 90), polarization=(0, 0, 1)
    )
    calculated = segment.volume
    expected = (2.0**2 - 1.0**2) * np.pi * 3.0 * 90 / 360  # (r2²-r1²)*π*h*φ/360
    assert abs(calculated - expected) < 1e-10


def test_CylinderSegment_centroid():
    """Test CylinderSegment centroid"""
    cylinder_seg = magpy.magnet.CylinderSegment(
        dimension=(1, 2, 3, -145, 145),  # r1, r2, h, phi1, phi2
        polarization=(0, 0, 1),
        position=(4, 5, 6),
    )
    expected = [4.35255872, 5.0, 6.0]
    assert np.allclose(cylinder_seg.centroid, expected)
