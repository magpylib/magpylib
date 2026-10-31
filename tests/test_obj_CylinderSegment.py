import numpy as np
import pytest

import magpylib as magpy


@pytest.mark.parametrize("turns", [-1, 1, 2])
@pytest.mark.parametrize("angles", [(-350, 9), (40, 130)])
def test_section_angles_wrap(angles, turns):
    """a section gives the same field when its angles are shifted by full turns"""
    obs = np.array(
        [(r * np.cos(a), r * np.sin(a), 0.1) for r in (0.1, 0.4) for a in range(-3, 4)]
    )
    dimension = np.array([0.2, 0.6, 1, *angles], dtype=float)
    ref = magpy.magnet.CylinderSegment(
        polarization=(0.3, -0.5, 0.8), dimension=dimension
    )
    shifted = ref.copy(dimension=dimension + 360 * turns * np.array([0, 0, 0, 1, 1]))
    np.testing.assert_allclose(shifted.getJ(obs), ref.getJ(obs))
    np.testing.assert_allclose(shifted.getB(obs), ref.getB(obs), atol=1e-12)


@pytest.mark.parametrize("angles", [(90, 360), (-360, -90), (0, 270)])
def test_section_face_at_zero_angle_is_a_surface(angles):
    """on a face the field is 0, also where the face angle is +-360"""
    magnet = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 1), dimension=(0, 0.5, 1, *angles)
    )
    np.testing.assert_array_equal(magnet.getB((0.25, 0, 0.2)), 0)


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
