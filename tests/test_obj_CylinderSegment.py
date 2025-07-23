import numpy as np

import magpylib as magpy


def test_repr():
    """test __repr__"""
    pm2 = magpy.magnet.CylinderSegment(
        polarization=(1, 2, 3), dimension=(1, 2, 3, 0, 90)
    )
    assert repr(pm2)[:15] == "CylinderSegment", "CylinderSegment repr failed"


def test_barycenter():
    """test if barycenter is computed correctly"""
    cs = magpy.magnet.CylinderSegment(
        polarization=(100, 0, 0), dimension=(1, 2, 1, 85, 170)
    )

    expected_barycenter_squeezed = np.array([-0.86248133, 1.12400755, 0.0])
    np.testing.assert_allclose(cs.barycenter, expected_barycenter_squeezed)

    cs.rotate_from_angax([76 * i for i in range(5)], "x", anchor=(0, 0, 5), start=0)

    expected_barycenter_path = np.array(
        [
            [-0.86248133, 1.12400755, 0.0],
            [-0.86248133, 5.12340067, 4.88101025],
            [-0.86248133, 1.35491805, 9.94242755],
            [-0.86248133, -4.46783198, 7.51035264],
            [-0.86248133, -3.51665082, 1.27219099],
        ]
    )
    np.testing.assert_allclose(cs.barycenter, expected_barycenter_path)


def test_CylinderSegment_volume():
    """Test CylinderSegment volume calculation."""
    segment = magpy.magnet.CylinderSegment(
        dimension=(1.0, 2.0, 3.0, 0, 90), polarization=(0, 0, 1)
    )
    calculated = segment.volume
    expected = (2.0**2 - 1.0**2) * np.pi * 3.0 * 90 / 360  # (r2²-r1²)*π*h*φ/360
    assert abs(calculated - expected) < 1e-10


def test_CylinderSegment_centroid():
    """Test CylinderSegment centroid - should return barycenter if available"""
    cylinder_seg = magpy.magnet.CylinderSegment(
        dimension=(1, 2, 3, -145, 145),  # r1, r2, h, phi1, phi2
        polarization=(0, 0, 1),
        position=(4, 5, 6)
    )
    expected = [4.35255872, 5.0, 6.0]
    assert np.allclose(cylinder_seg.centroid, expected)
