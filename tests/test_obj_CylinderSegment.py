import numpy as np

import magpylib as magpy


def test_repr():
    """test __repr__"""
    pm2 = magpy.magnet.CylinderSegment((1, 2, 3), (1, 2, 3, 0, 90))
    assert repr(pm2)[:15] == "CylinderSegment", "CylinderSegment repr failed"


def test_barycenter():
    """test if barycenter is computed correctly"""
    cs = magpy.magnet.CylinderSegment(
        magnetization=(100, 0, 0), dimension=(1, 2, 1, 85, 170)
    )

    expected_barycenter_squeezed = np.array([-0.86248133, 1.12400755, 0.0])
    np.testing.assert_allclose(cs.barycenter, expected_barycenter_squeezed)

    cs.rotate_from_angax([76 * i for i in range(0, 5)], "x", anchor=(0, 0, 5), start=0)

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
