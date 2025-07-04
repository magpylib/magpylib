import numpy as np

import magpylib as magpy


def test_two_ragged_Strips():
    """Tests if codes works for ragged strips"""

    strip1 = magpy.current.TriangleStrip(
    vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2,0,0]],
    current=10.1,
    position=([[1,2,3]]*2)
    )
    strip2 = magpy.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 2, 0], [1, 0, 0], [1, 2, 0]],
        current=10.1,
    )
    H = magpy.getH(
        sources = [strip1, strip2],
        observers = magpy.Sensor(pixel=[(2,3,4), (.1,.2,.3)]),
    )

    Htest = np.array([
        [[[-1.93363166e-17, -6.31250000e-01,  3.01329491e-01],
        [ 2.28321078e-18,  5.49844024e-02, -4.52367753e-02],],
        [[-1.93363166e-17, -6.31250000e-01,  3.01329491e-01],
        [ 2.28321078e-18,  5.49844024e-02, -4.52367753e-02],],],
        [[[ 6.89279913e-19, -3.04597135e-02,  1.45698945e-02],
        [ 2.97911957e-17, -9.07478634e-01, -5.58522415e-01],],
        [[ 6.89279913e-19, -3.04597135e-02,  1.45698945e-02],
        [ 2.97911957e-17, -9.07478634e-01, -5.58522415e-01],],],
    ])
    np.testing.assert_allclose(H, Htest, rtol=1e-8, atol=1e-8)


def test_two_Strips():
    """Test if two TriangleStrips can be added to a Collection."""

    strip1 = magpy.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
        current=10.1,
    )
    strip2 = magpy.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 2, 0], [1, 0, 0], [1, 2, 0]],
        current=10.1,
    )

    col = strip1 + strip2
    assert isinstance(col, magpy.Collection), "adding cuboids fail"

    H = magpy.getH(
        sources = [col, strip1, strip2],
        observers = magpy.Sensor(pixel=[(0,0,0), (1,2,3), (2,3,4), (.1,.2,.3)]),
    )
    print(H.shape)

    Htest = np.array([
        [[-4.21176218e-17, -8.83773949e-17, -4.45975006e-01],
            [-1.35307153e-18, -1.32067982e-01,  5.14696153e-02],
            [ 1.28707461e-18, -5.69332991e-02,  3.09483760e-02],
            [ 4.66523453e-17, -2.61515030e+00, -1.26928234e+00],],
        [[-2.54084028e-17, -6.13413107e-17, -2.93424150e-01],
            [-6.21795244e-19, -6.12093851e-02,  2.99544630e-02],
            [ 5.97794697e-19, -2.64735856e-02,  1.63784815e-02],
            [ 1.68611496e-17, -1.70767166e+00, -7.10759925e-01],],
        [[-1.67092189e-17, -2.70360842e-17, -1.52550855e-01],
            [-7.31276283e-19, -7.08585966e-02,  2.15151524e-02],
            [ 6.89279913e-19, -3.04597135e-02,  1.45698945e-02],
            [ 2.97911957e-17, -9.07478634e-01, -5.58522415e-01],],
    ])

    np.testing.assert_allclose(H, Htest, rtol=1e-6, atol=1e-6)


def test_TriangleStrip_repr():
    """TriangleStrip repr test"""
    ts = magpy.current.TriangleStrip()
    assert repr(ts)[:13] == "TriangleStrip", "TriangleStrip repr failed"
