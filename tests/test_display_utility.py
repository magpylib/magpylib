import numpy as np
from magpylib._src.display.display_utility import draw_arrow_from_vertices

def test_draw_arrow_from_vertices():
    """tests also the edge case when a vertex is in -y direction"""
    vertices = np.array([
        [-1.,  1.,  1.],
        [-1., -1.,  1.],
        [-1., -1., -1.],
        [-1.,  1., -1.],
        [-1.,  1.,  1.]
    ])
    result = draw_arrow_from_vertices(vertices, current=1, arrow_size=1)
    expected = np.array([
        [-1.  , -1.  , -1.12, -1.  , -0.88, -1.  , -1.  , -1.  , -1.  ,
        -1.12, -1.  , -0.88, -1.  , -1.  , -1.  , -1.  , -1.12, -1.  ,
        -0.88, -1.  , -1.  , -1.  , -1.  , -1.12, -1.  , -0.88, -1.  ,
        -1.  ],
       [-1.  ,  0.  ,  0.2 ,  0.  ,  0.2 ,  0.  ,  1.  , -1.  , -1.  ,
        -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,  0.  , -0.2 ,  0.  ,
        -0.2 ,  0.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
         1.  ],
       [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  0.  ,
         0.2 ,  0.  ,  0.2 ,  0.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
        -1.  , -1.  , -1.  , -1.  ,  0.  , -0.2 ,  0.  , -0.2 ,  0.  ,
         1.  ]
    ])
    assert np.allclose(result, expected), 'draw arrow from vertices failed'
