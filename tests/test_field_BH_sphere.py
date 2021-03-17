from magpylib._lib.fields.field_BH_sphere import field_BH_sphere
import numpy as np


def test_field_sphere_vs_v2():
    """ testing against old version
    """
    result_v2 = np.array([
        [22., 44., 66.],
        [22., 44., 66.],
        [38.47035383, 30.77628307, 23.0822123 ],
        [0.60933932, 0.43524237, 1.04458169],
        [22., 44., 66.],
        [-0.09071337, -0.18142674, -0.02093385],
        [-0.17444878, -0.0139559,  -0.10466927],
        ])

    dim = np.array([1.23]*7)
    mag = np.array([(33,66,99)]*7)
    poso = np.array([(0,0,0),(.2,.2,.2),(.4,.4,.4),(-1,-1,-2),(.1,.1,.1),(1,2,-3),(-3,2,1)])
    B = field_BH_sphere(True, mag, dim, poso )

    assert np.allclose(result_v2, B), 'vs_v2 failed'


def test_field_BH_sphere_mag0():
    """ test box field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n)
    pos = np.random.rand(n,3)
    B = field_BH_sphere(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Box mag=0 case broken'
