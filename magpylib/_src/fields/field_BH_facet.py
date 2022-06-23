"""
Implementations of analytical expressions for the magnetic field of a triangular facet.
Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input

#############
# constants
#############
EPS = 1.0e-12
#############

#############
# help functions
#############
def norm_vector(v):
    """
    Calculates normalized orthogonal vector on a plane defined by three vertices.
    """
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n, axis=-1)
    return n / np.expand_dims(n_norm, axis=-1)


def mydot(a, b):
    """
    Vectorized 3d dot-product.
    """
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def scalar3(R1, R2, R3):
    """
    Calculates (oriented) volume of the parallelepiped defined by the vectors R1, R2, R3
    """
    return mydot(R1, np.cross(R2, R3))


def solid_angle(R1, R2, R3, r1, r2, r3):
    """
    Calculates the solid angle of the triangle defined by the position vectors R1, R2, R3
    """
    N = scalar3(R1, R2, R3)
    D = r1 * r2 * r3 + mydot(R1, R2) * r3 + mydot(R1, R3) * r2 + mydot(R2, R3) * r1
    return 2.0 * np.arctan2(N, D)


def next_i(i):
    """
    Returns next index modulo 3
    """
    if i == 2:
        return 0
    else:
        return i + 1


#############


def facet_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Code for the field calculation of a uniformly magnetized triangular facet

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of [mT], if `field='H'` return H-field
        in units of [kA/m].

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of [mm].

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of [mT].

    vertices: ndarray, shape (n,3,3)
        Vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) of triangular facet

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

    Notes
    -----
    Field computations via publication:
    Guptasarma: GEOPHYSICS 1999 64:1, 70-74
    """
    # pylint: disable=too-many-statements

    bh = check_field_input(field, "facet_field()")

    num_targets = observers.shape[0]

    B = np.zeros((num_targets, 3))

    n = norm_vector(vertices)
    sigma = mydot(n, magnetization)
    R = []
    r = []
    for i in range(3):
        R.append(vertices[:, i] - observers)
        r.append(np.linalg.norm(R[i], axis=1))
    inout = np.fabs(n[:, 0] * R[i][:, 0] + n[:, 1] * R[i][:, 1] + n[:, 2] * R[i][:, 2])
    solid_angle_results = np.where(
        inout <= EPS, 0.0, solid_angle(R[2], R[1], R[0], r[2], r[1], r[0])
    )
    PP = np.zeros(num_targets)
    QQ = np.zeros(num_targets)
    RR = np.zeros(num_targets)
    for i in range(3):
        ii = next_i(i)
        L = vertices[:, ii] - vertices[:, i]
        b = 2.0 * (R[i][:, 0] * L[:, 0] + R[i][:, 1] * L[:, 1] + R[i][:, 2] * L[:, 2])
        l = np.linalg.norm(L, axis=-1)
        bl = b / (2.0 * l)
        ind = np.fabs(r[i] + bl)
        I = np.where(
            ind > EPS,
            (1.0 / l)
            * np.log(
                (np.sqrt(l * l + b + r[i] * r[i]) + l + bl) / (np.fabs(r[i] + bl))
            ),
            -(1.0 / l) * np.log(np.fabs(l - r[i]) / r[i]),
        )
        PP += I * L[:, 0]
        QQ += I * L[:, 1]
        RR += I * L[:, 2]
    B[:, 0] += sigma * (n[:, 0] * solid_angle_results + n[:, 2] * QQ - n[:, 1] * RR)
    B[:, 1] += sigma * (n[:, 1] * solid_angle_results + n[:, 0] * RR - n[:, 2] * PP)
    B[:, 2] += sigma * (n[:, 2] * solid_angle_results + n[:, 1] * PP - n[:, 0] * QQ)

    B /= np.pi * 4

    # return B or compute and return H -------------
    if bh:
        return B

    H = B * 10 / 4 / np.pi  # mT -> kA/m
    return H
