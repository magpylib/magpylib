"""Charged triangle field implementation."""

# pylance: disable=Code is unreachable

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.input_checks import check_field_input


def _vcross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    vectorized cross product for 3d vectors. Is ~4x faster than np.cross when
    arrays are smallish. Only slightly faster for large arrays.
    input shape a, b: (n, 3)
    returns: (n, 3)
    """
    # receives nan values at corners
    with np.errstate(invalid="ignore"):
        result = np.array(
            [
                a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
            ]
        )
    return result.T


def _norm_vector(v) -> np.ndarray:
    """
    Calculates normalized orthogonal vector on a plane defined by three vertices.
    """
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    n = _vcross3(a, b)
    n_norm = np.linalg.norm(n, axis=-1)
    return n / np.expand_dims(n_norm, axis=-1)


def _solid_angle(R: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the solid angle of triangles.

    Triangle point indices are 1, 2, 3, different triangles are denoted by a, b, c, ...
    The first triangle is defined as R1a, R2a, R3a.

    Input:
    R = [(R1a, R1b, R1c, ...), (R2a, R2b, R2c, ...), (R3a, R3b, R3c, ...)]
    r = [(|R1a|, |R1b|, |R1c|, ...), (|R2a|, |R2b|, |R2c|, ...), (|R3a|, |R3b|, |R3c|, ...)]

    Returns:
    [sangle_a, sangle_b, sangle_c, ...]
    """

    # Calculates (oriented) volume of the parallelepiped in vectorized form.
    N = np.einsum("ij, ij->i", R[2], _vcross3(R[1], R[0]))

    D = (
        r[0] * r[1] * r[2]
        + np.einsum("ij, ij->i", R[2], R[1]) * r[0]
        + np.einsum("ij, ij->i", R[2], R[0]) * r[1]
        + np.einsum("ij, ij->i", R[1], R[0]) * r[2]
    )
    result = 2.0 * np.arctan2(N, D)

    # modulus 2pi to avoid jumps on edges in line
    # "B = sigma * ((n.T * _solid_angle(R, r)) - _vcross3(n, PQR).T)"
    # <-- bad fix :(

    return np.where(abs(result) > 6.2831853, 0, result)


def triangle_Bfield(
    observers: np.ndarray,
    vertices: np.ndarray,
    polarizations: np.ndarray,
) -> np.ndarray:
    """B-field of i magnetically charged triangular surfaces.

    The charge is proportional to the projection of the polarization vectors onto the
    triangle surfaces. The order of the triangle vertices defines the sign of the
    surface normal vector (right-hand-rule). The output is proportional to the
    polarization magnitude, and independent of the length units chosen for observers
    and vertices.

    This function can be used to compute fields of homogeneously magnetized bodies
    represented by triangular surface meshes. In that case each triangle must be
    oriented such that its surface normal points outwards.

    Parameters
    ----------
    observers : ndarray, shape (i, 3)
        Observer positions (x, y, z) in Cartesian coordinates.
    vertices : ndarray, shape (i, 3, 3)
        Triangle vertex positions ((P11, P12, P13), (P21, P22, P23), ...) in
        Cartesian coordinates.
    polarizations : ndarray, shape (i, 3)
        Magnetic polarization vectors.

    Returns
    -------
    ndarray, shape (i, 3)
        B-field in units of ``polarization`` input at the observer positions.

    Notes
    -----
    Returns (0, 0, 0) at corners.

    Field computation follows Guptasarma, Geophysics, 1999, 64(1), 70-74.
    Loss of precision occurs when approaching a triangle as
    (x - edge)**2 and with distance from the triangle as (distance**3).

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.triangle_Bfield(
    ...    observers=np.array([(2.0, 1.0, 1.0), (2.0, 2.0, 2.0)]),
    ...    vertices=np.array([
    ...        [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
    ...        [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
    ...    ]),
    ...    polarizations=np.array([(1.0, 1.0, 1.0), (1.0, 1.0, 0.0)]) * 1e3,
    ... )
    >>> with np.printoptions(precision=3):
    ...    print(B)
    [[7.452 4.62  3.136]
     [2.213 2.677 2.213]]
    """
    n = _norm_vector(vertices)
    sigma = np.einsum("ij, ij->i", n, polarizations)  # vectorized inner product

    # vertex <-> observer
    R = np.swapaxes(vertices, 0, 1) - observers
    r2 = np.sum(R * R, axis=-1)
    r = np.sqrt(r2)

    # vertex <-> vertex
    L = vertices[:, (1, 2, 0)] - vertices[:, (0, 1, 2)]
    L = np.swapaxes(L, 0, 1)
    l2 = np.sum(L * L, axis=-1)
    l1 = np.sqrt(l2)

    # vert-vert -- vert-obs
    b = np.einsum("ijk, ijk->ij", R, L)
    bl = b / l1
    ind = np.fabs(r + bl)  # closeness measure to corner and edge

    # The computation of ind is the origin of a major numerical instability
    #    when approaching the triangle because r ~ -bl. This number
    #    becomes small at the same rate as it looses precision.
    #    This is a major problem, because at distances 1e-8 and 1e8 all precision
    #    is already lost !!!
    # The second problem is at corner and edge extensions where ind also computes
    #    as 0. Here one approaches a special case where another evaluation should
    #    be used. This problem is solved in the following lines.
    # np.seterr must be used because of a NumPy bug. It does not interpret where
    #   correctly. The following code will raise a NumPy warning - but obviously shouldn't
    #
    # x = np.array([(0, 1, 2), (0, 0, 1)])
    # np.where(
    #     x>0,
    #     1/x,
    #     0
    # )

    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(  # noqa: E741
            ind > 1.0e-12,
            1.0 / l1 * np.log((np.sqrt(l2 + 2 * b + r2) + l1 + bl) / ind),
            -(1.0 / l1) * np.log(np.fabs(l1 - r) / r),
        )
    PQR = np.einsum("ij, ijk -> jk", I, L)
    B = sigma * (n.T * _solid_angle(R, r) - _vcross3(n, PQR).T)
    B = B / np.pi / 4.0

    B[np.isnan(B)] = 0.0  # set corners nan to 0

    return B.T


def _BHJM_triangle(
    field: str,
    observers: np.ndarray,
    vertices: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - translate triangle core field to BHJM
    """
    check_field_input(field)

    BHJM = polarization.astype(float) * 0.0

    if field == "M":
        return BHJM

    if field == "J":
        return BHJM

    BHJM = triangle_Bfield(
        observers=observers,
        vertices=vertices,
        polarizations=polarization,
    )

    # new MU0 problem:
    #   input is polarization -> output has MU0 on it and must be B
    #   H will then be connected via MU0

    if field == "B":
        return BHJM

    if field == "H":
        return BHJM / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover
