"""Charged triangle field implementation."""

# pylance: disable=Code is unreachable

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import (
    array_namespace,
    cross,
    dot,
    jax_padded,
    some,
    xp_promote,
)
from magpylib._src.input_checks import check_field_input


def _norm_vector(v) -> np.ndarray:
    """
    Calculates normalized orthogonal vector on a plane defined by three vertices.
    v: shape (n, 3, 3), returns: shape (n, 3)
    """
    xp = array_namespace(v)
    n = cross(v[:, 1, :] - v[:, 0, :], v[:, 2, :] - v[:, 0, :])
    return n / xp.sqrt(dot(n, n))[:, xp.newaxis]


def _solid_angle(R: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the solid angle of triangles.

    Input:
    R: shape (n, 3, 3), vectors from the observer to the three vertices
    r: shape (n, 3), their lengths

    Returns: shape (n,)
    """
    xp = array_namespace(R, r)
    R0, R1, R2 = R[:, 0, :], R[:, 1, :], R[:, 2, :]
    r0, r1, r2 = r[:, 0], r[:, 1], r[:, 2]

    # Calculates (oriented) volume of the parallelepiped in vectorized form.
    N = dot(R2, cross(R1, R0))

    D = r0 * r1 * r2 + dot(R2, R1) * r0 + dot(R2, R0) * r1 + dot(R1, R0) * r2
    result = 2.0 * xp.atan2(N, D)

    # modulus 2pi to avoid jumps on edges in line
    # "B = sigma * ((n.T * _solid_angle(R, r)) - _vcross3(n, PQR).T)"
    # <-- bad fix :(

    return xp.where(xp.abs(result) > 6.2831853, 0.0, result)


@jax_padded()
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
    xp = array_namespace(observers, vertices, polarizations)
    observers, vertices, polarizations = xp_promote(
        observers, vertices, polarizations, xp=xp
    )

    n = _norm_vector(vertices)
    sigma = dot(n, polarizations)[:, xp.newaxis]

    # vertex <-> observer, shape (i, 3 vertices, 3)
    R = vertices - observers[:, xp.newaxis, :]
    r2 = dot(R, R)
    r = xp.sqrt(r2)

    # vertex <-> vertex
    L = xp.roll(vertices, -1, axis=1) - vertices
    l2 = dot(L, L)
    l1 = xp.sqrt(l2)

    # vert-vert -- vert-obs
    b = dot(R, L)
    bl = b / l1
    ind = xp.abs(r + bl)  # closeness measure to corner and edge

    # The computation of ind is the origin of a major numerical instability
    #    when approaching the triangle because r ~ -bl. This number
    #    becomes small at the same rate as it looses precision.
    #    This is a major problem, because at distances 1e-8 and 1e8 all precision
    #    is already lost !!!
    # The second problem is at corner and edge extensions where ind also computes
    #    as 0. Here one approaches a special case where another evaluation should
    #    be used. This problem is solved in the following lines.
    mask_gen = ind > 1.0e-12

    # both evaluations are computed with harmless arguments where they are
    # not used or singular
    arg_gen = xp.sqrt(xp.where(mask_gen, l2 + 2 * b + r2, 1.0)) + l1 + bl
    arg_gen = arg_gen / xp.where(mask_gen, ind, 1.0)
    integral = xp.log(xp.where(mask_gen, arg_gen, 1.0))
    mask_corner = False
    if some(~mask_gen):
        # the other evaluation is singular at the two ends of an edge: the
        # corners
        mask_end = (r == 0) | (r == l1)
        mask_corner = xp.any(~mask_gen & mask_end, axis=-1, keepdims=True)
        arg_ext = xp.abs(l1 - r) / xp.where(mask_end, 1.0, r)
        integral = xp.where(
            mask_gen, integral, -xp.log(xp.where(mask_end, 1.0, arg_ext))
        )

    PQR = xp.sum((integral / l1)[:, :, xp.newaxis] * L, axis=1)
    B = sigma * (n * _solid_angle(R, r)[:, xp.newaxis] - cross(n, PQR))
    B = B / xp.pi / 4.0

    # return (0, 0, 0) at corners
    return xp.where(mask_corner | xp.isnan(B), 0.0, B)


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

    if field in "MJ":
        xp = array_namespace(observers, vertices, polarization)
        return xp.zeros_like(xp_promote(polarization, xp=xp))

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
