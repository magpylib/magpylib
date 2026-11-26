"""Homogeneously magnetized tetrahedron field implementation."""

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import array_namespace, xp_promote
from magpylib._src.fields.field_BH_triangle import _BHJM_triangle
from magpylib._src.input_checks import check_field_input


def _check_chirality(points: np.ndarray) -> np.ndarray:
    """
    Checks if quadruple of points (p0, p1, p2, p3) that forms tetrahedron is arranged in a way
    that the vectors p0p1, p0p2, p0p3 form a right-handed system

    Parameters
    -----------
    points: 3d-array of shape (m, 4, 3)
            m...number of tetrahedrons

    Returns
    ----------
    new array of points, where p2 and p3 are possibly exchanged so that all
    tetrahedron is given in a right-handed system.
    """

    xp = array_namespace(points)

    # the determinant of the three edge vectors from p0 is negative for a
    # left-handed system
    dets = xp.linalg.det(points[:, 1:, :] - points[:, :1, :])
    swapped = xp.stack(
        (points[:, 0, :], points[:, 1, :], points[:, 3, :], points[:, 2, :]), axis=1
    )
    return xp.where((dets < 0)[:, xp.newaxis, xp.newaxis], swapped, points)


def _point_inside(points: np.ndarray, vertices: np.ndarray, in_out: str) -> np.ndarray:
    """
    Takes points, as well as the vertices of a tetrahedra.
    Returns boolean array indicating whether the points are inside the tetrahedra.
    """
    xp = array_namespace(points, vertices)
    n = points.shape[0]

    if in_out == "inside":
        return xp.ones(n, dtype=xp.bool)

    if in_out == "outside":
        return xp.zeros(n, dtype=xp.bool)

    # coordinates of the points in the basis of the three edges from p0
    mat = xp.matrix_transpose(vertices[:, 1:, :] - vertices[:, :1, :])
    tetra = xp.linalg.inv(mat)
    newp = xp.matmul(tetra, (points - vertices[:, 0, :])[:, :, xp.newaxis])[:, :, 0]
    return (
        xp.all(newp >= 0, axis=1)
        & xp.all(newp <= 1, axis=1)
        & (xp.sum(newp, axis=1) <= 1)
    )


def _BHJM_magnet_tetrahedron(
    field: str,
    observers: np.ndarray,
    vertices: np.ndarray,
    polarization: np.ndarray,
    in_out="auto",
) -> np.ndarray:
    """
    - compute tetrahedron field from Triangle field
    - translate to BHJM
    - treat special cases
    """

    check_field_input(field)

    xp = array_namespace(observers, vertices, polarization)
    observers, vertices, polarization = xp_promote(
        observers, vertices, polarization, xp=xp
    )

    if field in "JM":
        mask_inside = _point_inside(observers, vertices, in_out)[:, xp.newaxis]
        BHJM = xp.where(mask_inside, polarization, 0.0)
        return BHJM if field == "J" else BHJM / MU0

    vertices = _check_chirality(vertices)

    # the four faces, with their normal vectors pointing outwards
    faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2))
    tri_vertices = xp.concat(
        [xp.stack([vertices[:, i, :] for i in face], axis=1) for face in faces],
        axis=0,
    )
    tri_field = _BHJM_triangle(
        field=field,
        observers=xp.tile(observers, (4, 1)),
        vertices=tri_vertices,
        polarization=xp.tile(polarization, (4, 1)),
    )
    n = observers.shape[0]
    BHJM = (
        tri_field[:n, :]
        + tri_field[n : 2 * n, :]
        + tri_field[2 * n : 3 * n, :]
        + tri_field[3 * n :, :]
    )

    if field == "H":
        return BHJM

    if field == "B":
        mask_inside = _point_inside(observers, vertices, in_out)[:, xp.newaxis]
        return xp.where(mask_inside, BHJM + polarization, BHJM)

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover
