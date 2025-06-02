"""
Implementation for the magnetic field of homogeneously
magnetized tetrahedra. Computation details in function docstrings.
"""

from __future__ import annotations

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.fields.field_BH_triangle import BHJM_triangle
from magpylib._src.input_checks import check_field_input


def check_chirality(points: np.ndarray) -> np.ndarray:
    """
    Checks if quadruple of points (p0,p1,p2,p3) that forms tetrahedron is arranged in a way
    that the vectors p0p1, p0p2, p0p3 form a right-handed system

    Parameters
    -----------
    points: 3d-array of shape (m x 4 x 3)
            m...number of tetrahedrons

    Returns
    ----------
    new list of points, where p2 and p3 are possibly exchanged so that all
    tetrahedron is given in a right-handed system.
    """
    xp = array_namespace(points)
    vecs = xp.zeros(((points.shape[0]), 3, 3))
    vecs[:, :, 0] = points[:, 1, :] - points[:, 0, :]
    vecs[:, :, 1] = points[:, 2, :] - points[:, 0, :]
    vecs[:, :, 2] = points[:, 3, :] - points[:, 0, :]

    dets = xp.linalg.det(vecs)
    dets_neg = dets < 0

    if xp.any(dets_neg):
        points[:, 2:, :][dets_neg] = points[:, 3:1:-1, :][dets_neg]

    return points


def point_inside(points: np.ndarray, vertices: np.ndarray, in_out: str) -> np.ndarray:
    """
    Takes points, as well as the vertices of a tetrahedra.
    Returns boolean array indicating whether the points are inside the tetrahedra.
    """
    xp = array_namespace(points, vertices)
    if in_out == "inside":
        return xp.repeat(True, (points.shape[0]))

    if in_out == "outside":
        return xp.repeat(False, (points.shape[0]))

    mat = xp.moveaxis(vertices[:, 1:, ...], (0, 1), (1, 0)) - vertices[:, 0, ...]
    mat = xp.moveaxis(mat, (0, 1, 2), (2, 0, 1))

    mat = xp.astype(mat, xp.float64)
    tetra = xp.linalg.inv(mat)
    newp = xp.matmul(tetra, xp.reshape(points - vertices[:, 0, :], (*points.shape, 1)))
    return xp.reshape(
        xp.all(newp >= 0, axis=1)
        & xp.all(newp <= 1, axis=1)
        & (xp.sum(newp, axis=1) <= 1),
        (-1,),
    )


def BHJM_magnet_tetrahedron(
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

    # allocate - try not to generate more arrays
    BHJM = xp.astype(polarization, (xp.float64))

    if field == "J":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[~mask_inside] = 0
        return BHJM

    if field == "M":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[~mask_inside] = 0
        return BHJM / MU0

    vertices = check_chirality(vertices)

    tri_vertices = xp.concat(
        (
            xp.concat(tuple(vertices[:, i : i + 1, :] for i in (0, 2, 1)), axis=1),
            xp.concat(tuple(vertices[:, i : i + 1, :] for i in (0, 1, 3)), axis=1),
            xp.concat(tuple(vertices[:, i : i + 1, :] for i in (1, 2, 3)), axis=1),
            xp.concat(tuple(vertices[:, i : i + 1, :] for i in (0, 3, 2)), axis=1),
        ),
        axis=0,
    )
    tri_field = BHJM_triangle(
        field=field,
        observers=xp.tile(observers, (4, 1)),
        vertices=tri_vertices,
        polarization=xp.tile(polarization, (4, 1)),
    )
    n = observers.shape[0]
    BHJM = (  # slightly faster than reshape + sum
        tri_field[:n, ...]
        + tri_field[n : 2 * n, ...]
        + tri_field[2 * n : 3 * n, ...]
        + tri_field[3 * n :, ...]
    )

    if field == "H":
        return BHJM

    if field == "B":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[mask_inside] += polarization[mask_inside]
        return BHJM

    msg = f"`output_field_type` must be one of ('B', 'H', 'M', 'J'), got {field!r}"
    raise ValueError(msg)  # pragma: no cover
