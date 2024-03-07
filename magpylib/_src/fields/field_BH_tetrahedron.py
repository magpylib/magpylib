"""
Implementation for the magnetic field of homogeneously
magnetized tetrahedra. Computation details in function docstrings.
"""

import numpy as np
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

    vecs = np.zeros((len(points), 3, 3))
    vecs[:, :, 0] = points[:, 1, :] - points[:, 0, :]
    vecs[:, :, 1] = points[:, 2, :] - points[:, 0, :]
    vecs[:, :, 2] = points[:, 3, :] - points[:, 0, :]

    dets = np.linalg.det(vecs)
    dets_neg = dets < 0

    if np.any(dets_neg):
        points[dets_neg, 2:, :] = points[dets_neg, 3:1:-1, :]

    return points


def point_inside(points: np.ndarray, vertices: np.ndarray, in_out: str) -> np.ndarray:
    """
    Takes points, as well as the vertices of a tetrahedra.
    Returns boolean array indicating whether the points are inside the tetrahedra.
    """
    if in_out == "inside":
        return np.array([True] * len(points))

    if in_out == "outside":
        return np.array([False] * len(points))

    mat = vertices[:, 1:].swapaxes(0, 1) - vertices[:, 0]
    mat = np.transpose(mat.swapaxes(0, 1), (0, 2, 1))

    tetra = np.linalg.inv(mat)
    newp = np.matmul(tetra, np.reshape(points - vertices[:, 0, :], (*points.shape, 1)))
    inside = (
        np.all(newp >= 0, axis=1)
        & np.all(newp <= 1, axis=1)
        & (np.sum(newp, axis=1) <= 1)
    ).flatten()

    return inside


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

    # allocate - try not to generate more arrays
    BHJM = polarization.astype(float)

    if field == "J":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[~mask_inside] = 0
        return BHJM

    if field == "M":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[~mask_inside] = 0
        return BHJM / MU0

    vertices = check_chirality(vertices)

    tri_vertices = np.concatenate(
        (
            vertices[:, (0, 2, 1), :],
            vertices[:, (0, 1, 3), :],
            vertices[:, (1, 2, 3), :],
            vertices[:, (0, 3, 2), :],
        ),
        axis=0,
    )
    tri_field = BHJM_triangle(
        field=field,
        observers=np.tile(observers, (4, 1)),
        vertices=tri_vertices,
        polarization=np.tile(polarization, (4, 1)),
    )
    n = len(observers)
    BHJM = (  # slightly faster than reshape + sum
        tri_field[:n]
        + tri_field[n : 2 * n]
        + tri_field[2 * n : 3 * n]
        + tri_field[3 * n :]
    )

    if field == "H":
        return BHJM

    if field == "B":
        mask_inside = point_inside(observers, vertices, in_out)
        BHJM[mask_inside] += polarization[mask_inside]
        return BHJM

    raise ValueError(  # pragma: no cover
        "`output_field_type` must be one of ('B', 'H', 'M', 'J'), " f"got {field!r}"
    )
