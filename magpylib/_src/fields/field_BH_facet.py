"""
Implementations of analytical expressions for the magnetic field of a triangular facet.
Computation details in function docstrings.
"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

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


def mask_inside_enclosing_box(points, vertices, tol=1e-15):
    """Return a mask for `points` which truth value tells if inside the
    bounding box"""
    xmin, ymin, zmin = np.min(vertices, axis=0)
    xmax, ymax, zmax = np.max(vertices, axis=0)
    x, y, z = points.T
    # within cuboid dimension with positive tolerance
    mx = (x - xmax < tol) & (xmin - x < tol)
    my = (y - ymax < tol) & (ymin - y < tol)
    mz = (x - zmax < tol) & (zmin - z < tol)
    return mx & my & mz


def mask_inside_facets_convexhull(points, vertices):
    """Return a mask for `points` which truth value tells if inside the
    convexhulls build from provided `vertices`."""
    # check first if points are in enclosing box, to save costly convexhull computation
    inside_enclosing_box = mask_inside_enclosing_box(points, vertices)
    hull = ConvexHull(vertices, incremental=True)
    simplices_init = hull.simplices
    for ind, pt in enumerate(points):
        if inside_enclosing_box[ind]:
            hull.add_points([pt])
            if hull.simplices.shape != simplices_init.shape or not np.all(
                hull.simplices == simplices_init
            ):
                inside_enclosing_box[ind] = False
                hull.close()
                hull = ConvexHull(vertices, incremental=True)
    hull.close()
    return inside_enclosing_box


#############


def facet_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    facets: np.ndarray,
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

    facets: ndarray, shape (n,3,3)
        Triangular facets of shape [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].

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

    n = norm_vector(facets)
    sigma = mydot(n, magnetization)
    R = []
    r = []
    for i in range(3):
        R.append(facets[:, i] - observers)
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
        L = facets[:, ii] - facets[:, i]
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


def magnet_facets_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    facets: np.ndarray,
    in_out="outside",
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

    facets: ndarray, shape (n,n1,3,3) or ragged sequence
        Triangular facets of shape [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
        `facets` can be a ragged sequence of facet children with different lengths

    in_out: {'auto', 'inside', 'outside'}
        Defines if the points are inside or outside the enclosing facets for the correct B/H-field
        calculation. By default `in_out='auto'` and the inside/outside mask is generated
        automatically using a convex hull algorithm over the vertices to determine which observers
        are inside and which are outside. For performance reasons, one can define `in_out='outside'`
        or `in_out='inside'` if it is known that all observers satisfy the same condition.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

    Notes
    -----
    Field computations via publication:
    Guptasarma: GEOPHYSICS 1999 64:1, 70-74
    """

    nvs = [f.shape[0] for f in facets]  # length of vertex set
    if facets.ndim != 1:  # all facets objects have same number of children
        n0, n1, *_ = facets.shape
        facets_tiled = facets.reshape(-1, 3, 3)
        observers_tiled = np.repeat(observers, n1, axis=0)
        magnetization_tiled = np.repeat(magnetization, n1, axis=0)
        B = facet_field(
            field="B",
            observers=observers_tiled,
            magnetization=magnetization_tiled,
            facets=facets_tiled,
        )
        B = B.reshape((n0, n1, 3))
        B = np.sum(B, axis=1)
    else:
        split_indices = np.cumsum(nvs)[:-1]  # remove last to avoid empty split
        facets_tiled = np.concatenate([f.reshape((-1, 3, 3)) for f in facets])
        observers_tiled = np.repeat(observers, nvs, axis=0)
        magnetization_tiled = np.repeat(magnetization, nvs, axis=0)
        B = facet_field(
            field="B",
            observers=observers_tiled,
            magnetization=magnetization_tiled,
            facets=facets_tiled,
        )
        b_split = np.split(B, split_indices)
        B = np.array([np.sum(bh, axis=0) for bh in b_split])

    if field == "B":
        if in_out=='auto':
            prev_ind = 0
            # group similar facets
            for new_ind, _ in enumerate(B):
                if facets[new_ind].shape != facets[0].shape or not np.all(
                    facets[new_ind] == facets[0]
                ):
                    sub_facets = np.concatenate(facets[prev_ind:new_ind])
                    vertices = np.unique(sub_facets.reshape(-1, 3), axis=0)
                    inside_mask = mask_inside_facets_convexhull(
                        observers[prev_ind:new_ind], vertices
                    )
                    # if inside magnet add magnetization vector
                    B[prev_ind:new_ind][inside_mask] += magnetization[prev_ind:new_ind][
                        inside_mask
                    ]
                    prev_ind = new_ind
        elif in_out=='inside':
            B += magnetization
        return B

    H = B * 10 / 4 / np.pi  # mT -> kA/m
    return H
