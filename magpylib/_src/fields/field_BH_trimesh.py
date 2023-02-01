"""
Implementations of analytical expressions for the magnetic field of a triangular surface.
Computation details in function docstrings.
"""
# pylance: disable=Code is unreachable
import numpy as np

from magpylib._src.fields.field_BH_triangle import triangle_field

# helper functions
def signed_volume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron
    defined by the line segment 's' and two vertices of the triangle 't'."""

    return np.sum((a - d) * np.cross(b - d, c - d), axis=2)


def segments_intersect_triangles(segments, triangles, summation=True):
    """For each line segment in `s`, this function computes how many times it intersects
    any of the triangles given in `t`.
    Parameters
    ----------
    segments: 2xSx3 array
        Array of `S` line segments where the first index specifies the start or end point of the
        segment, the second index refers to the S^th line segment, and the third index points to
        the x, y, z coordinates of the line segment point.

    triangles: 3xTx3
        Array of `T` triangles, where the first index specifies one of the three vertices
        (which don't have to be in any particular order), the second index refers to the T^th
        triangle, and the third index points to the x,y,z coordinates the the triangle vertex.

    summation: bool, optional
        If `True`, a binary array is return which only tells if the S^th line segment intersects
        any of  the triangles given. If `False`, returns a SxT array that tells which triangles are
        intersected.

    Returns
    -------
        If `'summation'` is `True`, returns a binary array of size S which tells whether the S^th
        line segment intersects any of the triangles given. Otherwise returns a SxT array that tells
        which triangles are intersected.
    """

    s, t = segments, triangles
    # compute the normals to each triangle
    normals = np.cross(t[2] - t[0], t[2] - t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the
    # plane.
    s0, s1 = s[0][:, np.newaxis], s[1][:, np.newaxis]  # -> S x T x 3 arrays
    sign1 = np.sign(np.sum(normals * (s0 - t[2]), axis=2))  # S x T
    sign2 = np.sign(np.sum(normals * (s1 - t[2]), axis=2))  # S x T

    # determine segments which cross the plane of a triangle.
    #  -> 1 if the sign of the end points of s is
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2) * (sign1 != 0) * (sign2 != 0)  # S x T

    # get signed volumes
    v = [
        np.sign(signed_volume(t[i], t[j], s0, s1)) for i, j in zip((0, 1, 2), (1, 2, 0))
    ]  # S x T
    same_volume = np.logical_and(
        (v[0] == v[1]), (v[1] == v[2])
    )  # 1 if s and t have same sign in v0, v1 and v2

    res = cross * same_volume
    if summation:
        res = np.sum(res, axis=1)

    return res


# masks
def mask_inside_enclosing_box(points, vertices, tol=1e-15):
    """Return a mask for `points` which truth value tells if inside the
    bounding box"""
    xmin, ymin, zmin = np.min(vertices, axis=0)
    xmax, ymax, zmax = np.max(vertices, axis=0)
    x, y, z = points.T
    # within cuboid dimension with positive tolerance
    mx = (x - xmax < tol) & (xmin - x < tol)
    my = (y - ymax < tol) & (ymin - y < tol)
    mz = (z - zmax < tol) & (zmin - z < tol)
    return mx & my & mz


def mask_inside_trimesh(points, facets):
    """Return a boolean mask corresponding to the truth values of which points are inside
    the triangular mesh defined by the provided facets, using the ray tracing method. A pre-filter
    is used to check if points are inside an enclosing box and only if the ones inside are further
    then checked upon."""
    # compute only points outside enclosing box more efficiently
    vertices = np.unique(facets.reshape((-1, 3)), axis=0)
    mask = mask_inside_enclosing_box(points, vertices)

    # choose a start point that is for sure outside the mesh
    pts_ins_box = points[mask]
    start_point = np.min(vertices, axis=0) - np.array([1.001, 0.992, 0.993])
    segments = np.tile(start_point, (pts_ins_box.shape[0], 1))
    t = facets.swapaxes(0, 1)
    s = np.concatenate([segments, pts_ins_box]).reshape((2, -1, 3))
    sums = segments_intersect_triangles(s, t)
    ray_tracing_mask = sums % 2 != 0
    mask[mask] = ray_tracing_mask
    return mask


def magnet_trimesh_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    facets: np.ndarray,
    in_out="auto",
) -> np.ndarray:
    """
    Code for the field calculation of a uniformly magnetized triangular facet body.

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
        `facets` can be a ragged sequence of facet children with different lengths.

    in_out: {'auto', 'inside', 'outside'}
        Tells if the points are inside or outside the enclosing facets for the correct B/H-field
        calculation. By default `in_out='auto'` and the inside/outside mask is automatically
        generated using a ray tracing algorigthm to determine which observers are inside and which
        are outside the closed body. For performance reasons, one can define `in_out='outside'`
        or `in_out='inside'` if it is known in advance that all observers satisfy the same
        condition.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

    Notes
    -----
    Field computations via publication:
    Guptasarma: GEOPHYSICS 1999 64:1, 70-74
    """

    if facets.ndim != 1:  # all vertices objects have same number of children
        n0, n1, *_ = facets.shape
        vertices_tiled = facets.reshape(-1, 3, 3)
        observers_tiled = np.repeat(observers, n1, axis=0)
        magnetization_tiled = np.repeat(magnetization, n1, axis=0)
        B = triangle_field(
            field="B",
            observers=observers_tiled,
            magnetization=magnetization_tiled,
            vertices=vertices_tiled,
        )
        B = B.reshape((n0, n1, 3))
        B = np.sum(B, axis=1)
    else:
        nvs = [f.shape[0] for f in facets]  # length of vertex set
        split_indices = np.cumsum(nvs)[:-1]  # remove last to avoid empty split
        vertices_tiled = np.concatenate([f.reshape((-1, 3, 3)) for f in facets])
        observers_tiled = np.repeat(observers, nvs, axis=0)
        magnetization_tiled = np.repeat(magnetization, nvs, axis=0)
        B = triangle_field(
            field="B",
            observers=observers_tiled,
            magnetization=magnetization_tiled,
            vertices=vertices_tiled,
        )
        b_split = np.split(B, split_indices)
        B = np.array([np.sum(bh, axis=0) for bh in b_split])

    if field == "B":
        if in_out == "auto":
            prev_ind = 0
            # group similar facets
            for new_ind, _ in enumerate(B):
                if (
                    new_ind == len(B) - 1
                    or facets[new_ind].shape != facets[prev_ind].shape
                    or not np.all(facets[new_ind] == facets[prev_ind])
                ):
                    if new_ind == len(B) - 1:
                        new_ind = len(B)
                    inside_mask = mask_inside_trimesh(
                        observers[prev_ind:new_ind], facets[prev_ind]
                    )
                    # if inside magnet add magnetization vector
                    B[prev_ind:new_ind][inside_mask] += magnetization[prev_ind:new_ind][
                        inside_mask
                    ]
                    prev_ind = new_ind
        elif in_out == "inside":
            B += magnetization
        return B

    H = B * 10 / 4 / np.pi  # mT -> kA/m
    return H
