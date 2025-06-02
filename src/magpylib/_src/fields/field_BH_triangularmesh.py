"""
Implementations of analytical expressions for the magnetic field of a triangular surface.
Computation details in function docstrings.
"""

# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-branches
# pylance: disable=Code is unreachable
from __future__ import annotations

import numpy as np
import scipy.spatial
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.fields.field_BH_triangle import BHJM_triangle


def calculate_centroid(vertices, faces):
    """
    Calculates the centroid of a 3D triangular surface mesh.

    Parameters:
    vertices (numpy.array): an n x 3 array of vertices
    faces (numpy.array): an m x 3 array of triangle indices

    Returns:
    numpy.array: The centroid of the mesh
    """
    xp = array_namespace(vertices, faces)

    # Calculate the centroids of each triangle
    triangle_centroids = xp.mean(vertices[faces], axis=1)

    # Compute the area of each triangle
    triangle_areas = 0.5 * xp.linalg.norm(
        xp.linalg.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )

    # Calculate the centroid of the entire mesh
    return xp.sum(triangle_centroids.T * triangle_areas, axis=1) / xp.sum(
        triangle_areas
    )


def v_norm2(a: np.ndarray) -> np.ndarray:
    """
    return |a|**2
    """
    a = a * a
    return a[..., 0] + a[..., 1] + a[..., 2]


def v_norm_proj(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    return a_dot_b/|a||b|
    assuming that |a|, |b| > 0
    """
    xp = array_namespace(a, b)
    ab = a * b
    ab = ab[..., 0] + ab[..., 1] + ab[..., 2]

    return ab / xp.sqrt(v_norm2(a) * v_norm2(b))


def v_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a x b
    """
    xp = array_namespace(a, b)
    return xp.asarray(
        (
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        )
    ).T


def v_dot_cross3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    a x b * c
    """
    return (
        (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0]
        + (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1]
        + (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2]
    )


def get_disconnected_faces_subsets(faces: list) -> list:
    """Return a list of disconnected faces sets"""
    subsets_inds = []
    tria_temp = faces.copy()
    while len(tria_temp) > 0:
        first, *rest = tria_temp
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2
        subsets_inds.append(list(first))
        tria_temp = rest
    return [faces[np.isin(faces, list(ps)).all(axis=1)] for ps in subsets_inds]


def get_open_edges(faces: np.ndarray) -> bool:
    """
    Check if given trimesh forms a closed surface.

    Input: faces: np.ndarray, shape (n,3), dtype int
        triples of indices

    Output: open edges
    """
    xp = array_namespace(faces)
    edges = xp.concat([faces[:, 0:2], faces[:, 1:3], faces[:, ::2]], axis=0)

    # unique edge pairs and counts how many
    edges = np.sort(edges, axis=1)
    edges_uniq, edge_counts = xp.unique(edges, axis=0, return_counts=True)

    # mesh is closed if each edge exists twice
    return edges_uniq[edge_counts != 2]


def fix_trimesh_orientation(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Check if all faces are oriented outwards. Fix the ones that are not, and return an
    array of properly oriented faces.

    Parameters
    ----------
    vertices: np.ndarray, shape (n,3)
        Vertices of the mesh

    faces: np.ndarray, shape (n,3), dtype int
        Triples of indices

    Returns
    -------
    faces: np.ndarray, shape (n,3), dtype int, or faces and 1D array of triples
        Fixed faces
    """
    xp = array_namespace(vertices, faces)
    # use first triangle as a seed, this one needs to be oriented via inside check
    # compute facet orientation (normalized)
    inwards_mask = get_inwards_mask(vertices, faces)
    new_faces = xp.asarray(faces, copy=True)
    new_faces[inwards_mask] = new_faces[inwards_mask][:, [0, 2, 1]]
    return new_faces


def is_facet_inwards(face, faces):
    """Return boolean whether facet is pointing inwards, via ray tracing"""
    xp = array_namespace(face, faces)
    v1 = face[0] - face[1]
    v2 = face[1] - face[2]
    orient = xp.linalg.cross(v1, v2)
    orient /= xp.linalg.norm(orient)  # for single facet numpy is fine

    # create a check point by displacing the facet center in facet orientation direction
    eps = 1e-5  # unfortunately this must be quite a 'large' number :(
    check_point = face.mean(axis=0) + orient * eps

    # find out if first point is inwards
    return mask_inside_trimesh(xp.asarray([check_point]), faces)[0]


def get_inwards_mask(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of normals from triangles.
    True -> Inwards, False -> Outwards.
    This function does not check if mesh is open, and if it is, it may deliver
    inconsistent results silently.

    Parameters
    ----------
    vertices: np.ndarray, shape (n,3)
        Vertices of the mesh

    triangles: np.ndarray, shape (n,3), dtype int
        Triples of indices

    Returns
    -------
    boolean mask : ndarray, shape (n,), dtype bool
        Boolean mask of inwards orientations from provided triangles
    """

    msh = vertices[triangles]
    mask = np.full(len(triangles), False)
    indices = list(range(len(triangles)))

    # incrementally add triangles sharing at least a common edge by looping among left over
    # triangles. If next triangle with common edge is reversed, flip it.
    any_connected = False
    while indices:
        if not any_connected:
            free_edges = set()
            is_inwards = is_facet_inwards(msh[indices[0]], msh[indices])
            mask[indices] = is_inwards
        for tri_ind in indices:
            tri = triangles[tri_ind]
            edges = {(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])}
            edges_r = {(tri[1], tri[0]), (tri[2], tri[1]), (tri[0], tri[2])}
            common = free_edges & edges
            flip = False
            if not free_edges:
                common = True
            elif common:
                edges = edges_r
                flip = True
            else:
                common = free_edges & edges_r
            if common:  # break loop on first common edge found
                free_edges ^= edges
                if flip:
                    mask[tri_ind] = not mask[tri_ind]
                indices.remove(tri_ind)
                any_connected = True
                break
        else:
            # if loop reaches the end and does not find any connected edge, while still
            # having some indices to go through -> mesh is is disconnected. A new seed is
            # needed and needs to be checked via ray tracing before continuing.
            any_connected = False
    return mask


def lines_end_in_trimesh(lines: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Check if 2-point lines, where the first point lies distinctly outside of a closed
    triangular mesh (no touch), ends on the inside of that mesh

    If line ends close to a triangle surface it counts as inside (touch).
    If line passes through triangle edge/corners it counts as intersection.

    Parameters
    ----------
    lines: ndarray shape (n,2,3)
        n line segments defined through respectively 2 (first index) positions with
        coordinates (x,y,z) (last index). The first point must lie outside of the mesh.

    faces: ndarray, shape (m,3,3)
        m faces defined through respectively 3 (first index) positions with coordinates
        (x,y,z) (last index). The faces must define a closed mesh.

    Returns
    -------
        np.ndarray shape (n,)

    Note
    ----
    Part 1: plane_cross
    Checks if start and end of lines are on the same or on different sides of the planes
    defined by the triangular faces. On the way check if end-point touches the plane.

    Part 2: pass_through
    Makes use of Line-Triangle intersection as described in
    https://www.iue.tuwien.ac.at/phd/ertl/node114.html
    to check if the extended line would pass through the triangular facet
    """
    xp = array_namespace(lines, faces)

    # Part 1 ---------------------------
    normals = v_cross(faces[:, 0, :] - faces[:, 2, :], faces[:, 1, :] - faces[:, 2, :])
    normals = xp.tile(normals, ((lines.shape[0]), 1, 1))

    l0 = lines[:, 0, ...][:, xp.newaxis, ...]  # outside points
    l1 = lines[:, 1, ...][:, xp.newaxis, ...]  # possible inside test-points

    # test-point might coincide with chosen in-plane reference point (chosen faces[:,2] here).
    # this then leads to bad projection computation
    # --> choose other reference points (faces[:,1]) in those specific cases
    ref_pts = xp.tile(faces[:, 2, ...], ((lines.shape[0]), 1, 1))
    eps = 1e-16  # note: norm square !
    coincide = v_norm2(l1 - ref_pts) < eps
    if xp.any(coincide):
        ref_pts2 = xp.tile(
            faces[:, 1, ...], ((lines.shape[0]), 1, 1)
        )  # <--inefficient tile !!! only small part needed
        ref_pts[coincide] = ref_pts2[coincide]

    proj0 = v_norm_proj(l0 - ref_pts, normals)
    proj1 = v_norm_proj(l1 - ref_pts, normals)

    eps = 1e-7
    # no need to check proj0 for touch because line init pts are outside
    plane_touch = xp.abs(proj1) < eps
    # print('plane_touch:')
    # print(plane_touch)

    plane_cross = xp.sign(proj0) != xp.sign(proj1)
    # print('plane_cross:')
    # print(plane_cross)

    # Part 2 ---------------------------
    # signed areas (no 0-problem because ss0 is the outside point)
    a = faces[:, 0, ...] - l0
    b = faces[:, 1, ...] - l0
    c = faces[:, 2, ...] - l0
    d = l1 - l0
    area1 = v_dot_cross3d(a, b, d)
    area2 = v_dot_cross3d(b, c, d)
    area3 = v_dot_cross3d(c, a, d)

    eps = 1e-12
    pass_through_boundary = (
        (xp.abs(area1) < eps) | (xp.abs(area2) < eps) | (xp.abs(area3) < eps)
    )
    # print('pass_through_boundary:')
    # print(pass_through_boundary)

    area1 = xp.sign(area1)
    area2 = xp.sign(area2)
    area3 = xp.sign(area3)
    pass_through_inside = (area1 == area2) & (area2 == area3)
    # print('pass_through_inside:')
    # print(pass_through_inside)

    pass_through = pass_through_boundary | pass_through_inside

    # Part 3 ---------------------------
    result_cross = pass_through & plane_cross
    result_touch = pass_through & plane_touch

    inside1 = xp.count_nonzero(result_cross, axis=1) % 2 != 0
    inside2 = xp.any(result_touch, axis=1)

    return inside1 | inside2


def segments_intersect_facets(segments, facets, eps=1e-6):
    """Pair-wise detect if set of segments intersect set of facets.

    Parameters
    -----------
    segments: np.ndarray, shape (n,3,3)
        Set of segments.

    facets: np.ndarray, shape (n,3,3)
        Set of facets.

    eps: float
        Point to point tolerance detection. Must be strictly positive,
        otherwise some triangles may be detected as intersecting themselves.
    """
    xp = array_namespace(segments, facets)
    if eps <= 0:  # pragma: no cover
        msg = "eps must be strictly positive"
        raise ValueError(msg)

    s, t = segments.swapaxes(0, 1), facets.swapaxes(0, 1)

    # compute the normals to each triangle
    normals = xp.linalg.cross(t[2] - t[0], t[2] - t[1])
    normals /= xp.linalg.norm(normals, axis=1, keepdims=True)

    # get sign of each segment endpoint, if the sign changes then we know this
    # segment crosses the plane which contains a triangle. If the value is zero
    # the endpoint of the segment lies on the plane.
    g1 = xp.sum(normals * (s[0] - t[2]), axis=1)
    g2 = xp.sum(normals * (s[1] - t[2]), axis=1)

    # determine segments which cross the plane of a triangle.
    #  -> 1 if the sign of the end points of s is
    # different AND one of end points of s is not a vertex of t
    cross = (xp.sign(g1) != xp.sign(g2)) * (xp.abs(g1) > eps) * (xp.abs(g2) > eps)

    v = []  # get signed volumes
    for i, j in zip((0, 1, 2), (1, 2, 0), strict=False):
        sv = xp.sum((t[i] - s[1]) * xp.linalg.cross(t[j] - s[1], s[0] - s[1]), axis=1)
        v.append(xp.sign(sv))

    # same volume if s and t have same sign in v0, v1 and v2
    same_volume = xp.logical_and((v[0] == v[1]), (v[1] == v[2]))

    return cross * same_volume


def get_intersecting_triangles(vertices, triangles, r=None, r_factor=1.5, eps=1e-6):
    """Return intersecting triangles indices from a triangular mesh described
    by vertices and triangles indices.

    Parameters
    ----------
    vertices: np.ndarray, shape (n,3)
        Vertices/points of the mesh.

    triangles: np.ndarray, shape (n,3), dtype int
        Triples of vertices indices that build each triangle of the mesh.

    r: float or None
        The radius of the ball-point query for the k-d tree. If None:
        r=max_distance_between_center_and_vertices*2

    r_factor: float
        The factor by which to multiply the radius `r` of the ball-point query.
        Note that increasing this value will drastically augment computation
        time.

    eps: float
        Point to point tolerance detection. Must be strictly positive,
        otherwise some triangles may be detected as intersecting themselves.
    """
    xp = array_namespace(vertices, triangles)
    if r_factor < 1:  # pragma: no cover
        msg = "r_factor must be greater or equal to 1"
        raise ValueError(msg)

    vertices = xp.astype(vertices, xp.float32)
    facets = vertices[triangles]
    centers = xp.mean(facets, axis=1)

    if r is None:
        r = r_factor * xp.sqrt(((facets - centers[:, None, :]) ** 2).sum(-1)).max()

    kdtree = scipy.spatial.KDTree(centers)
    near = kdtree.query_ball_point(centers, r, return_sorted=False, workers=-1)
    tria1 = xp.concat(near)
    tria2 = xp.repeat(xp.arange(len(near)), [len(n) for n in near])
    pairs = np.stack([tria1, tria2], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # remove check against itself
    f1, f2 = facets[pairs[:, 0]], facets[pairs[:, 1]]
    sums = 0
    for inds in [[0, 1], [1, 2], [2, 0]]:
        sums += segments_intersect_facets(f1[:, inds], f2, eps=eps)
    return xp.unique(pairs[sums > 0])


def mask_inside_enclosing_box(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Quick-check which points lie inside a bounding box of the mesh (defined by vertices).
    Returns True when inside, False when outside bounding box.

    Parameters
    ----------
    points, ndarray, shape (n,3)
    vertices, ndarray, shape (m,3)

    Returns
    -------
    ndarray, boolean, shape (n,)
    """
    xp = array_namespace(points, vertices)
    points = xp.astype(points, xp.float64)
    Xmin = xp.min(vertices, axis=0)
    Xmax = xp.max(vertices, axis=0)
    xmin, ymin, zmin = (Xmin[i, ...] for i in range(3))
    xmax, ymax, zmax = (Xmax[i, ...] for i in range(3))
    x, y, z = (points[:, i] for i in range(3))

    eps = 1e-12
    mx = (x < xmax + eps) & (x > xmin - eps)
    my = (y < ymax + eps) & (y > ymin - eps)
    mz = (z < zmax + eps) & (z > zmin - eps)

    return mx & my & mz


def mask_inside_trimesh(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Check which points lie inside of a closed triangular mesh (defined by faces).

    Parameters
    ----------
    points, ndarray, shape (n,3)
    faces, ndarray, shape (m,3,3)

    Returns
    -------
    ndarray, shape (n,)

    Note
    ----
    Method: ray-tracing.
    Faces must form a closed mesh for this to work.
    """
    xp = array_namespace(points, faces)
    vertices = xp.reshape(faces, (-1, 3))

    # test-points inside of enclosing box
    mask_inside = mask_inside_enclosing_box(points, vertices)
    pts_in_box = points[mask_inside]

    # create test-lines from outside to test-points
    start_point_outside = xp.min(vertices, axis=0) - xp.asarray(
        [12.0012345, 5.9923456, 6.9932109]
    )
    test_lines = xp.tile(start_point_outside, ((pts_in_box.shape[0]), 2, 1))
    test_lines[:, 1, ...] = pts_in_box

    # check if test-points are inside using ray tracing
    mask_inside2 = lines_end_in_trimesh(test_lines, faces)

    mask_inside[mask_inside] = mask_inside2

    return mask_inside


def BHJM_magnet_trimesh(
    field: str,
    observers: np.ndarray,
    mesh: np.ndarray,
    polarization: np.ndarray,
    in_out="auto",
) -> np.ndarray:
    """
    - Compute triangular mesh field from triangle fields.
    - Closed meshes are assumed (input comes only from TriangularMesh class)
    - Field computations via publication: Guptasarma: GEOPHYSICS 1999 64:1, 70-74
    """
    xp = array_namespace(observers, mesh, polarization)
    polarization = xp.astype(polarization, xp.float64)
    if field in "BH":
        if mesh.ndim != 1:  # all vertices objects have same number of children
            n0, n1, *_ = mesh.shape
            vertices_tiled = xp.reshape(mesh, (-1, 3, 3))
            observers_tiled = xp.repeat(observers, n1, axis=0)
            polarization_tiled = xp.repeat(polarization, n1, axis=0)
            BHJM = BHJM_triangle(
                field="B",
                observers=observers_tiled,
                vertices=vertices_tiled,
                polarization=polarization_tiled,
            )
            BHJM = xp.reshape(BHJM, (n0, n1, 3))
            BHJM = xp.sum(BHJM, axis=1)
        else:
            nvs = [f.shape[0] for f in mesh]  # length of vertex set
            split_indices = xp.cumulative_sum(xp.asarray(nvs))[
                :-1
            ]  # remove last to avoid empty split
            vertices_tiled = xp.concat([xp.reshape(f, (-1, 3, 3)) for f in mesh])
            observers_tiled = xp.repeat(observers, nvs, axis=0)
            polarization_tiled = xp.repeat(polarization, nvs, axis=0)
            BHJM = BHJM_triangle(
                field="B",
                observers=observers_tiled,
                vertices=vertices_tiled,
                polarization=polarization_tiled,
            )
            b_split = np.split(BHJM, split_indices)
            BHJM = xp.asarray([xp.sum(bh, axis=0) for bh in b_split])
    else:
        BHJM = xp.zeros_like(observers, dtype=xp.float64)

    if field == "H":
        return BHJM / MU0

    if in_out == "auto":
        prev_ind = 0
        # group similar meshes for inside-outside evaluation and adding B
        for new_ind_item in range(BHJM.shape[0]):
            new_ind = new_ind_item
            if (
                new_ind == (BHJM.shape[0]) - 1
                or mesh[new_ind, ...].shape != mesh[prev_ind, ...].shape
                or not xp.all(mesh[new_ind, ...] == mesh[prev_ind, ...])
            ):
                if new_ind == (BHJM.shape[0]) - 1:
                    new_ind = BHJM.shape[0]
                mask_inside = mask_inside_trimesh(
                    observers[prev_ind:new_ind, ...], mesh[prev_ind, ...]
                )
                # if inside magnet add polarization vector
                BHJM[prev_ind:new_ind, ...][mask_inside] += polarization[
                    prev_ind:new_ind, ...
                ][mask_inside]
                prev_ind = new_ind
    elif in_out == "inside":
        BHJM += polarization

    if field == "B":
        return BHJM

    if field == "J":
        return BHJM

    if field == "M":
        return BHJM / MU0

    msg = f"`output_field_type` must be one of ('B', 'H', 'M', 'J'), got {field!r}"
    raise ValueError(msg)  # pragma: no cover
