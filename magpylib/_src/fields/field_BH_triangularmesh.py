"""
Implementations of analytical expressions for the magnetic field of a triangular surface.
Computation details in function docstrings.
"""
# pylint: disable=too-many-nested-blocks
# pylance: disable=Code is unreachable
import numpy as np

from magpylib._src.fields.field_BH_triangle import triangle_field


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
    ab = a * b
    ab = ab[..., 0] + ab[..., 1] + ab[..., 2]

    return ab / np.sqrt(v_norm2(a) * v_norm2(b))


def v_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a x b
    """
    result = np.array(
        (
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        )
    ).T
    return result


def v_norm_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a x b / |a x b|

    """
    res = np.array(
        (
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        )
    )
    res2 = res**2
    vnorm = np.sqrt(res2[0] + res2[1] + res2[2])
    res = res / vnorm
    return res.T


def v_dot_cross3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    a x b * c
    """
    result = (
        (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0]
        + (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1]
        + (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2]
    )
    return result


def get_disjoint_triangles_subsets(triangles: list) -> list:
    """Return a list of disjoint triangles sets"""
    subsets_inds = []
    tria_temp = triangles.copy()
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
    subsets = [
        triangles[np.isin(triangles, list(ps)).all(axis=1)] for ps in subsets_inds
    ]
    return subsets


def trimesh_is_closed(triangles: np.ndarray) -> bool:
    """
    Check if given trimesh forms a closed surface.

    Input: triangles: np.ndarray, shape (n,3), dtype int
        triples of indices

    Output: bool (True if closed, False if open)
    """
    edges = np.concatenate(
        [triangles[:, 0:2], triangles[:, 1:3], triangles[:, ::2]], axis=0
    )

    # unique edge pairs and counts how many
    edges = np.sort(edges, axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)

    # mesh is closed if each edge exists twice
    return np.all(edge_counts == 2)


def fix_trimesh_orientation(vertices: np.ndarray, triangles: np.ndarray)-> np.ndarray:
    """Check if all triangles are oriented outwards. Fix the ones that are not, and return an
    array of properly oriented triangles.

    Parameters
    ----------
    vertices: np.ndarray, shape (n,3)
        vertices of the mesh

    triangles: np.ndarray, shape (n,3), dtype int
        triples of indices

    Returns
    -------
    triangles: np.ndarray, shape (n,3), dtype int
        fixed triangles
    """

    # use first triangle as a seed, this one needs to be oriented via inside check
    # compute facet orientation (normalized)
    facets = vertices[triangles]
    facet0 = facets[0]
    v1 = facet0[0] - facet0[1]
    v2 = facet0[1] - facet0[2]
    orient = np.cross(v1, v2)
    orient /= np.linalg.norm(orient) # for single facet numpy is fine

    # create a check point by displacing the facet center in facet orientation direction
    eps = 1e-6  # unfortunately this must be quite a 'large' number :(
    check_point = facet0.mean(axis=0) + orient * eps

    # find out if point is inside
    first_is_inside = mask_inside_trimesh(np.array([check_point]), facets)[0]

    tri_temp=triangles.copy() # do not modify input triangles

    if first_is_inside:
        tri_temp[0] = tri_temp[0, [0,2,1]]

    new_triangles=[]
    free_edges = set()
    # incrementally add triangles sharing at least a common edge by looping among left over
    # triangles. If next triangle with common edge is reversed, flip it.
    while len(tri_temp)!=0:
        for tri_ind, tri in enumerate(tri_temp):
            edges = {(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])}
            edges_r = {(tri[1], tri[0]), (tri[2], tri[1]), (tri[0], tri[2])}
            common = free_edges & edges
            if not free_edges:
                common=True
            elif common:
                edges = edges_r
                tri = tri[[0,2,1]] # flip triangle
            else:
                common = free_edges & edges_r
            if common: # break loop on first common edge found
                new_triangles.append(tri)
                free_edges ^= edges
                tri_temp = np.delete(tri_temp, tri_ind, 0)
                break
    return np.array(new_triangles)


def lines_end_in_trimesh(lines: np.ndarray, facets: np.ndarray) -> np.ndarray:
    """
    Check if 2-point lines, where the first point lies distinctly outside of a closed
    triangular mesh (no touch), ends on the inside of that mesh

    If line ends close to a triangle surface it counts as inside (touch).
    If line passes through triangle edge/corners it counts as intersection.

    Parameters
    ----------
    lines: ndarray shape (n,2,3)
        n line segements defined through respectively 2 (first index) positions with
        coordinates (x,y,z) (last index). The first point must lie outside of the mesh.

    facets: ndarray, shape (m,3,3)
        m facets defined through respectively 3 (first index) positions with coordinates
        (x,y,z) (last index). The facets must define a closed mesh.

    Returns
    -------
        np.ndarray shape (n,)

    Note
    ----
    Part 1: plane_cross
    Checks if start and end of lines are on the same or on different sides of the planes
    defined by the triangular facets. On the way check if end-point touches the plane.

    Part 2: pass_through
    Makes use of Line-Triangle intersection as described in
    https://www.iue.tuwien.ac.at/phd/ertl/node114.html
    to check if the extended line would pass through the triangular facet
    """

    # Part 1 ---------------------------
    normals = v_cross(facets[:, 0] - facets[:, 2], facets[:, 1] - facets[:, 2])
    normals = np.tile(normals, (len(lines), 1, 1))

    l0 = lines[:, 0][:, np.newaxis]  # outside points
    l1 = lines[:, 1][:, np.newaxis]  # possible inside test-points

    # test-point might coincide with chosen in-plane reference point (chosen facets[:,2] here).
    # this then leads to bad projection computation
    # --> choose other reference points (facets[:,1]) in those specific cases
    ref_pts = np.tile(facets[:, 2], (len(lines), 1, 1))
    eps = 1e-16  # note: norm square !
    coincide = v_norm2(l1 - ref_pts) < eps
    if np.any(coincide):
        ref_pts2 = np.tile(
            facets[:, 1], (len(lines), 1, 1)
        )  # <--inefficient tile !!! only small part needed
        ref_pts[coincide] = ref_pts2[coincide]
    coincide = v_norm2(l1 - ref_pts) < eps

    proj0 = v_norm_proj(l0 - ref_pts, normals)
    proj1 = v_norm_proj(l1 - ref_pts, normals)

    eps = 1e-7
    plane_touch = (
        np.abs(proj1) < eps
    )  # no need to check proj0 for touch because line init pts are outside
    # print('plane_touch:')
    # print(plane_touch)

    plane_cross = np.sign(proj0) != np.sign(proj1)
    # print('plane_cross:')
    # print(plane_cross)

    # Part 2 ---------------------------
    # signed areas (no 0-problem because ss0 is the outside point)
    a = facets[:, 0] - l0
    b = facets[:, 1] - l0
    c = facets[:, 2] - l0
    d = l1 - l0
    area1 = v_dot_cross3d(a, b, d)
    area2 = v_dot_cross3d(b, c, d)
    area3 = v_dot_cross3d(c, a, d)

    eps = 1e-12
    pass_through_boundary = (
        (np.abs(area1) < eps) | (np.abs(area2) < eps) | (np.abs(area3) < eps)
    )
    # print('pass_through_boundary:')
    # print(pass_through_boundary)

    area1 = np.sign(area1)
    area2 = np.sign(area2)
    area3 = np.sign(area3)
    pass_through_inside = (area1 == area2) * (area2 == area3)
    # print('pass_through_inside:')
    # print(pass_through_inside)

    pass_through = pass_through_boundary | pass_through_inside

    # Part 3 ---------------------------
    result_cross = pass_through * plane_cross
    result_touch = pass_through * plane_touch

    inside1 = np.sum(result_cross, axis=1) % 2 != 0
    inside2 = np.any(result_touch, axis=1)

    return inside1 | inside2


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
    xmin, ymin, zmin = np.min(vertices, axis=0)
    xmax, ymax, zmax = np.max(vertices, axis=0)
    x, y, z = points.T

    eps = 1e-12
    mx = (x < xmax + eps) & (x > xmin - eps)
    my = (y < ymax + eps) & (y > ymin - eps)
    mz = (z < zmax + eps) & (z > zmin - eps)

    return mx & my & mz


def mask_inside_trimesh(points: np.ndarray, facets: np.ndarray) -> np.ndarray:
    """
    Check which points lie inside of a closed triangular mesh (defined by facets).

    Parameters
    ----------
    points, ndarray, shape (n,3)
    facets, ndarray, shape (m,3,3)

    Returns
    -------
    ndarray, shape (n,)

    Note
    ----
    Method: ray-tracing.
    Facets must form a closed mesh for this to work.
    """
    vertices = facets.reshape((-1, 3))

    # test-points inside of enclosing box
    mask_inside = mask_inside_enclosing_box(points, vertices)
    pts_in_box = points[mask_inside]

    # create test-lines from outside to test-points
    start_point_outside = np.min(vertices, axis=0) - np.array(
        [12.0012345, 5.9923456, 6.9932109]
    )
    test_lines = np.tile(start_point_outside, (len(pts_in_box), 2, 1))
    test_lines[:, 1] = pts_in_box

    # check if test-points are inside using ray tracing
    mask_inside2 = lines_end_in_trimesh(test_lines, facets)

    mask_inside[mask_inside] = mask_inside2

    return mask_inside


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