"""
Implementations of analytical expressions for the magnetic field of a triangular surface.
Computation details in function docstrings.
"""
# pylint: disable=too-many-nested-blocks
# pylance: disable=Code is unreachable
import numpy as np
import scipy.spatial

from magpylib._src.fields.field_BH_triangle import triangle_field


def calculate_centroid(vertices, faces):
    """
    Calculates the centroid of a 3D triangular surface mesh.

    Parameters:
    vertices (numpy.array): an n x 3 array of vertices
    faces (numpy.array): an m x 3 array of triangle indices

    Returns:
    numpy.array: The centroid of the mesh
    """

    # Calculate the centroids of each triangle
    triangle_centroids = np.mean(vertices[faces], axis=1)

    # Compute the area of each triangle
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )

    # Calculate the centroid of the entire mesh
    mesh_centroid = np.sum(triangle_centroids.T * triangle_areas, axis=1) / np.sum(
        triangle_areas
    )

    return mesh_centroid


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
    subsets = [faces[np.isin(faces, list(ps)).all(axis=1)] for ps in subsets_inds]
    return subsets


def get_open_edges(faces: np.ndarray) -> bool:
    """
    Check if given trimesh forms a closed surface.

    Input: faces: np.ndarray, shape (n,3), dtype int
        triples of indices

    Output: open edges
    """
    edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, ::2]], axis=0)

    # unique edge pairs and counts how many
    edges = np.sort(edges, axis=1)
    edges_uniq, edge_counts = np.unique(edges, axis=0, return_counts=True)

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
    # use first triangle as a seed, this one needs to be oriented via inside check
    # compute facet orientation (normalized)
    inwards_mask = get_inwards_mask(vertices, faces)
    new_faces = faces.copy()
    new_faces[inwards_mask] = new_faces[inwards_mask][:, [0, 2, 1]]
    return new_faces


def is_facet_inwards(face, faces):
    """Return boolean whether facet is pointing inwards, via ray tracing"""
    v1 = face[0] - face[1]
    v2 = face[1] - face[2]
    orient = np.cross(v1, v2)
    orient /= np.linalg.norm(orient)  # for single facet numpy is fine

    # create a check point by displacing the facet center in facet orientation direction
    eps = 1e-5  # unfortunately this must be quite a 'large' number :(
    check_point = face.mean(axis=0) + orient * eps

    # find out if first point is inwards
    return mask_inside_trimesh(np.array([check_point]), faces)[0]


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
            # having some indices to go trough -> mesh is is disconnected. A new seed is
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
        n line segements defined through respectively 2 (first index) positions with
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

    # Part 1 ---------------------------
    normals = v_cross(faces[:, 0] - faces[:, 2], faces[:, 1] - faces[:, 2])
    normals = np.tile(normals, (len(lines), 1, 1))

    l0 = lines[:, 0][:, np.newaxis]  # outside points
    l1 = lines[:, 1][:, np.newaxis]  # possible inside test-points

    # test-point might coincide with chosen in-plane reference point (chosen faces[:,2] here).
    # this then leads to bad projection computation
    # --> choose other reference points (faces[:,1]) in those specific cases
    ref_pts = np.tile(faces[:, 2], (len(lines), 1, 1))
    eps = 1e-16  # note: norm square !
    coincide = v_norm2(l1 - ref_pts) < eps
    if np.any(coincide):
        ref_pts2 = np.tile(
            faces[:, 1], (len(lines), 1, 1)
        )  # <--inefficient tile !!! only small part needed
        ref_pts[coincide] = ref_pts2[coincide]

    proj0 = v_norm_proj(l0 - ref_pts, normals)
    proj1 = v_norm_proj(l1 - ref_pts, normals)

    eps = 1e-7
    # no need to check proj0 for touch because line init pts are outside
    plane_touch = np.abs(proj1) < eps
    # print('plane_touch:')
    # print(plane_touch)

    plane_cross = np.sign(proj0) != np.sign(proj1)
    # print('plane_cross:')
    # print(plane_cross)

    # Part 2 ---------------------------
    # signed areas (no 0-problem because ss0 is the outside point)
    a = faces[:, 0] - l0
    b = faces[:, 1] - l0
    c = faces[:, 2] - l0
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
    if eps <= 0:  # pragma: no cover
        raise ValueError("eps must be strictly positive")

    s, t = segments.swapaxes(0, 1), facets.swapaxes(0, 1)

    # compute the normals to each triangle
    normals = np.cross(t[2] - t[0], t[2] - t[1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # get sign of each segment endpoint, if the sign changes then we know this
    # segment crosses the plane which contains a triangle. If the value is zero
    # the endpoint of the segment lies on the plane.
    g1 = np.sum(normals * (s[0] - t[2]), axis=1)
    g2 = np.sum(normals * (s[1] - t[2]), axis=1)

    # determine segments which cross the plane of a triangle.
    #  -> 1 if the sign of the end points of s is
    # different AND one of end points of s is not a vertex of t
    cross = (np.sign(g1) != np.sign(g2)) * (np.abs(g1) > eps) * (np.abs(g2) > eps)

    v = []  # get signed volumes
    for i, j in zip((0, 1, 2), (1, 2, 0)):
        sv = np.sum((t[i] - s[1]) * np.cross(t[j] - s[1], s[0] - s[1]), axis=1)
        v.append(np.sign(sv))

    # same volume if s and t have same sign in v0, v1 and v2
    same_volume = np.logical_and((v[0] == v[1]), (v[1] == v[2]))

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
    if r_factor < 1:  # pragma: no cover
        raise ValueError("r_factor must be greater or equal to 1")

    vertices = vertices.astype(np.float32)
    facets = vertices[triangles]
    centers = np.mean(facets, axis=1)

    if r is None:
        r = r_factor * np.sqrt(((facets - centers[:, None, :]) ** 2).sum(-1)).max()

    kdtree = scipy.spatial.KDTree(centers)
    near = kdtree.query_ball_point(centers, r, return_sorted=False, workers=-1)
    tria1 = np.concatenate(near)
    tria2 = np.repeat(np.arange(len(near)), [len(n) for n in near])
    pairs = np.stack([tria1, tria2], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # remove check against itself
    f1, f2 = facets[pairs[:, 0]], facets[pairs[:, 1]]
    sums = 0
    for inds in [[0, 1], [1, 2], [2, 0]]:
        sums += segments_intersect_facets(f1[:, inds], f2, eps=eps)
    return np.unique(pairs[sums > 0])


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
    vertices = faces.reshape((-1, 3))

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
    mask_inside2 = lines_end_in_trimesh(test_lines, faces)

    mask_inside[mask_inside] = mask_inside2

    return mask_inside


def magnet_trimesh_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    mesh: np.ndarray,
    in_out="auto",
) -> np.ndarray:
    """
    core-like function that computes the field of triangular meshes using the triangle_field
    - closed nice meshes are assumed (input comes only from TriangularMesh class)

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of mT.

    mesh: ndarray, shape (n,n1,3,3) or ragged sequence
        Triangular mesh of shape [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
        `mesh` can be a ragged sequence of mesh-children with different lengths.

    in_out: {'auto', 'inside', 'outside'}
        Tells if the points are inside or outside the enclosing mesh for the correct B/H-field
        calculation. By default `in_out='auto'` and the inside/outside mask is automatically
        generated using a ray tracing algorithm to determine which observers are inside and which
        are outside the closed body. For performance reasons, one can define `in_out='outside'`
        or `in_out='inside'` if it is known in advance that all observers satisfy the same
        condition.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Notes
    -----
    Field computations via publication:
    Guptasarma: GEOPHYSICS 1999 64:1, 70-74
    """
    if mesh.ndim != 1:  # all vertices objects have same number of children
        n0, n1, *_ = mesh.shape
        vertices_tiled = mesh.reshape(-1, 3, 3)
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
        nvs = [f.shape[0] for f in mesh]  # length of vertex set
        split_indices = np.cumsum(nvs)[:-1]  # remove last to avoid empty split
        vertices_tiled = np.concatenate([f.reshape((-1, 3, 3)) for f in mesh])
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
            # group similar meshs for inside-outside evaluation and adding B
            for new_ind, _ in enumerate(B):
                if (
                    new_ind == len(B) - 1
                    or mesh[new_ind].shape != mesh[prev_ind].shape
                    or not np.all(mesh[new_ind] == mesh[prev_ind])
                ):
                    if new_ind == len(B) - 1:
                        new_ind = len(B)
                    inside_mask = mask_inside_trimesh(
                        observers[prev_ind:new_ind], mesh[prev_ind]
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
