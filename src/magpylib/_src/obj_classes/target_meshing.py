"""Meshing functions"""

# pylint: disable=too-many-lines
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-function-args

from itertools import product

import numpy as np


def _apportion_triple(triple, min_val=1, max_iter=30):
    """Apportion values of a triple, so that the minimum value `min_val` is respected
    and the product of all values remains the same.
    Example: _apportion_triple([1, 2, 50], min_val=3)
    -> [ 2.99999999  3.         11.11111113]
    """
    triple = np.abs(np.array(triple, dtype=float))
    count = 0
    while any(n < min_val for n in triple) and count < max_iter:
        count += 1
        amin, amax = triple.argmin(), triple.argmax()
        factor = min_val / triple[amin]
        if triple[amax] >= factor * min_val:
            triple /= factor**0.5
            triple[amin] *= factor**1.5
    return triple


def _cells_from_dimension(
    dim,
    target_elems,
    min_val=1,
    strict_max=False,
    parity=None,
):
    """Divide a dimension triple with a target scalar of elements, while apportioning
    the number of elements based on the dimension proportions. The resulting divisions
    are the closest to cubes.

    Parameters
    ----------
    dim: array-like of length 3
        Dimensions of the object to be divided.
    target_elems: int,
        Total number of elements as target for the procedure. Actual final number is
        likely to differ.
    min_val: int
        Minimum value of the number of divisions per dimension.
    strict_max: bool
        If ``True``, the ``target_elem`` value becomes a strict maximum and the product of
        the resulting triple will be strictly smaller than the target.
    parity: {None, 'odd', 'even'}
        All elements of the resulting triple will match the given parity. If ``None``, no
        parity check is performed.

    Returns
    -------
    numpy.ndarray of length 3
        array corresponding of the number of divisions for each dimension

    Examples
    --------
    >>> _cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=True)
    [ 4  9 25]  # Actual total: 900
    >>> _cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=False)
    [ 4  9 26]  # Actual total: 936
    >>> _cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=True)
    [ 3 11 27]  # Actual total: 891
    >>> _cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=False)
    [ 5  7 27]  # Actual total: 945
    >>> _cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=True)
    [ 4  8 26]  # Actual total: 832
    >>> _cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=False)
    [ 4 10 24]  # Actual total: 960
    """
    elems = np.prod(target_elems)  # in case target_elems is an iterable

    # define parity functions
    if parity == "odd":
        funcs = [
            lambda x, add=add, fn=fn: int(2 * fn(x / 2) + add)
            for add in (-1, 1)
            for fn in (np.ceil, np.floor)
        ]
    elif parity == "even":
        funcs = [lambda x, fn=fn: int(2 * fn(x / 2)) for fn in (np.ceil, np.floor)]
    else:
        funcs = [np.ceil, np.floor]

    # make sure the number of elements is sufficient
    elems = max(min_val**3, elems)

    # float estimate of the elements while product=target_elems and proportions are kept
    x, y, z = np.abs(dim)
    a = x ** (2 / 3) * (elems / y / z) ** (1 / 3)
    b = y ** (2 / 3) * (elems / x / z) ** (1 / 3)
    c = z ** (2 / 3) * (elems / x / y) ** (1 / 3)
    a, b, c = _apportion_triple((a, b, c), min_val=min_val)
    epsilon = elems
    # run all combinations of rounding methods, including parity matching to find the
    # closest triple with the target_elems constrain
    result = [funcs[0](k) for k in (a, b, c)]  # first guess
    for fn in product(*[funcs] * 3):
        res = [f(k) for f, k in zip(fn, (a, b, c), strict=False)]
        epsilon_new = elems - np.prod(res)
        if (
            np.abs(epsilon_new) <= epsilon
            and all(r >= min_val for r in res)
            and (not strict_max or epsilon_new >= 0)
        ):
            epsilon = np.abs(epsilon_new)
            result = res
    return np.array(result).astype(int)


def _get_cuboid_mesh_single(a, b, c, target_elems):
    """Helper function to generate a single cuboid mesh."""
    if isinstance(target_elems, int):
        if target_elems == 1:
            n1, n2, n3 = 1, 1, 1
        else:
            cell_size = (a * b * c / target_elems) ** (1 / 3)
            n1 = max(1, int(np.round(a / cell_size)))
            n2 = max(1, int(np.round(b / cell_size)))
            n3 = max(1, int(np.round(c / cell_size)))
    else:
        n1, n2, n3 = target_elems

    n_total = n1 * n2 * n3

    def _cell_centers(bounds, n):
        return (bounds[:-1] + bounds[1:]) / 2 if n > 1 else np.array([0.0])

    xs_cent = _cell_centers(np.linspace(-a / 2, a / 2, n1 + 1), n1)
    ys_cent = _cell_centers(np.linspace(-b / 2, b / 2, n2 + 1), n2)
    zs_cent = _cell_centers(np.linspace(-c / 2, c / 2, n3 + 1), n3)

    xx, yy, zz = np.meshgrid(xs_cent, ys_cent, zs_cent, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    volume = a * b * c / n_total

    return pts, volume


def _target_mesh_cuboid(dimension, magnetization, target_elems):
    """Cuboid mesh in the local object coordinates with path-varying parameters.

    Generates a point-cloud of n1 x n2 x n3 points inside a cuboid with sides a, b, c.
    The points are centers of cubical cells that fill the Cuboid.

    Parameters
    ----------
    dimension: np.ndarray, shape (p, 3)
        Dimensions of the cuboid (length, width, height) along path.
        Already path-enabled from the class.
    magnetization: np.ndarray, shape (p, 3)
        Magnetization vector for the mesh points along path.
        Already path-enabled from the class.
    target_elems: int or tuple (n1, n2, n3)
        Target number of elements in the mesh. If an integer is provided, it is treated as
        the total number of elements.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) or (p, n, 3) - mesh points
        "moments": np.ndarray, shape (n, 3) or (p, n, 3) - moments associated with each point
    }
    """
    p_len = len(dimension)

    # Check for path variation
    # Optimization: Compute unique dimensions once and reuse
    unique_dims, indices = np.unique(dimension, axis=0, return_inverse=True)
    dimension_varying = len(unique_dims) > 1

    magnetization_varying = np.unique(magnetization, axis=0).shape[0] > 1
    has_path_varying = dimension_varying or magnetization_varying

    if dimension_varying:
        unique_pts_list = []
        unique_volumes_list = []

        for a, b, c in unique_dims:
            pts, volume = _get_cuboid_mesh_single(a, b, c, target_elems)
            unique_pts_list.append(pts)
            unique_volumes_list.append(volume)

        # Pad and stack
        max_n = max(len(pts) for pts in unique_pts_list)
        pts_array = np.zeros((p_len, max_n, 3))
        moments_array = np.zeros((p_len, max_n, 3))

        # Reconstruct full path arrays using vectorized assignment where possible
        for i in range(len(unique_dims)):
            # Find all path indices corresponding to this unique dimension
            mask = indices == i
            n = len(unique_pts_list[i])

            # Assign points (broadcasted)
            pts_array[mask, :n] = unique_pts_list[i]

            # Assign moments
            # magnetization[mask] is (k, 3), unique_volumes_list[i] is scalar
            # Result is (k, 3), which broadcasts to (k, n, 3) if we reshape or use broadcasting
            mags_subset = magnetization[mask]  # Shape (k, 3)
            vol = unique_volumes_list[i]

            # We need to assign to moments_array[mask, :n] which is (k, n, 3)
            # mags_subset * vol is (k, 3)
            # We need to broadcast (k, 3) to (k, n, 3)
            moments_array[mask, :n] = (mags_subset * vol)[:, np.newaxis, :]

    else:
        # Constant dimensions - compute mesh once and broadcast
        # Use unique_dims[0] which corresponds to the single unique dimension
        a, b, c = unique_dims[0]
        pts_single, volume = _get_cuboid_mesh_single(a, b, c, target_elems)
        n_total = len(pts_single)

        pts_array = np.broadcast_to(pts_single[None, :, :], (p_len, n_total, 3))
        volumes = np.full(p_len, volume)

        # Calculate moments
        moments_array = np.broadcast_to(
            volumes[:, None, None] * magnetization[:, None, :],
            (p_len, n_total, 3),
        )

    # Squeeze if no path variation
    if not has_path_varying:
        pts_array, moments_array = pts_array[0], moments_array[0]

    return {"pts": pts_array, "moments": moments_array}


def _get_cylinder_mesh_single(r1, r2, h, phi1, phi2, target_elems):
    """Helper function to generate a single cylinder mesh."""
    n = target_elems
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    # "unroll" the cylinder and distribute the target number of elements along the
    # circumference, radius and height.
    nphi, nr, nh = _cells_from_dimension(dim, n)

    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cells = []
    volumes = []
    for r_ind in range(nr):
        # redistribute number divisions proportionally to the radius
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            # use a cylinder for the innermost cells if there are at least 3 layers and
            # if it is closed, use cylinder segments otherwise
            if nr >= 3 and r[r_ind] == 0 and phi2 - phi1 == 360:
                cell = (0, 0, pos_h)
                cells.append(cell)
                volumes.append(np.pi * r[r_ind + 1] ** 2 * dh)
            else:
                for phi_ind in range(nphi_r):
                    radial_coord = (r[r_ind] + r[r_ind + 1]) / 2
                    angle_coord = (phi[phi_ind] + phi[phi_ind + 1]) / 2

                    cell = (
                        radial_coord * np.cos(np.deg2rad(angle_coord)),
                        radial_coord * np.sin(np.deg2rad(angle_coord)),
                        pos_h,
                    )
                    cells.append(cell)
                    volumes.append(
                        np.pi
                        * (r[r_ind + 1] ** 2 - r[r_ind] ** 2)
                        * dh
                        / nphi_r
                        * (phi2 - phi1)
                        / 360
                    )

    pts = np.array(cells)
    volumes = np.array(volumes)
    return pts, volumes


def _target_mesh_cylinder(r1, r2, h, phi1, phi2, magnetization, target_elems):
    """Cylinder mesh in the local object coordinates with path-varying parameters.

    Generates a point-cloud of mesh points inside a cylinder or cylinder segment.

    Parameters
    ----------
    r1: np.ndarray, shape (p,)
        Inner radius of the cylinder along path.
        Already path-enabled from the class.
    r2: np.ndarray, shape (p,)
        Outer radius of the cylinder along path.
        Already path-enabled from the class.
    h: np.ndarray, shape (p,)
        Height of the cylinder along path.
        Already path-enabled from the class.
    phi1: np.ndarray, shape (p,)
        Start angle of the cylinder in degrees along path.
        Already path-enabled from the class.
    phi2: np.ndarray, shape (p,)
        End angle of the cylinder in degrees along path.
        Already path-enabled from the class.
    magnetization: np.ndarray, shape (p, 3)
        Magnetization vector for the mesh points along path.
        Already path-enabled from the class.
    target_elems: int
        Target number of elements in the mesh.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) or (p, n, 3) - mesh points
        "moments": np.ndarray, shape (n, 3) or (p, n, 3) - moments associated with each point
    }
    """
    p_len = len(r1)

    # Combine geometry parameters for uniqueness check
    # Shape (p, 5)
    geometry_params = np.stack([r1, r2, h, phi1, phi2], axis=1)

    # Check for path variation
    # Optimization: Compute unique geometries once and reuse
    unique_geoms, indices = np.unique(geometry_params, axis=0, return_inverse=True)
    geometry_varying = len(unique_geoms) > 1

    magnetization_varying = np.unique(magnetization, axis=0).shape[0] > 1
    has_path_varying = geometry_varying or magnetization_varying

    if geometry_varying:
        unique_pts_list = []
        unique_volumes_list = []

        for r1_i, r2_i, h_i, phi1_i, phi2_i in unique_geoms:
            pts, volumes = _get_cylinder_mesh_single(
                r1_i, r2_i, h_i, phi1_i, phi2_i, target_elems
            )
            unique_pts_list.append(pts)
            unique_volumes_list.append(volumes)

        # Pad and stack
        max_n = max(len(pts) for pts in unique_pts_list)
        pts_array = np.zeros((p_len, max_n, 3))
        moments_array = np.zeros((p_len, max_n, 3))

        # Reconstruct full path arrays using vectorized assignment where possible
        for i in range(len(unique_geoms)):
            # Find all path indices corresponding to this unique geometry
            mask = indices == i
            n = len(unique_pts_list[i])

            # Assign points (broadcasted)
            pts_array[mask, :n] = unique_pts_list[i]

            # Assign moments
            # magnetization[mask] is (k, 3), unique_volumes_list[i] is (n,)
            # Result is (k, n, 3)
            mags_subset = magnetization[mask]  # Shape (k, 3)
            vols = unique_volumes_list[i]  # Shape (n,)

            # We need to assign to moments_array[mask, :n] which is (k, n, 3)
            # mags_subset[:, None, :] is (k, 1, 3)
            # vols[None, :, None] is (1, n, 1)
            # Product is (k, n, 3)
            moments_array[mask, :n] = mags_subset[:, None, :] * vols[None, :, None]

    else:
        # Constant geometry - compute mesh once and broadcast
        r1_i, r2_i, h_i, phi1_i, phi2_i = unique_geoms[0]
        pts_single, volumes = _get_cylinder_mesh_single(
            r1_i, r2_i, h_i, phi1_i, phi2_i, target_elems
        )
        n_total = len(pts_single)

        pts_array = np.broadcast_to(pts_single[None, :, :], (p_len, n_total, 3))

        # Calculate moments
        # volumes is (n,), magnetization is (p, 3)
        # Result should be (p, n, 3)
        moments_array = magnetization[:, None, :] * volumes[None, :, None]

    # Squeeze if no path variation
    if not has_path_varying:
        pts_array, moments_array = pts_array[0], moments_array[0]

    return {"pts": pts_array, "moments": moments_array}


def _target_mesh_circle(diameter, current, n_points):
    """
    Circle meshing in the local object coordinates with path-varying parameters

    Parameters
    ----------
    diameter: array_like, shape (p,) - Diameter of the circle along path.
    current: array_like, shape (p,) - electric current along path
    n_points: int >= 4 - Number of points along the circle.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (p, n, 3) - central edge positions along path
        "cvecs": np.ndarray, shape (p, n, 3) - current vectors along path
    }
    """
    r, i0, n = np.atleast_1d(diameter / 2), np.atleast_1d(current), n_points
    has_path_varying = (np.unique(r).shape[0] > 1) or (np.unique(i0).shape[0] > 1)
    if not has_path_varying:
        r, i0 = r[:1], i0[:1]
    p_len = len(r)

    # Pre-compute angle arrays
    angles = 2 * np.pi * np.arange(n + 1) / n

    # Vectorized computation for all path positions
    # Shape: (p,) -> (p, 1) for broadcasting
    r_expanded = r.reshape(-1, 1)
    i0_expanded = i0.reshape(-1, 1)

    # construct polygon with same area as circle for all path positions
    r1 = r_expanded * np.sqrt((2 * np.pi) / (n * np.sin(2 * np.pi / n)))

    # Compute vertices for all path positions: shape (p, n+1)
    vx = r1 * np.cos(angles)  # Broadcasting: (p, 1) * (n+1,) -> (p, n+1)
    vy = r1 * np.sin(angles)

    # compute midpoints: shape (p, n)
    midx = (vx[:, :-1] + vx[:, 1:]) / 2
    midy = (vy[:, :-1] + vy[:, 1:]) / 2
    midz = np.zeros((p_len, n))

    # compute tangents: shape (p, n)
    tx = vx[:, 1:] - vx[:, :-1]
    ty = vy[:, 1:] - vy[:, :-1]

    # Stack to create pts: shape (p, n, 3)
    pts = np.stack([midx, midy, midz], axis=2)

    # Create cvecs with current scaling: shape (p, n, 3)
    cvecs = np.stack([tx, ty, midz], axis=2) * i0_expanded.reshape(p_len, 1, 1)

    if not has_path_varying:
        pts, cvecs = pts[0], cvecs[0]
    return {"pts": pts, "cvecs": cvecs}


def _subdiv(triangles: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """
    Subdivides the given triangles based on the specified number of splits
    using bisection along longest edge.

    Idea of this algorithm:
    Loop over maximal number of splits. In each step select triangles to be split
    from TRIA, split them and store them back into TRIA.

    Returns: triangles np.ndarray shape (n, 3, 3)
    """
    n_sub = 2**splits  # subdivisions per tria
    n_tot = np.sum(n_sub)  # total number of trias

    # Group indices in TRIA and MASK for broadcasting
    ends = np.cumsum(n_sub)
    starts = np.r_[0, ends[:-1]]

    # Store triangles only here
    TRIA = np.empty((n_tot, 3, 3))
    TRIA[starts] = triangles  # store input

    # Masking TRIA for selection and broadcasting
    MASK = np.zeros((n_tot), dtype=bool)

    # Create initial selection mask
    MASK[starts] = True

    for i in range(max(splits)):
        # Reset selection MASK of completed groups
        mask_split = i == splits
        for start in starts[mask_split]:
            MASK[start : start + 2**i] = False

        # Select all triangles that should be split
        triangles = TRIA[MASK]

        # Create broadcasting mask
        mask_split = (
            i < splits
        )  # select triangle groups where further splitting is required
        for start in starts[mask_split]:
            MASK[start : start + 2 ** (i + 1)] = True

        # Vectorized Bisection algorithm ########################################
        A = triangles[:, 0]
        B = triangles[:, 1]
        C = triangles[:, 2]

        # Squared lengths of edges
        d2_AB = np.sum((B - A) ** 2, axis=1)
        d2_BC = np.sum((C - B) ** 2, axis=1)
        d2_CA = np.sum((A - C) ** 2, axis=1)

        case1 = (d2_AB >= d2_BC) * (d2_AB >= d2_CA)
        case2 = d2_BC >= d2_CA
        case3 = ~(case1 | case2)

        # instead of creating this array, we could direvctly use TRIA
        new_triangles = np.empty((len(triangles), 2, 3, 3), dtype=float)

        if np.any(case1):
            new_triangles[case1, 0, 0] = A[case1]
            new_triangles[case1, 0, 1] = (A[case1] + B[case1]) / 2.0
            new_triangles[case1, 0, 2] = C[case1]
            new_triangles[case1, 1, 0] = (A[case1] + B[case1]) / 2.0
            new_triangles[case1, 1, 1] = B[case1]
            new_triangles[case1, 1, 2] = C[case1]

        if np.any(case2):
            new_triangles[case2, 0, 0] = B[case2]
            new_triangles[case2, 0, 1] = (B[case2] + C[case2]) / 2.0
            new_triangles[case2, 0, 2] = A[case2]
            new_triangles[case2, 1, 0] = (B[case2] + C[case2]) / 2.0
            new_triangles[case2, 1, 1] = C[case2]
            new_triangles[case2, 1, 2] = A[case2]

        if np.any(case3):
            new_triangles[case3, 0, 0] = C[case3]
            new_triangles[case3, 0, 1] = (C[case3] + A[case3]) / 2.0
            new_triangles[case3, 0, 2] = B[case3]
            new_triangles[case3, 1, 0] = (C[case3] + A[case3]) / 2.0
            new_triangles[case3, 1, 1] = A[case3]
            new_triangles[case3, 1, 2] = B[case3]

        TRIA[MASK] = new_triangles.reshape(-1, 3, 3)

    return TRIA


def _target_mesh_triangle_current(
    triangles: np.ndarray, cds: np.ndarray, n_target: int
):
    """
    Refines input triangles into >n_target triangles using bisection along longest edge.
    n_target must be at least number of input triangles in which case one mesh point
    per triangle is created.

    Parameters:
    - triangles (n, 3, 3) or (p, n, 3, 3) array, triangles with optional path dimension
    - cds: (n, 3) or (p, n, 3) array, current density vectors with optional path dimension
    - n_target: int, target number of mesh points

    Returns dict:
    - mesh: centroids of refined triangles
    - cvecs: current vectors
    """
    # Handle different input shapes
    triangles = np.asarray(triangles)
    cds = np.asarray(cds)

    # Check if we have path dimension
    has_path_dim = triangles.ndim == 4

    if not has_path_dim:
        # Original behavior for no path dimension - add path dimension for uniform processing
        triangles = triangles[np.newaxis, ...]  # Shape: (1, n, 3, 3)
        cds = cds[np.newaxis, ...]  # Shape: (1, n, 3)

    p_len = triangles.shape[0]
    n_tria = triangles.shape[1]

    # Check for path-varying parameters
    has_path_varying = (np.unique(triangles, axis=0).shape[0] > 1) or (
        np.unique(cds, axis=0).shape[0] > 1
    )

    # Vectorized computation of surfaces for all paths: shape (p, n)
    surfaces = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
        ),
        axis=2,
    )

    # Calculate splits for all paths
    # Note: splits are the same for all paths since they're based on n_target
    splits = np.zeros(n_tria, dtype=int)
    surfaces_temp = surfaces[0].copy()  # Use first path for split calculation
    n_tria_temp = n_tria
    while n_tria_temp < n_target:
        idx = np.argmax(surfaces_temp)
        surfaces_temp[idx] /= 2.0
        splits[idx] += 1
        n_tria_temp = np.sum(2**splits)

    # Apply the surface divisions to all paths (vectorized)
    # Each surface gets divided by 2^splits[i]
    surfaces = surfaces / (2.0 ** splits[np.newaxis, :])

    # Vectorized subdivision and centroid calculation for all paths
    # Apply _subdiv to each path and stack results
    trias_refined_list = [_subdiv(triangles[p_idx], splits) for p_idx in range(p_len)]
    trias_refined = np.stack(trias_refined_list, axis=0)  # Shape: (p, n_refined, 3, 3)

    # Calculate centroids for all paths at once: shape (p, n_refined, 3)
    pts = np.mean(trias_refined, axis=2)

    # Expand surfaces and cds for all paths
    # We need to repeat along the triangle dimension (axis=1 after adding path dim)
    surfaces_expanded_list = [
        np.repeat(surfaces[p_idx], 2**splits) for p_idx in range(p_len)
    ]
    cvecs_list = [
        np.repeat(cds[p_idx], 2**splits, axis=0)
        * surfaces_expanded_list[p_idx][:, np.newaxis]
        for p_idx in range(p_len)
    ]

    # Stack results: shape (p, n_refined, 3)
    cvecs = np.stack(cvecs_list, axis=0)

    # If no path variation, return squeezed arrays
    if not has_path_varying:
        pts = pts[0]
        cvecs = cvecs[0]

    return {"pts": pts, "cvecs": cvecs}


def _target_mesh_polyline(vertices, current, n_points):
    """
    Polyline meshing in the local object coordinates with path-varying parameters

    Parameters
    ----------
    vertices: array-like, shape (n, 3) or (p, n, 3) - vertices of the polyline
    i0: array-like, shape (p,) or scalar - electric current along path
        Note: For internal use, lengths are guaranteed to match vertices path dimension
    n_points: int >= n_segments - Number of points along the polyline

    If n_points is int, the algorithm tries to distribute these points evenly
    over the polyline, enforcing at least one point per segment.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (m, 3) or (p, m, 3) - central segment positions
        "cvecs": np.ndarray, shape (m, 3) or (p, m, 3) - current vectors
    }
    """
    # Handle different input shapes
    vertices = np.asarray(vertices)
    i0 = np.atleast_1d(current)

    p_len = i0.shape[0]

    # Check for path-varying parameters
    has_path_varying = (np.unique(vertices, axis=0).shape[0] > 1) or (
        np.unique(i0).shape[0] > 1
    )

    n_vertices = vertices.shape[1]
    n_segments = n_vertices - 1

    # Calculate segment lengths for all path positions
    segment_vectors = vertices[:, 1:] - vertices[:, :-1]  # Shape: (p, n_segments, 3)
    segment_lengths = np.linalg.norm(segment_vectors, axis=2)  # Shape: (p, n_segments)
    total_lengths = np.sum(segment_lengths, axis=1)  # Shape: (p,)

    # DISTRIBUTE POINTS OVER SEGMENTS #######################################
    # 1. one point per segment for all path positions
    points_per_segment = np.ones((p_len, n_segments), dtype=int)

    # 2. distribute remaining points proportionally to segment lengths
    remaining_points = n_points - n_segments
    if remaining_points > 0:
        # Calculate how many extra points each segment should get for each path position
        proportional_extra = (
            segment_lengths / total_lengths[:, np.newaxis]
        ) * remaining_points
        extra_points = np.round(proportional_extra).astype(int)
        points_per_segment += extra_points

    # Update n_points to actual distributed points for each path position
    actual_n_points = np.sum(points_per_segment, axis=1)
    max_n_points = np.max(actual_n_points)

    # GENERATE MESH AND CVEC ##########################################
    pts_list = []
    cvecs_list = []

    for p_idx in range(p_len):
        n_pts_path = actual_n_points[p_idx]
        parts = np.empty(n_pts_path)
        idx = 0

        for _, n_pts in enumerate(points_per_segment[p_idx]):
            parts[idx : idx + n_pts] = [(2 * j + 1) / (2 * n_pts) for j in range(n_pts)]
            idx += n_pts

        # Generate points for this path position
        pts_path = np.repeat(segment_vectors[p_idx], points_per_segment[p_idx], axis=0)
        pts_path = pts_path * parts[:, np.newaxis]
        pts_path += np.repeat(
            vertices[p_idx, :-1], points_per_segment[p_idx], axis=0
        )  # add starting point of each segment

        cvecs_path = (
            np.repeat(
                segment_vectors[p_idx] / points_per_segment[p_idx, :, np.newaxis],
                points_per_segment[p_idx],
                axis=0,
            )
            * i0[p_idx]
        )

        pts_list.append(pts_path)
        cvecs_list.append(cvecs_path)

    # Convert to arrays with consistent shapes
    # Pad shorter arrays with zeros to match max_n_points
    pts = np.zeros((p_len, max_n_points, 3))
    cvecs = np.zeros((p_len, max_n_points, 3))

    for p_idx in range(p_len):
        n_pts = len(pts_list[p_idx])
        pts[p_idx, :n_pts] = pts_list[p_idx]
        cvecs[p_idx, :n_pts] = cvecs_list[p_idx]

    # If no path variation, return squeezed arrays
    if not has_path_varying:
        pts, cvecs = pts[0], cvecs[0]
        # Remove padding zeros
        if max_n_points > actual_n_points[0]:
            pts = pts[: actual_n_points[0]]
            cvecs = cvecs[: actual_n_points[0]]

    return {"pts": pts, "cvecs": cvecs}


def _create_grid(dimensions, spacing):
    """
    Create a regular cubic grid that covers a cuboid volume defined by dimensions

    We tried with FCC and HCP packing but they do not give significant advantages

    Parameters
    ----------
    dimensions : array-like, shape (6,) - Bounding box [x0, y0, z0, x1, y1, z1]
    spacing : float - Desired lattice constant

    Returns
    -------
    mesh : np.ndarray, shape (n, 3) that covers the given cuboid, ready for inside_masks
    """
    x0, y0, z0, x1, y1, z1 = dimensions

    # Create centered grids to ensure symmetric distribution around origin
    # Calculate number of steps needed to cover the range
    nx = int(np.ceil((x1 - x0) / spacing))
    ny = int(np.ceil((y1 - y0) / spacing))
    nz = int(np.ceil((z1 - z0) / spacing))

    # Create centered grids
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    center_z = (z0 + z1) / 2

    x = center_x + spacing * (np.arange(nx + 1) - nx // 2)
    y = center_y + spacing * (np.arange(ny + 1) - ny // 2)
    z = center_z + spacing * (np.arange(nz + 1) - nz // 2)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _get_tetrahedron_mesh_single(n_points, vertices, magnetization):
    """Helper function to generate a single tetrahedron mesh.

    Parameters
    ----------
    n_points : int
        Target number of mesh points.
    vertices : np.ndarray, shape (4, 3)
        Vertices of the tetrahedron.
    magnetization : np.ndarray, shape (3,)
        Magnetization vector.

    Returns
    -------
    dict
        {"pts": np.ndarray, shape (n, 3), "moments": np.ndarray, shape (n, 3)}
    """
    # Calculate tetrahedron volume
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    v3 = vertices[3] - vertices[0]
    tet_volume = abs(np.linalg.det(np.column_stack([v1, v2, v3]))) / 6.0

    # Return centroid for single point
    if n_points == 1:
        centroid = np.mean(vertices, axis=0)
        pts = np.array([centroid])
        moments = np.array([tet_volume * magnetization])
        return {"pts": pts, "moments": moments}

    # Find the optimal number of subdivisions to get closest to n_points
    def points_for_n_div(n):
        return (n + 1) * (n + 2) * (n + 3) // 6

    n_div_estimate = max(1, int(np.round((6 * n_points) ** (1 / 3) - 1.5)))

    best_n_div = n_div_estimate
    best_diff = abs(points_for_n_div(n_div_estimate) - n_points)
    for test_n_div in range(max(1, n_div_estimate - 2), n_div_estimate + 4):
        test_points = points_for_n_div(test_n_div)
        diff = abs(test_points - n_points)
        if diff < best_diff:
            best_diff = diff
            best_n_div = test_n_div

    n_div = best_n_div

    pts_list = []
    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            for k in range(n_div + 1 - i - j):
                L = n_div - i - j - k
                if L >= 0:
                    u1 = i / n_div
                    u2 = j / n_div
                    u3 = k / n_div
                    u4 = L / n_div
                    point = (
                        u1 * vertices[0]
                        + u2 * vertices[1]
                        + u3 * vertices[2]
                        + u4 * vertices[3]
                    )
                    pts_list.append(point)

    pts = np.array(pts_list)
    volume_per_point = tet_volume / len(pts)
    volumes = np.full(len(pts), volume_per_point)
    moments = volumes[:, np.newaxis] * magnetization
    return {"pts": pts, "moments": moments}


def _target_mesh_tetrahedron(
    n_points: int, vertices: np.ndarray, magnetization: np.ndarray
):
    """
    Generate mesh of tetrahedral body using a uniform barycentric coordinate grid.

    Supports path-enabled inputs: `vertices` may be a single tetrahedron with shape
    (4, 3) or a path of tetrahedra with shape (p, 4, 3). Similarly, `magnetization`
    may be shape (3,) or (p, 3). When path inputs are provided and no per-path
    variation exists, the function returns the same squeezed outputs as before.

    The function creates a uniform grid of points inside the tetrahedron using
    structured barycentric coordinates. The actual number of generated points may
    differ slightly from the target for efficiency.

    Parameters
    ----------
    n_points : int
        Target number of mesh points per tetrahedron.
    vertices : array-like, shape (4, 3) or (p, 4, 3)
        Vertices of the tetrahedron or a path of tetrahedra.
    magnetization : array-like, shape (3,) or (p, 3)
        Magnetization vector for the tetrahedron or per-path magnetizations.

    Returns
    -------
    dict
        {
            "pts": np.ndarray,
                - shape (n, 3) for single tetrahedron input,
                - shape (p, n, 3) for path-enabled inputs where each path
                  produced the same number of mesh points;
            "moments": np.ndarray,
                - shape (n, 3) for single tetrahedron input,
                - shape (p, n, 3) for path-enabled inputs where each path
                  produced the same number of mesh points.
        }

    Notes
    -----
    - For path-enabled inputs, the function generates a mesh per path and stacks
      the results. If different paths produce different numbers of mesh points,
      a ``NotImplementedError`` is raised (this mirrors behavior in other meshers
      that currently require uniform output shapes for stacking).
    - If no path variation is detected, the function preserves the original
      single-body behavior and returns squeezed arrays for backward compatibility.
    """

    # Support path-enabled inputs: vertices can be (4,3) or (p,4,3)
    verts = np.atleast_1d(vertices)
    mags = np.atleast_1d(magnetization)

    # Normalize shapes to (p, 4, 3) and (p, 3)
    verts_exp = verts[np.newaxis, ...] if verts.ndim == 2 else verts
    mags_exp = mags[np.newaxis, ...] if mags.ndim == 1 else mags

    p_len = len(verts_exp)

    # Detect path variation by flattening per-path vertices
    verts_varying = np.unique(verts_exp.reshape(p_len, -1), axis=0).shape[0] > 1
    mags_varying = np.unique(mags_exp, axis=0).shape[0] > 1
    has_path_varying = verts_varying or mags_varying

    # If no path variation, keep original behavior
    if not has_path_varying:
        return _get_tetrahedron_mesh_single(n_points, verts_exp[0], mags_exp[0])

    # Path-varying: use np.unique optimization
    # Flatten vertices per path for uniqueness check: (p, 12)
    verts_flat = verts_exp.reshape(p_len, -1)
    unique_verts_flat, indices = np.unique(verts_flat, axis=0, return_inverse=True)

    unique_pts_list = []
    unique_volumes_list = []

    for verts_flat_i in unique_verts_flat:
        # Reshape back to (4, 3)
        verts_i = verts_flat_i.reshape(4, 3)
        # Use first magnetization for geometry (magnetization will be applied later)
        out = _get_tetrahedron_mesh_single(n_points, verts_i, mags_exp[0])
        pts_i = out["pts"]

        # Calculate volume for this geometry
        v1 = verts_i[1] - verts_i[0]
        v2 = verts_i[2] - verts_i[0]
        v3 = verts_i[3] - verts_i[0]
        tet_volume = abs(np.linalg.det(np.column_stack([v1, v2, v3]))) / 6.0
        volume_per_point = tet_volume / len(pts_i)
        volumes_i = np.full(len(pts_i), volume_per_point)

        unique_pts_list.append(pts_i)
        unique_volumes_list.append(volumes_i)

    # Pad and stack
    max_n = max(len(pts) for pts in unique_pts_list)
    pts_array = np.zeros((p_len, max_n, 3))
    moments_array = np.zeros((p_len, max_n, 3))

    # Reconstruct full path arrays using vectorized assignment
    for i in range(len(unique_verts_flat)):
        # Find all path indices corresponding to this unique geometry
        mask = indices == i
        n = len(unique_pts_list[i])

        # Assign points (broadcasted)
        pts_array[mask, :n] = unique_pts_list[i]

        # Assign moments
        # magnetization[mask] is (k, 3), unique_volumes_list[i] is (n,)
        mags_subset = mags_exp[mask]  # Shape (k, 3)
        vols = unique_volumes_list[i]  # Shape (n,)

        # Broadcast to (k, n, 3)
        moments_array[mask, :n] = mags_subset[:, None, :] * vols[None, :, None]

    return {"pts": pts_array, "moments": moments_array}


def _get_triangularmesh_mesh_single(
    vertices, faces, target_points, volume, magnetization
):
    """Helper function to generate a single triangular mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, 3)
        Mesh vertices.
    faces : np.ndarray, shape (m, 3)
        Mesh faces (triangles) as vertex indices.
    target_points : int
        Target number of mesh points.
    volume : float
        Volume of the body.
    magnetization : np.ndarray, shape (3,)
        Magnetization vector.

    Returns
    -------
    dict
        {"pts": np.ndarray, shape (n, 3), "moments": np.ndarray, shape (n, 3)}
    """
    from magpylib._src.fields.field_BH_triangularmesh import (  # noqa: PLC0415
        _calculate_centroid,
        _mask_inside_trimesh,
    )

    # Return barycenter (centroid) if only one point is requested
    if target_points == 1:
        barycenter = _calculate_centroid(vertices, faces)
        pts = np.array([barycenter])
        moments = np.array([volume * magnetization])
        return {"pts": pts, "moments": moments}

    # Generate regular cubic grid
    spacing = (volume / target_points) ** (1 / 3)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    padding = spacing * 0.5
    dimensions = [
        min_coords[0] - padding,
        min_coords[1] - padding,
        min_coords[2] - padding,
        max_coords[0] + padding,
        max_coords[1] + padding,
        max_coords[2] + padding,
    ]
    points = _create_grid(dimensions, spacing)

    # Apply inside/outside mask
    inside_mask = _mask_inside_trimesh(points, vertices[faces])
    pts = points[inside_mask]

    if len(pts) == 0:
        barycenter = _calculate_centroid(vertices, faces)
        pts = np.array([barycenter])
        volumes = np.full(1, volume)
    else:
        volumes = np.full(len(pts), volume / len(pts))

    moments = volumes[:, np.newaxis] * magnetization
    return {"pts": pts, "moments": moments}


def _target_mesh_triangularmesh(vertices, faces, target_points, volume, magnetization):
    """
    Generate mesh points inside a triangular mesh volume for force computations.

    Uses regular cubic grid generation around the object and applies inside/outside masking
    similar to the sphere approach. When target_points is 1, returns the
    barycenter of the mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, 3) or (p, n, 3)
        Mesh vertices.
    faces : np.ndarray, shape (m, 3)
        Mesh faces (triangles) as vertex indices.
    target_points : int
        Target number of mesh points.
    volume : float | ndarray, shape (p,)
        Volume of the body.
    magnetization : np.ndarray, shape (3,) or (p, 3)
        Magnetization vector for the mesh points.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) or (p, n, 3) - mesh points
        "moments": np.ndarray, shape (n, 3) or (p, n, 3) - moments associated with each point
    }
    """
    # Normalize inputs
    verts = np.atleast_1d(vertices)
    mags = np.atleast_1d(magnetization)
    vols = np.atleast_1d(volume)

    # Normalize shapes
    verts_exp = verts[np.newaxis, ...] if verts.ndim == 2 else verts
    mags_exp = mags[np.newaxis, ...] if mags.ndim == 1 else mags
    vols_exp = vols[np.newaxis] if vols.ndim == 0 else vols

    p_len = len(verts_exp)

    # Detect path variation
    verts_varying = np.unique(verts_exp.reshape(p_len, -1), axis=0).shape[0] > 1
    mags_varying = np.unique(mags_exp, axis=0).shape[0] > 1
    vols_varying = np.unique(vols_exp).shape[0] > 1
    has_path_varying = verts_varying or mags_varying or vols_varying

    # If no path variation, keep original behavior
    if not has_path_varying:
        return _get_triangularmesh_mesh_single(
            verts_exp[0], faces, target_points, vols_exp[0], mags_exp[0]
        )

    # Path-varying: generate per-path meshes and pad
    pts_list = []
    moments_list = []

    for p_idx in range(p_len):
        out = _get_triangularmesh_mesh_single(
            verts_exp[p_idx], faces, target_points, vols_exp[p_idx], mags_exp[p_idx]
        )
        pts_list.append(out["pts"])
        moments_list.append(out["moments"])

    # Pad and stack
    max_n = max(len(pts) for pts in pts_list)
    pts_array = np.zeros((p_len, max_n, 3))
    moments_array = np.zeros((p_len, max_n, 3))

    for p_idx in range(p_len):
        n = len(pts_list[p_idx])
        pts_array[p_idx, :n] = pts_list[p_idx]
        moments_array[p_idx, :n] = moments_list[p_idx]

    return {"pts": pts_array, "moments": moments_array}
