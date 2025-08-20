import itertools
from itertools import product

import numpy as np


def apportion_triple(triple, min_val=1, max_iter=30):
    """Apportion values of a triple, so that the minimum value `min_val` is respected
    and the product of all values remains the same.
    Example: apportion_triple([1,2,50], min_val=3)
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


def cells_from_dimension(
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
    dim: array_like of length 3
        Dimensions of the object to be divided.
    target_elems: int,
        Total number of elements as target for the procedure. Actual final number is
        likely to differ.
    min_val: int
        Minimum value of the number of divisions per dimension.
    strict_max: bool
        If `True`, the `target_elem` value becomes a strict maximum and the product of
        the resulting triple will be strictly smaller than the target.
    parity: {None, 'odd', 'even'}
        All elements of the resulting triple will match the given parity. If `None`, no
        parity check is performed.

    Returns
    -------
    numpy.ndarray of length 3
        array corresponding of the number of divisions for each dimension

    Examples
    --------
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=True)
    [ 4  9 25]  # Actual total: 900
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=False)
    [ 4  9 26]  # Actual total: 936
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=True)
    [ 3 11 27]  # Actual total: 891
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=False)
    [ 5  7 27]  # Actual total: 945
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=True)
    [ 4  8 26]  # Actual total: 832
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=False)
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
    a, b, c = apportion_triple((a, b, c), min_val=min_val)
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


def target_mesh_cuboid(target_elems, dimension, magnetization):
    """
    Cuboid mesh in the local object coordinates.

    Generates a point-cloud of n1 x n2 x n3 points inside a cuboid with sides a, b, c.
    The points are centers of cubical cells that fill the Cuboid.

    Parameters
    ----------
    target_elems: int or tuple (n1,n2,n3)
        Target number of elements in the mesh. If an integer is provided, it is treated as
        the total number of elements.
    dimension: array_like, shape (3,)
        Dimensions of the cuboid (length, width, height).
    magnetization: np.ndarray, shape (3,)
        Magnetization vector for the mesh points.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) - mesh points
        "moments": np.ndarray, shape (n,3) - moments associated with each point
    }
    """
    a, b, c = dimension

    # Scalar meshing input
    if isinstance(target_elems, int):
        if target_elems == 1:
            n1, n2, n3 = (1, 1, 1)
        else:
            # estimate splitting with aspect ratio~1
            cell_size = (a * b * c / target_elems) ** (1 / 3)
            n1 = max(1, int(np.round(a / cell_size)))
            n2 = max(1, int(np.round(b / cell_size)))
            n3 = max(1, int(np.round(c / cell_size)))
    else:
        n1, n2, n3 = target_elems

    # could improve auto-splitting by reducing the aspect error
    # print(n1*n2*n3)
    # print((a/n1)/(b/n2), (b/n2)/(c/n3), (c/n3)/(a/n1))

    xs = np.linspace(-a / 2, a / 2, n1 + 1)
    ys = np.linspace(-b / 2, b / 2, n2 + 1)
    zs = np.linspace(-c / 2, c / 2, n3 + 1)

    dx = xs[1] - xs[0] if len(xs) > 1 else a
    dy = ys[1] - ys[0] if len(ys) > 1 else b
    dz = zs[1] - zs[0] if len(zs) > 1 else c

    xs_cent = xs[:-1] + dx / 2 if len(xs) > 1 else xs + dx / 2
    ys_cent = ys[:-1] + dy / 2 if len(ys) > 1 else ys + dy / 2
    zs_cent = zs[:-1] + dz / 2 if len(zs) > 1 else zs + dz / 2

    pts = np.array(list(itertools.product(xs_cent, ys_cent, zs_cent)))
    volumes = np.tile(a * b * c / n1 / n2 / n3, (len(pts),))

    moments = volumes[:, np.newaxis] * magnetization

    mesh_dict = {"pts": pts, "moments": moments}

    return mesh_dict


def target_mesh_cylinder(r1, r2, h, phi1, phi2, n, magnetization):
    """
    Cylinder mesh in the local object coordinates.

    Parameters
    ----------
    r1: float
        Inner radius of the cylinder.
    r2: float
        Outer radius of the cylinder.
    h: float
        Height of the cylinder.
    phi1: float
        Start angle of the cylinder in degrees.
    phi2: float
        End angle of the cylinder in degrees.
    n: int
        Number of points in mesh.
    magnetization: np.ndarray, shape (3,)
        Magnetization vector.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) - mesh points
        "moments": np.ndarray, shape (n,3) - moments associated with each point
    }
    """
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    # "unroll" the cylinder and distribute the target number of elements along the
    # circumference, radius and height.
    nphi, nr, nh = cells_from_dimension(dim, n)

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

    moments = volumes[:, np.newaxis] * magnetization

    mesh_dict = {"pts": pts, "moments": moments}
    return mesh_dict


def target_mesh_circle(r, n, i0):
    """
    Circle meshing in the local object coordinates

    Parameters
    ----------
    r: float - Radius of the circle.
    n: int >= 4 - Number of points along the circle.
    i0: float - electric current

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) - central edge positions
        "currents": np.ndarray, shape (n,) - electric current at each edge
        "tvecs": np.ndarray, shape (n, 3) - tangent vectors (=edge vectors)
    }
    """
    # construct polygon with same area as circle
    r1 = r * np.sqrt((2 * np.pi) / (n * np.sin(2 * np.pi / n)))
    vx = r1 * np.cos(2 * np.pi * np.arange(n + 1) / n)
    vy = r1 * np.sin(2 * np.pi * np.arange(n + 1) / n)

    # compute midpoints and tangents of polygon edges
    midx = (vx[:-1] + vx[1:]) / 2
    midy = (vy[:-1] + vy[1:]) / 2
    midz = np.zeros((n,))

    tx = vx[1:] - vx[:-1]
    ty = vy[1:] - vy[:-1]

    pts = np.column_stack((midx, midy, midz))
    tvecs = np.column_stack((tx, ty, midz))
    currents = np.full(n, i0)

    mesh_dict = {"pts": pts, "currents": currents, "tvecs": tvecs}

    return mesh_dict


def target_mesh_polyline(vertices, i0, n_points):
    """
    Polyline meshing in the local object coordinates

    Parameters
    ----------
    vertices: array_like, shape (n, 3) - vertices of the polyline
    i0: float - electric current
    n_points: int >= n_segments

    If n_points is int, the algorithm trys to distribute these points evenly
    over the polyline, enforcing at least one point per segment.

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (m, 3) - central segment positions
        "currents": np.ndarray, shape (m,) - electric current at each segment
        "tvecs": np.ndarray, shape (m, 3) - tangent vectors (=segment vectors)
    }
    """
    n_segments = len(vertices) - 1

    # Calculate segment lengths
    segment_vectors = vertices[1:] - vertices[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = np.sum(segment_lengths)

    # DISTRIBUTE POINTS OVER SEGMENTS #######################################
    # 1. one point per segment
    points_per_segment = np.ones(n_segments, dtype=int)

    # 2. distribute remaining points proportionally to segment lengths
    remaining_points = n_points - n_segments
    if remaining_points > 0:
        # Calculate how many extra points each segment should get
        proportional_extra = (segment_lengths / total_length) * remaining_points
        extra_points = np.round(proportional_extra).astype(int)
        points_per_segment += extra_points

        # possibly there will now be n_segments too much or too few points
        n_points = np.sum(points_per_segment)

    # GENERATE MESH AND TVEC ##########################################
    parts = np.empty(n_points)
    idx = 0
    for n_pts in points_per_segment:
        parts[idx : idx + n_pts] = [(2 * j + 1) / (2 * n_pts) for j in range(n_pts)]
        idx += n_pts

    pts = np.repeat(segment_vectors, points_per_segment, axis=0)
    pts = pts * parts[:, np.newaxis]
    pts += np.repeat(
        vertices[:-1], points_per_segment, axis=0
    )  # add starting point of each segment

    tvecs = np.repeat(
        segment_vectors / points_per_segment[:, np.newaxis], points_per_segment, axis=0
    )

    currents = np.full(n_points, i0)

    mesh_dict = {"pts": pts, "currents": currents, "tvecs": tvecs}

    return mesh_dict


def create_grid(dimensions, spacing):
    """
    Create a regular cubic grid that covers a cuboid volume defined by dimensions

    We tried with FCC and HCP packing but they do not give significant advantages

    Parameters
    ----------
    dimensions : array_like, shape (6,) - Bounding box [x0, y0, z0, x1, y1, z1]
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

    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    return points


def target_mesh_tetrahedron(
    n_points: int, vertices: np.ndarray, magnetization: np.ndarray
):
    """
    Generate mesh of tetrahedral body using uniform barycentric coordinate grid.

    This function creates a uniform grid of points inside the tetrahedron using
    structured barycentric coordinates, ensuring homogeneous density distribution.
    The actual number of points may differ slightly from the target for efficiency.

    Parameters
    ----------
    n_points : int
        Target number of mesh points.
    vertices : array_like, shape (4, 3)
        Vertices of the tetrahedron.
    magnetization : array_like, shape (3,)
        Magnetization vector.

    Returns
    -------
     dict: {
        "pts": np.ndarray, shape (n, 3) - mesh points
        "moments": np.ndarray, shape (n,3) - moments associated with each point
    }
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

        mesh_dict = {"pts": pts, "moments": moments}
        return mesh_dict

    # Find the optimal number of subdivisions to get closest to n_points
    # For a tetrahedron with n_div divisions, the exact number of points is:
    # (n_div+1)(n_div+2)(n_div+3)/6
    def points_for_n_div(n):
        return (n + 1) * (n + 2) * (n + 3) // 6

    # Start with a rough estimate
    n_div_estimate = max(1, int(np.round((6 * n_points) ** (1 / 3) - 1.5)))

    # Test a few values around the estimate to find the closest match
    best_n_div = n_div_estimate
    best_diff = abs(points_for_n_div(n_div_estimate) - n_points)

    for test_n_div in range(max(1, n_div_estimate - 2), n_div_estimate + 4):
        test_points = points_for_n_div(test_n_div)
        diff = abs(test_points - n_points)
        if diff < best_diff:
            best_diff = diff
            best_n_div = test_n_div

    n_div = best_n_div

    # Generate structured barycentric coordinates
    pts_list = []

    # Create uniform grid in barycentric coordinates
    # We need u1 + u2 + u3 + u4 = 1 and all ui >= 0
    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            for k in range(n_div + 1 - i - j):
                l = n_div - i - j - k
                if l >= 0:
                    # Barycentric coordinates (normalized)
                    u1 = i / n_div
                    u2 = j / n_div
                    u3 = k / n_div
                    u4 = l / n_div

                    # Convert to Cartesian coordinates
                    point = (
                        u1 * vertices[0]
                        + u2 * vertices[1]
                        + u3 * vertices[2]
                        + u4 * vertices[3]
                    )
                    pts_list.append(point)

    pts = np.array(pts_list)

    # Calculate volume per point based on actual number of points generated
    volume_per_point = tet_volume / len(pts)
    volumes = np.full(len(pts), volume_per_point)

    moments = volumes[:, np.newaxis] * magnetization

    mesh_dict = {"pts": pts, "moments": moments}

    return mesh_dict


def target_mesh_triangularmesh(vertices, faces, target_points, volume, magnetization):
    """
    Generate mesh points inside a triangular mesh volume for force computations.

    Uses regular cubic grid generation around the object and applies inside/outside masking
    similar to the sphere approach. When target_points is 1, returns the
    barycenter of the mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, 3) - Mesh vertices
    faces : np.ndarray, shape (m, 3) - Mesh faces (triangles) as vertex indices
    target_points : int - Target number of mesh points
    volume : float - Volume of the body
    magnetization : np.ndarray, shape (3,) - Magnetization vector for the mesh points

    Returns
    -------
    dict: {
        "pts": np.ndarray, shape (n, 3) - mesh points
        "moments": np.ndarray, shape (n,3) - moments associated with each point
    }
    """
    # Import the required functions from triangular mesh field module
    from magpylib._src.fields.field_BH_triangularmesh import (
        calculate_centroid,
        mask_inside_trimesh,
    )

    # Return barycenter (centroid) if only one point is requested
    if target_points == 1:
        barycenter = calculate_centroid(vertices, faces)
        pts = np.array([barycenter])
        moments = np.array([volume * magnetization])

        mesh_dict = {"pts": pts, "moments": moments}
        return mesh_dict

    # Generate regular cubic grid
    spacing = (volume / target_points) ** (1 / 3)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    padding = (
        spacing * 0.5
    )  # add half a cell size padding to ensure optimal homo loc-vol matching
    dimensions = [
        min_coords[0] - padding,
        min_coords[1] - padding,
        min_coords[2] - padding,
        max_coords[0] + padding,
        max_coords[1] + padding,
        max_coords[2] + padding,
    ]
    points = create_grid(dimensions, spacing)

    # Apply inside/outside mask
    inside_mask = mask_inside_trimesh(points, vertices[faces])
    pts = points[inside_mask]

    if len(pts) == 0:
        barycenter = calculate_centroid(vertices, faces)
        pts = np.array([barycenter])
        volumes = np.array([volume])
        mesh_dict = {"pts": pts, "volumes": volumes}
        return mesh_dict

    # Volumes
    volumes = np.full(len(pts), volume / len(pts))

    moments = volumes[:, np.newaxis] * magnetization

    mesh_dict = {"pts": pts, "moments": moments}

    return mesh_dict


if __name__ == "__main__":
    for n in [1, 10, 50, 100, 500, 1000, 5000]:
        target_mesh_cuboid(n, (1.1, 2.2, 1))
