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


def target_mesh_cuboid(n1, n2, n3, a, b, c):
    """
    Cuboid mesh in the local object coordinates.

    Generates a point-cloud of n1 x n2 x n3 points inside a cuboid with sides a, b, c.
    The points are centers of cubical cells that fill the Cuboid.

    Parameters
    ----------
    n1: int - Number of divisions along the x-axis.
    n2: int - Number of divisions along the y-axis.
    n3: int - Number of divisions along the z-axis.
    a: float - Length of the cuboid along the x-axis.
    b: float - Length of the cuboid along the y-axis.
    c: float - Length of the cuboid along the z-axis.

    Returns
    -------
    mesh (np.ndarray, shape (n, 3)), volumes (np.ndarray, shape (n,))
    """    
    xs = np.linspace(-a / 2, a / 2, n1 + 1)
    ys = np.linspace(-b / 2, b / 2, n2 + 1)
    zs = np.linspace(-c / 2, c / 2, n3 + 1)

    dx = xs[1] - xs[0] if len(xs) > 1 else a
    dy = ys[1] - ys[0] if len(ys) > 1 else b
    dz = zs[1] - zs[0] if len(zs) > 1 else c

    xs_cent = xs[:-1] + dx / 2 if len(xs) > 1 else xs + dx / 2
    ys_cent = ys[:-1] + dy / 2 if len(ys) > 1 else ys + dy / 2
    zs_cent = zs[:-1] + dz / 2 if len(zs) > 1 else zs + dz / 2

    mesh = np.array(list(itertools.product(xs_cent, ys_cent, zs_cent)))
    volumes = np.tile(a*b*c/n1/n2/n3, (len(mesh), ))

    return mesh, volumes


def target_mesh_cylinder(r1, r2, h, phi1, phi2, n):
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

    Returns
    -------
    mesh (np.ndarray, shape (n, 3)), volumes (np.ndarray, shape (n,))
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
                volumes.append(np.pi * r[r_ind+1] ** 2 * dh)
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
                        np.pi * (r[r_ind + 1] ** 2 - r[r_ind] ** 2) * dh / nphi_r * (phi2-phi1)/360
                    )
    return np.array(cells), np.array(volumes)


def target_mesh_circle(r, n, i0):
    """
    Circle meshing in the local object coordinates

    Parameters
    ----------
    r: float - Radius of the circle.
    n: int - Number of points along the circle.
    i0: float - electric current

    Returns
    -------
    mesh np.ndarray, shape (n, 3) - central edge positions
    currents np.ndarray, shape (n,) - electric current at each edge
    tvecs np.ndarray, shape (n, 3) - tangent vectors (=edge vectors)
    """

    if n < 3:
        raise ValueError("Number of points must be at least 3 for a circle mesh.")

    # construct polygon with same area as circle
    r1 = r * np.sqrt( (2*np.pi) / (n * np.sin(2*np.pi/n)) )
    vx =  r1 * np.cos(2 * np.pi * np.arange(n+1) / n)
    vy =  r1 * np.sin(2 * np.pi * np.arange(n+1) / n)

    # compute midpoints and tangents of polygon edges
    midx = (vx[:-1] + vx[1:]) / 2
    midy = (vy[:-1] + vy[1:]) / 2
    midz = np.zeros((n,))

    tx = vx[1:] - vx[:-1]
    ty = vy[1:] - vy[:-1]

    mesh = np.column_stack((midx, midy, midz))
    tvecs = np.column_stack((tx, ty, midz))
    currents = np.full(n, i0)

    return mesh, currents, tvecs


def target_mesh_polyline(vertices, i0, meshing):
    """
    Polyline meshing in the local object coordinates

    Parameters
    ----------
    vertices: array_like, shape (n, 3) - vertices of the polyline
    i0: float - electric current
    meshing: int

    If meshing is int, the algorithm trys to distribute these points evenly
    over the polyline, enforcing at least one point per segment.

    Returns
    -------
    mesh np.ndarray, shape (m, 3) - central edge positions
    currents np.ndarray, shape (m,) - electric current at each edge
    tvecs np.ndarray, shape (m, 3) - tangent vectors (=edge vectors)
    """
    if isinstance(meshing, int):
        n_points = meshing

    n_segments = len(vertices) - 1
    
    # Calculate segment lengths
    segment_vectors = vertices[1:] - vertices[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = np.sum(segment_lengths)
    
    # if fewer points than segments
    if n_points < n_segments:
        import warnings
        msg = (
            "Bad meshing input - number of points is less than number of Polyline segments."
            " Setting one point per segment in computation"
        )
        warnings.warn(msg)
        n_points = n_segments

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
        parts[idx:idx+n_pts] = [(2*j + 1) / (2 * n_pts) for j in range(n_pts)]
        idx += n_pts

    mesh = np.repeat(segment_vectors, points_per_segment, axis=0)
    mesh = mesh * parts[:, np.newaxis]
    mesh += np.repeat(vertices[:-1], points_per_segment, axis=0)  # add starting point of each segment

    tvecs = np.repeat(segment_vectors/points_per_segment[:, np.newaxis], points_per_segment, axis=0)

    currents = np.full(n_points, i0)

    return mesh, currents, tvecs


def create_HCP_grid(dimensions, spacing):
    """
    Create an HCP (Hexagonal Close Packed) grid that covers a cuboid volume defined by sides

    Parameters
    ----------
    sides : array_like, shape (3,) - Dimensions of the cuboid (a, b, c) in units of length.
    spacing : float - Desired lattice constant
    
    Returns
    -------
    mesh : np.ndarray, shape (n, 3) that covers the given cuboid, ready for inside_masks
    """
    x0, y0, z0, x1, y1, z1 = dimensions
    center_loc = np.array([(x1+x0), (y1+y0), (z1+z0)])/2
    side_x = x1-x0
    side_y = y1-y0
    side_z = z1-z0
    
    # HCP lattice constants
    h = np.sqrt(2/3) * spacing       # Layer height
    dy = np.sqrt(3)/2 * spacing      # Row spacing in hex grid
    
    # Calculate grid bounds more tightly
    nx = int(np.ceil(side_x / 2 / spacing)) + 1
    ny = int(np.ceil(side_y / 2 / dy)) + 1
    nz = int(np.ceil(side_z / 2 / h)) + 1

    # Generate all layer indices as float to avoid casting
    k = np.arange(-nz, nz+1, dtype=float)
    j = np.arange(-ny, ny+1, dtype=float)
    i = np.arange(-nx, nx+1, dtype=float)

    # Meshgrid for x/y plane (hex lattice)
    jj, ii, kk = np.meshgrid(j, i, k, indexing='ij')

    # Base coordinates
    x = ii * spacing
    y = jj * dy
    z = kk * h

    # Offset every odd row (hex staggering)
    x += (jj % 2) * (spacing / 2)

    # Offset for B-layers (every other layer)
    x += (kk % 2) * (spacing / 2)
    y += (kk % 2) * (dy / 3)

    # Pre-allocate and fill output array efficiently
    n_points = x.size
    pts = np.empty((n_points, 3), dtype=float)
    pts[:, 0] = x.ravel()
    pts[:, 1] = y.ravel()
    pts[:, 2] = z.ravel()

    return pts + center_loc


def target_mesh_sphere(r0, target_points):
    """
    Sphere meshing using HCP grid with filtering
    
    Parameters
    ----------
    r0: float
        Radius of the sphere.
    target_points: int
        Desired number of points in the sphere.
    
    Returns
    -------
    mesh: np.ndarray, shape (n, 3)
    volumes: np.ndarray, shape (n,)
    """
    sphere_volume = (4/3) * np.pi * r0**3
    if target_points == 1:
        # If only one point is requested, return the center of the sphere
        return np.array([[0, 0, 0]]), np.array([sphere_volume])
    
    # Estimate spacing based on target points and HCP packing efficiency
    # HCP packing efficiency is π/(3√2) ≈ 0.74048
    packing_efficiency = np.pi / (3 * np.sqrt(2))
    volume_per_point = sphere_volume / target_points
    spacing = (volume_per_point / packing_efficiency)**(1/3)
    
    # Create bounding box dimensions for sphere
    dimensions = [-r0, -r0, -r0, r0, r0, r0]
    
    # Generate HCP grid covering the bounding box
    points = create_HCP_grid(dimensions, spacing)
    
    # Filter points inside sphere
    mask_sphere = np.linalg.norm(points, axis=1) <= r0
    mesh = points[mask_sphere]
    
    # Calculate volume per point
    n_mesh = len(mesh)
    vols = np.full(n_mesh, sphere_volume / n_mesh)

    return mesh, vols


def points_in_tetrahedron_matrix(points, v0, v1, v2, v3):
    # Pre-compute transformation matrix (inverse)
    T = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        # Degenerate tetrahedron
        return np.zeros(len(points), dtype=bool)
    
    # Transform all points at once
    P = points - v0
    coords = P @ T_inv.T  # (N, 3) - fastest matrix mult
    
    # Ultra-fast boolean operations
    return ((coords >= 0) & (coords <= 1)).all(axis=1) & (coords.sum(axis=1) <= 1)


def target_mesh_tetrahedron(n_points: int, vertices: np.ndarray):
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
    
    Returns
    -------
    mesh : np.ndarray, shape (n, 3)
        Mesh points inside the tetrahedron.
    volumes : np.ndarray, shape (n,)
        Volume associated with each mesh point.
    """
    
    # Calculate tetrahedron volume
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0] 
    v3 = vertices[3] - vertices[0]
    tet_volume = abs(np.linalg.det(np.column_stack([v1, v2, v3]))) / 6.0
    
    if n_points == 1:
        # Return centroid for single point
        centroid = np.mean(vertices, axis=0)
        return np.array([centroid]), np.array([tet_volume])
    
    # Find the optimal number of subdivisions to get closest to n_points
    # For a tetrahedron with n_div divisions, the exact number of points is:
    # (n_div+1)(n_div+2)(n_div+3)/6
    def points_for_n_div(n):
        return (n + 1) * (n + 2) * (n + 3) // 6
    
    # Start with a rough estimate
    n_div_estimate = max(1, int(np.round((6 * n_points)**(1/3) - 1.5)))
    
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
    mesh_points = []
    
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
                    point = u1 * vertices[0] + u2 * vertices[1] + u3 * vertices[2] + u4 * vertices[3]
                    mesh_points.append(point)
    
    mesh_points = np.array(mesh_points)
    
    # Calculate volume per point based on actual number of points generated
    volume_per_point = tet_volume / len(mesh_points)
    volumes = np.full(len(mesh_points), volume_per_point)
    
    return mesh_points, volumes


def target_mesh_triangularmesh(vertices, faces, target_points, volume=None):
    """
    Generate mesh points inside a triangular mesh volume for force computations.
    
    Uses HCP grid generation around the object and applies inside/outside masking
    similar to the sphere approach. When target_points is 1, returns the 
    barycenter of the mesh.
    
    Parameters
    ----------
    vertices : np.ndarray, shape (n, 3) - Mesh vertices
    faces : np.ndarray, shape (m, 3) - Mesh faces (triangles) as vertex indices
    target_points : int - Target number of mesh points
    volume : float - Volume of the body

    Returns
    -------
    mesh : np.ndarray, shape (k, 3) - Mesh points inside the triangular mesh
    volumes : np.ndarray, shape (k,) - Volume associated with each mesh point
    """
    # Import the required functions from triangular mesh field module
    from magpylib._src.fields.field_BH_triangularmesh import (
        calculate_centroid,
        mask_inside_trimesh,
    )

    if target_points == 1:
        # Return barycenter (centroid) for single point
        barycenter = calculate_centroid(vertices, faces)
        return np.array([barycenter]), np.array([volume])
    
    # Generate HCP grid
    # Estimate spacing based on target points and HCP packing efficiency
    packing_efficiency = np.pi / (3 * np.sqrt(2))  # HCP packing efficiency ≈ 0.74048
    volume_per_point = volume / target_points
    spacing = (volume_per_point / packing_efficiency)**(1/3)
    
    # Generate HCP grid covering a bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    padding = spacing * 0.1 # padding to ensure coverage
    dimensions = [
        min_coords[0] - padding, min_coords[1] - padding, min_coords[2] - padding,
        max_coords[0] + padding, max_coords[1] + padding, max_coords[2] + padding
    ]
    points = create_HCP_grid(dimensions, spacing)
    
    # Apply inside/outside mask
    inside_mask = mask_inside_trimesh(points, vertices[faces])
    
    # Filter points that are inside the mesh
    points = points[inside_mask]
    
    if len(points) == 0:
        barycenter = calculate_centroid(vertices, faces)
        return np.array([barycenter]), np.array([volume])
    
    volumes = np.full(len(points), volume / len(points))
    
    return points, volumes




if __name__ == "__main__":
    
    # This function is a placeholder and should be implemented based on the specific
    # requirements for tetrahedral meshing.
    # It should return mesh points and volumes similar to other target mesh functions.

    # # sphere
    #r0 = 0.5
    # dimensions = [-r0]*3 + [r0]*3
    # spacing = 0.1
    #points, vols = target_mesh_sphere(r0, 200)
    #print(f"Number of points in sphere: {len(points)}")

    import pyvista as pv
    import magpylib as magpy

    # Create a complex Pyvista PolyData object using a boolean operation. Start with
    # finer mesh and clean after operation
    sphere = pv.Sphere(radius=0.6)
    dodecahedron = pv.Dodecahedron().triangulate().subdivide(2)

    # Construct magnet from PolyData object
    magnet = magpy.magnet.TriangularMesh.from_pyvista(
        polarization=(0, 0, .1),
        polydata=dodecahedron,
        style_label="magnet",
        meshing = 1000
    )
    points, vols = target_mesh_triangularmesh(magnet.vertices, magnet.faces, target_points=1000, volume=magnet.volume)



    # tetra
    #vertices = np.array([(0,0,0), (1,0,0), (0,1,0), (0,0,1)])
    #    points, vols = target_mesh_tetrahedron(n, vertices)
    #    print(f"Number of points in tetrahedron: {len(points)}")

    # mask_sphere = np.linalg.norm(points, axis=1) <= r0
    # points = points[mask_sphere]
    # print(f"Number of points in sphere: {len(points)}")

    # # tetrahedron
    # dimensions = [min(vertices[:,i]) for i in range(3)] + [max(vertices[:,i]) for i in range(3)]
    # spacing = 0.1
    # points = create_HCP_grid(dimensions, spacing)

    # def points_in_tetrahedron_matrix(points, v0, v1, v2, v3):
    #     """
    #     Matrix operations optimized for maximum speed.
    #     Pre-computes everything possible.
    #     """
    #     # Pre-compute transformation matrix (inverse)
    #     T = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    #     T_inv = np.linalg.inv(T)  # Compute once
        
    #     # Transform all points at once
    #     P = points - v0
    #     coords = P @ T_inv.T  # (N, 3) - fastest matrix mult
        
    #     # Ultra-fast boolean operations
    #     return ((coords >= 0) & (coords <= 1)).all(axis=1) & (coords.sum(axis=1) <= 1)

    # mask_tetrahedron = points_in_tetrahedron_matrix(points, *vertices)
    # points = points[mask_tetrahedron]
    # print(f"Number of points in tetrahedron: {len(points)}")


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    plt.show()
