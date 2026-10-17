"""Meshing functions"""

# pylint: disable=too-many-lines
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-function-args

from itertools import pairwise

import numpy as np


def _near_cubic_divisions(sides, meshing):
    """Divisions of a box into about ``meshing`` cells as close to cubes as possible.

    The cell edge is h = (volume / meshing)^(1/3), and each side is cut into
    round(side / h) divisions, at least 1. ``meshing=1`` gives one cell. getFT has
    cut cuboids this way since 5.2.

    Parameters
    ----------
    sides : array-like, shape (3,)
        Side lengths of the box.
    meshing : int
        Target number of cells.

    Returns
    -------
    ndarray, shape (3,)
        Number of divisions along each side.

    Examples
    --------
    >>> print(_near_cubic_divisions((3, 2, 1), 100))
    [8 5 3]
    """
    sides = np.abs(np.asarray(sides, dtype=float))
    if meshing == 1:
        return np.ones(3, dtype=int)
    edge = (np.prod(sides) / meshing) ** (1 / 3)
    return np.maximum(1, np.round(sides / edge)).astype(int)


# CELL MESHERS ################################################################
# A cell mesher splits one magnet geometry (local frame, one path step) into
# cells. It returns a dict with the cell geometry, the cell centroids, and the
# cell volumes. The force computation uses the centroids and volumes; the cell
# geometry is what a cell-based computation such as demagnetization needs.


def _cuboid_cells(dimension, meshing):
    """Split a cuboid into cuboid cells.

    Parameters
    ----------
    dimension : array-like, shape (3,)
        Side lengths (a, b, c).
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the number of divisions (n1, n2, n3) along the
        sides. An integer sets the cell edge h = (a * b * c / meshing)^(1/3) and
        cuts each side into round(side / h) divisions, at least 1, so the cells are
        as close to cubes as possible.

    Returns
    -------
    dict
        ``type`` 'cuboid', cell ``positions`` and ``dimensions`` of shape (n, 3),
        ``centroids`` of shape (n, 3) (the positions), and ``volumes`` of shape (n,).
    """
    dimension = np.asarray(dimension, dtype=float)
    if np.ndim(meshing) == 1:
        divisions = np.array(meshing, dtype=int)
    else:
        divisions = _near_cubic_divisions(dimension, meshing)
    cell_dim = dimension / divisions
    axes = [
        (np.arange(n) + 0.5) * d / n - d / 2
        for d, n in zip(dimension, divisions, strict=True)
    ]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n_cells = len(centers)
    return {
        "type": "cuboid",
        "positions": centers,
        "dimensions": np.tile(cell_dim, (n_cells, 1)),
        "centroids": centers,
        "volumes": np.full(n_cells, np.prod(cell_dim)),
    }


def _sectors_with_centroid_inside(r_in, r_out, span, n):
    """Smallest number of sectors, at least n, that cuts the ring r_in..r_out over
    span degrees so that each sector's centroid lies within a quarter of the ring's
    thickness of its middle radius. The rule of magpylib-material-response, whose
    demagnetization matches each cell's field at its centroid: a wide sector of a
    thin ring has its centroid outside the material.
    """
    if r_in == 0:
        return n
    r_mid = (r_in + r_out) / 2
    tolerance = (r_out - r_in) / 4
    r_bar = 2 / 3 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)
    while True:
        half_angle = np.deg2rad(span / n) / 2
        if r_mid - r_bar * np.sin(half_angle) / half_angle <= tolerance:
            return n
        n += 1


def _cylinder_segment_cells(dimension, meshing):
    """Split a cylinder segment into cylinder-segment cells.

    The segment is unrolled into a box (arc length at the mean radius, r2 - r1, h),
    and that box is cut into (nphi, nr, nh) near-cubic divisions, as a Cuboid is.
    Each ring gets angular divisions in proportion to its middle radius,
    so that all cells have about the same arc length and the rings add up to
    nphi * nr cells. A thin ring gets more sectors where needed, so that each cell's
    centroid lies near the ring's middle radius. When the segment is a full cylinder
    with at least 3 rings, the innermost ring stays one cell per height layer. This
    is the rule of ``mesh_Cylinder`` in magpylib-material-response.

    Parameters
    ----------
    dimension : array-like, shape (5,)
        (r1, r2, h, phi1, phi2) with the angles in degrees. A full cylinder of
        diameter d is (0, d/2, h, 0, 360).
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the divisions (nphi, nr, nh) along the arc at the
        mean radius, the radius and the height.

    Returns
    -------
    dict
        ``type`` 'cylinder_segment', cell ``positions`` of shape (n, 3) on the axis,
        cell ``dimensions`` of shape (n, 5) as (r1, r2, h, phi1, phi2), ``centroids``
        of shape (n, 3), and ``volumes`` of shape (n,).
    """
    r1, r2, h, phi1, phi2 = (float(x) for x in dimension)
    arclen = (r1 + r2) * np.pi * (phi2 - phi1) / 360
    if isinstance(meshing, tuple):
        nphi, nr, nh = meshing
    else:
        nphi, nr, nh = _near_cubic_divisions((arclen, r2 - r1, h), meshing)
    radii = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    zs = (np.arange(nh) + 0.5) * dh - h / 2
    is_full = phi2 - phi1 == 360

    dims, poss = [], []
    for r_in, r_out in pairwise(radii):
        if nr >= 3 and r_in == 0 and is_full:
            nphi_r = 1
        else:
            nphi_r = max(1, round((r_in + r_out) / (r1 + r2) * nphi))
            nphi_r = _sectors_with_centroid_inside(r_in, r_out, phi2 - phi1, nphi_r)
        phis = np.linspace(phi1, phi2, nphi_r + 1)
        ring = np.empty((nh, nphi_r, 5))
        ring[..., 0] = r_in
        ring[..., 1] = r_out
        ring[..., 2] = dh
        ring[..., 3] = phis[:-1]
        ring[..., 4] = phis[1:]
        ring_pos = np.zeros((nh, nphi_r, 3))
        ring_pos[..., 2] = zs[:, np.newaxis]
        dims.append(ring.reshape(-1, 5))
        poss.append(ring_pos.reshape(-1, 3))
    dims = np.concatenate(dims)
    poss = np.concatenate(poss)

    # centroid of an annular sector: on its bisector, at
    # 2/3 * (r2³ - r1³) / (r2² - r1²) * sin(alpha/2) / (alpha/2) from the axis
    r_in, r_out, dh, p_a, p_b = dims.T
    alpha = np.deg2rad(p_b - p_a)
    mid = np.deg2rad((p_a + p_b) / 2)
    r_c = 2 / 3 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)
    r_c = r_c * np.sinc(alpha / (2 * np.pi))
    centroids = poss + np.column_stack(
        [r_c * np.cos(mid), r_c * np.sin(mid), np.zeros_like(r_c)]
    )
    volumes = alpha / 2 * (r_out**2 - r_in**2) * dh
    return {
        "type": "cylinder_segment",
        "positions": poss,
        "dimensions": dims,
        "centroids": centroids,
        "volumes": volumes,
    }


def _tetrahedron_lattice(n_div):
    """Vertex indices of the n_div-fold subdivision of a tetrahedron.

    The barycentric lattice with step 1/n_div cuts a tetrahedron into n_div³ cells of
    equal volume: upward tetrahedra, downward tetrahedra, and octahedra that are cut
    along one diagonal into 4 tetrahedra.

    Returns
    -------
    ndarray, shape (n_div**3, 4, 4)
        Integer barycentric coordinates (summing to n_div) of the 4 vertices of each
        cell.
    """
    ijk = np.indices((n_div, n_div, n_div)).reshape(3, -1).T
    eye = np.eye(4, dtype=int)

    def _bases(level):
        rest = level - ijk.sum(axis=1)
        return np.column_stack([ijk[rest >= 0], rest[rest >= 0]])

    up = _bases(n_div - 1)[:, np.newaxis, :] + eye
    octa = _bases(n_div - 2)
    a, b, c, d, e, f = (
        octa + shift
        for shift in (
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        )
    )
    octa_split = np.concatenate(
        [
            np.stack(quad, axis=1)
            for quad in ((a, f, b, c), (a, f, c, e), (a, f, e, d), (a, f, d, b))
        ]
    )
    down = _bases(n_div - 3)[:, np.newaxis, :] + 1 - eye
    return np.concatenate([up, octa_split, down])


def _tetrahedron_cells(vertices, meshing):
    """Split a tetrahedron into tetrahedron cells of equal volume.

    Parameters
    ----------
    vertices : array-like, shape (4, 3)
        Tetrahedron vertices.
    meshing : int
        Target number of cells. The tetrahedron is split n_div-fold, with n_div³ the
        cube number closest to ``meshing``.

    Returns
    -------
    dict
        ``type`` 'tetrahedron', cell ``vertices`` of shape (n, 4, 3), ``centroids`` of
        shape (n, 3), and ``volumes`` of shape (n,).
    """
    n_div = max(1, int(np.floor(np.cbrt(meshing))))
    if abs((n_div + 1) ** 3 - meshing) < abs(n_div**3 - meshing):
        n_div += 1
    lattice = _tetrahedron_lattice(n_div)
    cell_verts = lattice @ np.asarray(vertices, dtype=float) / n_div
    edges = cell_verts[:, 1:] - cell_verts[:, :1]
    return {
        "type": "tetrahedron",
        "vertices": cell_verts,
        "centroids": cell_verts.mean(axis=1),
        "volumes": np.abs(np.linalg.det(edges)) / 6,
    }


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


def _triangularmesh_cells(vertices, faces, volume, meshing):
    """Split a closed triangular mesh into cube cells.

    A regular grid with pitch (volume / meshing)^(1/3) covers the bounding box, and
    the grid points inside the surface are the cell centers. The cubes are sized so
    that their volumes add up to the body volume. With ``meshing`` of 1, or when no
    grid point falls inside, the one cell sits at the centroid.

    Parameters
    ----------
    vertices : ndarray, shape (n, 3)
        Mesh vertices.
    faces : ndarray, shape (m, 3)
        Triangle face indices.
    volume : float
        Body volume.
    meshing : int
        Target number of cells.

    Returns
    -------
    dict
        ``type`` 'cuboid', cell ``positions`` and ``dimensions`` of shape (n, 3),
        ``centroids`` of shape (n, 3) (the positions), and ``volumes`` of shape (n,).
    """
    from magpylib._src.fields.field_BH_triangularmesh import (  # noqa: PLC0415
        _calculate_centroid,
        _mask_inside_trimesh,
    )

    centers = np.empty((0, 3))
    if meshing > 1:
        spacing = (volume / meshing) ** (1 / 3)
        padding = spacing * 0.5
        box = [
            *(np.min(vertices, axis=0) - padding),
            *(np.max(vertices, axis=0) + padding),
        ]
        points = _create_grid(box, spacing)
        centers = points[_mask_inside_trimesh(points, vertices[faces])]
    if len(centers) == 0:
        centers = np.array([_calculate_centroid(vertices, faces)])

    n_cells = len(centers)
    side = (volume / n_cells) ** (1 / 3)
    return {
        "type": "cuboid",
        "positions": centers,
        "dimensions": np.full((n_cells, 3), side),
        "centroids": centers,
        "volumes": np.full(n_cells, volume / n_cells),
    }


# MAGNET MESHES ###############################################################


def _mesh_magnet(geometry, magnetization, cells_at):
    """Mesh a magnet along its path from its cells.

    Parameters
    ----------
    geometry : ndarray, shape (p, ...)
        Geometry per path step, such as the dimension (p, 3) or the vertices
        (p, 4, 3). Cells are generated once per distinct geometry.
    magnetization : ndarray, shape (p, 3)
        Magnetization per path step.
    cells_at : callable
        Returns the cells dict of path step i, as the cell meshers produce it.

    Returns
    -------
    dict
        ``pts`` the cell centroids and ``moments`` the cell volumes times the
        magnetization, of shape (n, 3) when neither geometry nor magnetization varies
        along the path and (p, n, 3) otherwise, zero-padded when the cell count
        varies; and ``cells``, the cells dict, or a list of p cells dicts when the
        mesh has a path axis.
    """
    p_len = len(geometry)
    flat = np.asarray(geometry).reshape(p_len, -1)
    _, first, inverse = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    cells_unique = [cells_at(int(i)) for i in first]
    magnetization_varying = np.unique(magnetization, axis=0).shape[0] > 1

    if len(first) == 1 and not magnetization_varying:
        cells = cells_unique[0]
        return {
            "pts": cells["centroids"],
            "moments": cells["volumes"][:, np.newaxis] * magnetization[0],
            "cells": cells,
        }

    if len(first) == 1:
        # one geometry, magnetization varies: broadcast instead of looping over p
        cells = cells_unique[0]
        n = len(cells["volumes"])
        return {
            "pts": np.broadcast_to(cells["centroids"], (p_len, n, 3)),
            "moments": cells["volumes"][np.newaxis, :, np.newaxis]
            * magnetization[:, np.newaxis, :],
            "cells": [cells] * p_len,
        }

    cells_path = [cells_unique[g] for g in inverse]
    n_max = max(len(cells["volumes"]) for cells in cells_unique)
    pts = np.zeros((p_len, n_max, 3))
    moments = np.zeros((p_len, n_max, 3))
    for p, (cells, mag) in enumerate(zip(cells_path, magnetization, strict=True)):
        n = len(cells["volumes"])
        pts[p, :n] = cells["centroids"]
        moments[p, :n] = cells["volumes"][:, np.newaxis] * mag
    return {"pts": pts, "moments": moments, "cells": cells_path}


def generate_mesh_cuboid(dimension, magnetization, meshing):
    """Generate the force mesh of a cuboid magnet, see ``_cuboid_cells``.

    Parameters
    ----------
    dimension : ndarray, shape (p, 3)
        Cuboid dimensions (a, b, c) per path position.
    magnetization : ndarray, shape (p, 3)
        Magnetization vector per path position.
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the divisions (n1, n2, n3) along the sides.

    Returns
    -------
    dict
        See ``_mesh_magnet``.
    """
    return _mesh_magnet(
        dimension, magnetization, lambda i: _cuboid_cells(dimension[i], meshing)
    )


def generate_mesh_cylindersegment(r1, r2, h, phi1, phi2, magnetization, meshing):
    """Generate the force mesh of a cylinder segment magnet, see
    ``_cylinder_segment_cells``.

    Parameters
    ----------
    r1, r2, h, phi1, phi2 : ndarray, shape (p,)
        Inner radius, outer radius, height, start angle and end angle (degrees) per
        path position. A full cylinder has r1 = 0, phi1 = 0 and phi2 = 360.
    magnetization : ndarray, shape (p, 3)
        Magnetization vector per path position.
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the divisions (nphi, nr, nh).

    Returns
    -------
    dict
        See ``_mesh_magnet``.
    """
    geometry = np.stack([r1, r2, h, phi1, phi2], axis=1)
    return _mesh_magnet(
        geometry,
        magnetization,
        lambda i: _cylinder_segment_cells(geometry[i], meshing),
    )


def generate_mesh_tetrahedron(vertices, magnetization, meshing):
    """Generate the force mesh of a tetrahedron magnet, see ``_tetrahedron_cells``.

    Parameters
    ----------
    vertices : ndarray, shape (p, 4, 3)
        Tetrahedron vertices per path position.
    magnetization : ndarray, shape (p, 3)
        Magnetization vector per path position.
    meshing : int
        Target number of cells.

    Returns
    -------
    dict
        See ``_mesh_magnet``.
    """
    return _mesh_magnet(
        vertices, magnetization, lambda i: _tetrahedron_cells(vertices[i], meshing)
    )


def generate_mesh_triangularmesh(vertices, faces, volume, magnetization, meshing):
    """Generate the force mesh of a triangular mesh magnet, see
    ``_triangularmesh_cells``.

    Parameters
    ----------
    vertices : ndarray, shape (p, n, 3)
        Mesh vertices per path position.
    faces : ndarray, shape (m, 3)
        Triangle face indices (same for all path positions).
    volume : ndarray, shape (p,)
        Volume per path position.
    magnetization : ndarray, shape (p, 3)
        Magnetization vector per path position.
    meshing : int
        Target number of cells.

    Returns
    -------
    dict
        See ``_mesh_magnet``.
    """
    return _mesh_magnet(
        vertices,
        magnetization,
        lambda i: _triangularmesh_cells(vertices[i], faces, volume[i], meshing),
    )


# CURRENT MESHES ##############################################################


def generate_mesh_circle(diameter, current, target_elems):
    """Generate mesh for circular current loop.

    Parameters
    ----------
    diameter : ndarray, shape (p,)
        Circle diameter per path position.
    current : ndarray, shape (p,)
        Electric current per path position.
    target_elems : int
        Number of mesh elements along circle (must be >= 4).

    Returns
    -------
    dict
        {"pts": ndarray, shape (n, 3) or (p, n, 3),
         "cvecs": ndarray, shape (n, 3) or (p, n, 3)}
    """
    r, i0 = diameter / 2, current
    n = target_elems
    p_len = len(r)
    has_path_varying = (np.unique(r).shape[0] > 1) or (np.unique(i0).shape[0] > 1)

    if not has_path_varying:
        r, i0 = r[:1], i0[:1]
        p_len = 1

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


def generate_mesh_triangle_current(
    triangles: np.ndarray, cds: np.ndarray, n_target: int
):
    """Generate mesh for triangle current sources.

    Refines input triangles using bisection along longest edge until reaching n_target
    mesh points. Returns centroids and current vectors.

    Parameters
    ----------
    triangles : ndarray, shape (p, n, 3, 3)
        Triangle vertices per path position.
    cds : ndarray, shape (p, n, 3)
        Current density vectors per path position.
    n_target : int
        Target number of mesh points (must be >= number of input triangles).

    Returns
    -------
    dict
        {"pts": ndarray, shape (p, m, 3), "cvecs": ndarray, shape (p, m, 3)}
    """
    # Inputs are already path-enabled from the class with shape (p, n, 3, 3) and (p, n, 3)
    p_len = triangles.shape[0]
    n_tria = triangles.shape[1]

    # Check for path-varying parameters. Keep the triangle-geometry dedup
    # (unique rows + inverse map) so the expensive _subdiv runs once per unique
    # geometry instead of once per path step.
    unique_trias, tria_inv = np.unique(triangles, axis=0, return_inverse=True)
    tria_inv = tria_inv.ravel()  # numpy version-robust (2.0 returned (n, 1))
    triangles_varying = unique_trias.shape[0] > 1
    has_path_varying = triangles_varying or (np.unique(cds, axis=0).shape[0] > 1)

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

    # Vectorized subdivision and centroid calculation for all paths.
    # _subdiv depends only on the (path-independent) splits and the per-step
    # geometry, so run it once per unique geometry and scatter back.
    refined_unique = [_subdiv(tria, splits) for tria in unique_trias]
    trias_refined = np.stack(
        [refined_unique[k] for k in tria_inv], axis=0
    )  # Shape: (p, n_refined, 3, 3)

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


def generate_mesh_polyline(vertices, current, target_elems):
    """Generate mesh for polyline current path.

    Distributes target_elems along polyline segments proportionally to segment lengths,
    with at least one point per segment.

    Parameters
    ----------
    vertices : ndarray, shape (p, n, 3)
        Polyline vertices per path position.
    current : ndarray, shape (p,)
        Electric current per path position.
    target_elems : int
        Total number of mesh elements (must be >= number of segments).

    Returns
    -------
    dict
        {"pts": ndarray, shape (p, m, 3), "cvecs": ndarray, shape (p, m, 3)}
    """
    # Inputs are already path-enabled from the class
    p_len = len(current)

    # Check for path-varying parameters
    has_path_varying = (np.unique(vertices, axis=0).shape[0] > 1) or (
        np.unique(current).shape[0] > 1
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
    remaining_points = target_elems - n_segments
    if remaining_points > 0:
        # Calculate how many extra points each segment should get for each path position
        proportional_extra = (
            segment_lengths / total_lengths[:, np.newaxis]
        ) * remaining_points
        extra_points = np.round(proportional_extra).astype(int)
        points_per_segment += extra_points

    # Update target_elems to actual distributed points for each path position
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
            * current[p_idx]
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
