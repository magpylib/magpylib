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
    n1: int
        Number of divisions along the x-axis.
    n2: int
        Number of divisions along the y-axis.
    n3: int
        Number of divisions along the z-axis.
    a: float
        Length of the cuboid along the x-axis.
    b: float
        Length of the cuboid along the y-axis.
    c: float
        Length of the cuboid along the z-axis.

    Returns
    -------
    np.ndarray, shape (n1*n2*n3, 3)
    
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

    return np.array(list(itertools.product(xs_cent, ys_cent, zs_cent)))



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
    np.ndarray, shape (n, 3)
    """
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    # "unroll" the cylinder and distribute the target number of elements along the
    # circumference, radius and height.
    nphi, nr, nh = cells_from_dimension(dim, n)

    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cells = []
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
    # return _collection_from_obj_and_cells(cylinder, cells, **kwargs)
    return np.array(cells)


def target_mesh_circle(r, n):
    """
    Circle mesh in the local object coordinates

    Parameters
    ----------
    r: float
        Radius of the circle.
    n: int
        Number of points along the circle.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """

    if n < 3:
        raise ValueError("Number of points must be at least 3 for a circle mesh.")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Create circle points in x-y plane
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    z = np.zeros_like(x)

    return np.column_stack([x, y, z])
