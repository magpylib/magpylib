"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Cuboids. Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input


def magnet_cuboid_field(
    field: str, observers: np.ndarray, magnetization: np.ndarray, dimension: np.ndarray
) -> np.ndarray:
    """Magnetic field of a homogeneously magnetized cuboid.

    The cuboid sides are parallel to the coordinate axes. The geometric center of the
    cuboid lies in the origin.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of mT.

    dimension: ndarray, shape (n,3)
        Cuboid side lengths in units of mm.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Examples
    --------
    Compute the field of three different instances.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> mag = np.array([(222,333,555), (33,44,55), (0,0,100)])
    >>> dim = np.array([(1,1,1), (2,3,4), (1,2,3)])
    >>> obs = np.array([(1,2,3), (2,3,4), (0,0,0)])
    >>> B = magpy.core.magnet_cuboid_field('B', obs, mag, dim)
    >>> print(B)
    [[ 0.49343022  1.15608356  1.65109312]
     [ 0.82221622  1.18511282  1.46945423]
     [ 0.          0.         88.77487579]]

    Notes
    -----
    Field computations via magnetic surface charge density. Published
    several times with similar expressions:

    Yang: Superconductor Science and Technology 3(12):591 (1999)

    Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4 (2005)

    Camacho: Revista Mexicana de Fisica E 59 (2013) 8–17

    Avoiding indeterminate forms:

    In the above implementations there are several indeterminate forms
    where the limit must be taken. These forms appear at positions
    that are extensions of the edges in all xyz-octants except bottQ4.
    In the vicinity of these indeterminate forms the formula becomes
    numerically instable.

    Chosen solution: use symmetries of the problem to change all
    positions to their bottQ4 counterparts. see also

    Cichon: IEEE Sensors Journal, vol. 19, no. 7, April 1, 2019, p.2509
    """
    # pylint: disable=too-many-statements

    bh = check_field_input(field, "magnet_cuboid_field()")

    magx, magy, magz = magnetization.T
    a, b, c = np.abs(dimension.T) / 2
    x, y, z = observers.T

    # This implementation is completely scale invariant as only observer/dimension
    # ratios appear in equations below.

    # dealing with special cases -----------------------------------

    # allocate B with zeros
    B_all = np.zeros((len(magx), 3))

    # SPECIAL CASE 1: mag = (0,0,0)
    mask1 = (magx == 0) * (magy == 0) * (magz == 0)  # 2x faster than np.all()

    # SPECIAL CASE 2: 0 in dimension
    mask2 = (a * b * c).astype(bool)

    # SPECIAL CASE 3: observer lies on-edge/corner
    # -> 1e-15 to account for numerical imprecision when e.g. rotating
    # -> /a /b /c to account for the "missing" scaling (1e-15 is large when
    #    a is e.g. 1e-15 itself)

    mx1 = abs(abs(x) - a) < 1e-15 * a  # on surface
    my1 = abs(abs(y) - b) < 1e-15 * b  # on surface
    mz1 = abs(abs(z) - c) < 1e-15 * c  # on surface

    mx2 = (abs(x) - a) < 1e-15 * a  # within cuboid dimension
    my2 = (abs(y) - b) < 1e-15 * b  # within cuboid dimension
    mz2 = (abs(z) - c) < 1e-15 * c  # within cuboid dimension

    mask_xedge = my1 & mz1 & mx2
    mask_yedge = mx1 & mz1 & my2
    mask_zedge = mx1 & my1 & mz2
    mask3 = mask_xedge | mask_yedge | mask_zedge

    # on-wall is not a special case

    # continue only with general cases ----------------------------
    mask_gen = ~mask1 & mask2 & ~mask3

    if np.any(mask_gen):
        magx, magy, magz = magnetization[mask_gen].T
        a, b, c = dimension[mask_gen].T / 2
        x, y, z = np.copy(observers[mask_gen]).T

        # avoid indeterminate forms by evaluating in bottQ4 only --------
        # basic masks
        maskx = x < 0
        masky = y > 0
        maskz = z > 0

        # change all positions to their bottQ4 counterparts
        x[maskx] = x[maskx] * -1
        y[masky] = y[masky] * -1
        z[maskz] = z[maskz] * -1

        # create sign flips for position changes
        qsigns = np.ones((len(magx), 3, 3))
        qs_flipx = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
        qs_flipy = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        qs_flipz = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
        # signs flips can be applied subsequently
        qsigns[maskx] = qsigns[maskx] * qs_flipx
        qsigns[masky] = qsigns[masky] * qs_flipy
        qsigns[maskz] = qsigns[maskz] * qs_flipz

        # field computations --------------------------------------------
        # Note: in principle the computation for all three mag-components can be
        #   vectorized itself using symmetries. However, tiling the three
        #   components will cost more than is gained by the vectorized evaluation

        # Note: making the following computation steps is not necessary
        #   as mkl will cache such small computations
        xma, xpa = x - a, x + a
        ymb, ypb = y - b, y + b
        zmc, zpc = z - c, z + c

        xma2, xpa2 = xma**2, xpa**2
        ymb2, ypb2 = ymb**2, ypb**2
        zmc2, zpc2 = zmc**2, zpc**2

        mmm = np.sqrt(xma2 + ymb2 + zmc2)
        pmp = np.sqrt(xpa2 + ymb2 + zpc2)
        pmm = np.sqrt(xpa2 + ymb2 + zmc2)
        mmp = np.sqrt(xma2 + ymb2 + zpc2)
        mpm = np.sqrt(xma2 + ypb2 + zmc2)
        ppp = np.sqrt(xpa2 + ypb2 + zpc2)
        ppm = np.sqrt(xpa2 + ypb2 + zmc2)
        mpp = np.sqrt(xma2 + ypb2 + zpc2)

        with np.errstate(divide="ignore", invalid="ignore"):
            ff2x = np.log(
                (xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp)
            ) - np.log((xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp))

            ff2y = np.log(
                (-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp)
            ) - np.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

            ff2z = np.log(
                (-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp)
            ) - np.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

        ff1x = (
            np.arctan2((ymb * zmc), (xma * mmm))
            - np.arctan2((ymb * zmc), (xpa * pmm))
            - np.arctan2((ypb * zmc), (xma * mpm))
            + np.arctan2((ypb * zmc), (xpa * ppm))
            - np.arctan2((ymb * zpc), (xma * mmp))
            + np.arctan2((ymb * zpc), (xpa * pmp))
            + np.arctan2((ypb * zpc), (xma * mpp))
            - np.arctan2((ypb * zpc), (xpa * ppp))
        )

        ff1y = (
            np.arctan2((xma * zmc), (ymb * mmm))
            - np.arctan2((xpa * zmc), (ymb * pmm))
            - np.arctan2((xma * zmc), (ypb * mpm))
            + np.arctan2((xpa * zmc), (ypb * ppm))
            - np.arctan2((xma * zpc), (ymb * mmp))
            + np.arctan2((xpa * zpc), (ymb * pmp))
            + np.arctan2((xma * zpc), (ypb * mpp))
            - np.arctan2((xpa * zpc), (ypb * ppp))
        )

        ff1z = (
            np.arctan2((xma * ymb), (zmc * mmm))
            - np.arctan2((xpa * ymb), (zmc * pmm))
            - np.arctan2((xma * ypb), (zmc * mpm))
            + np.arctan2((xpa * ypb), (zmc * ppm))
            - np.arctan2((xma * ymb), (zpc * mmp))
            + np.arctan2((xpa * ymb), (zpc * pmp))
            + np.arctan2((xma * ypb), (zpc * mpp))
            - np.arctan2((xpa * ypb), (zpc * ppp))
        )

        # contributions from x-magnetization
        bx_magx = (
            magx * ff1x * qsigns[:, 0, 0]
        )  # the 'missing' third sign is hidden in ff1x
        by_magx = magx * ff2z * qsigns[:, 0, 1]
        bz_magx = magx * ff2y * qsigns[:, 0, 2]
        # contributions from y-magnetization
        bx_magy = magy * ff2z * qsigns[:, 1, 0]
        by_magy = magy * ff1y * qsigns[:, 1, 1]
        bz_magy = -magy * ff2x * qsigns[:, 1, 2]
        # contributions from z-magnetization
        bx_magz = magz * ff2y * qsigns[:, 2, 0]
        by_magz = -magz * ff2x * qsigns[:, 2, 1]
        bz_magz = magz * ff1z * qsigns[:, 2, 2]

        # summing all contributions
        bx_tot = bx_magx + bx_magy + bx_magz
        by_tot = by_magx + by_magy + by_magz
        bz_tot = bz_magx + bz_magy + bz_magz

        # B = np.c_[bx_tot, by_tot, bz_tot]      # faster for 10^5 and more evaluations
        B = np.concatenate(((bx_tot,), (by_tot,), (bz_tot,)), axis=0).T

        # combine with special edge/corner cases
        B_all[mask_gen] = B

    B = B_all / (4 * np.pi)

    # return B or compute and return H -------------
    if bh:
        return B

    # if inside magnet subtract magnetization vector
    mask_inside = mx2 & my2 & mz2
    B[mask_inside] -= magnetization[mask_inside]
    H = B * 10 / 4 / np.pi  # mT -> kA/m
    return H
