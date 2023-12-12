"""
Implementations of analytical expressions of line current segments
"""
import warnings

import numpy as np
from numpy.linalg import norm

from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.input_checks import check_field_input


def current_vertices_field(
    field: str,
    observers: np.ndarray,
    current: np.ndarray,
    vertices: np.ndarray = None,
    segment_start=None,  # list of mix3 ndarrays
    segment_end=None,
) -> np.ndarray:
    """
    This function accepts n (mi,3) shaped vertex-sets, creates a single long
    input array for field_BH_polyline(), computes, sums and returns a single field for each
    vertex-set at respective n observer positions.

    ### Args:
    - bh (boolean): True=B, False=H
    - current (ndarray n): current on line in units of A
    - vertex_sets (list of len n): n vertex sets (each of shape (mi,3))
    - pos_obs (ndarray nx3): n observer positions in units of mm

    ### Returns:
    - B-field (ndarray nx3): B-field vectors at pos_obs in units of mT
    """
    if vertices is None:
        return current_polyline_field(
            field, observers, current, segment_start, segment_end
        )

    nvs = np.array([f.shape[0] for f in vertices])  # lengths of vertices sets
    if all(v == nvs[0] for v in nvs):  # if all vertices sets have the same lenghts
        n0, n1, *_ = vertices.shape
        BH = current_polyline_field(
            field=field,
            observers=np.repeat(observers, n1 - 1, axis=0),
            current=np.repeat(current, n1 - 1, axis=0),
            segment_start=vertices[:, :-1].reshape(-1, 3),
            segment_end=vertices[:, 1:].reshape(-1, 3),
        )
        BH = BH.reshape((n0, n1 - 1, 3))
        BH = np.sum(BH, axis=1)
    else:
        split_indices = np.cumsum(nvs - 1)[:-1]  # remove last to avoid empty split
        BH = current_polyline_field(
            field=field,
            observers=np.repeat(observers, nvs - 1, axis=0),
            current=np.repeat(current, nvs - 1, axis=0),
            segment_start=np.concatenate([vert[:-1] for vert in vertices]),
            segment_end=np.concatenate([vert[1:] for vert in vertices]),
        )
        bh_split = np.split(BH, split_indices)
        BH = np.array([np.sum(bh, axis=0) for bh in bh_split])
    return BH


# CORE
def current_polyline_field(
    field: str,
    observers: np.ndarray,
    current: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> np.ndarray:
    """Magnetic field of line current segments.

    The current flows from start to end positions. The field is set to (0,0,0) on a
    line segment.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    current: ndarray, shape (n,)
        Electrical current in units of A.

    start: ndarray, shape (n,3)
        Polyline start positions (x,y,z) in Cartesian coordinates in units of mm.

    end: ndarray, shape (n,3)
        Polyline end positions (x,y,z) in Cartesian coordinates in units of mm.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of current in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Examples
    --------
    Compute the field of two segments. The 2nd observer lies on the segment
    so that [0  0  0] is returned.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> curr = np.array([1,2])
    >>> start = np.array([(-1,0,0), (-1,0,0)])
    >>> end   = np.array([( 1,0,0), ( 2,0,0)])
    >>> obs   = np.array([( 0,0,1), ( 0,0,0)])
    >>> B = magpy.core.current_polyline_field('B', obs, curr, start, end)
    >>> print(B)
    [[ 0.         -0.14142136  0.        ]
     [ 0.          0.          0.        ]]

    Notes
    -----
    Field computation via law of Biot Savart. See also countless online resources.
    eg. http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
    """
    # pylint: disable=too-many-statements
    bh = check_field_input(field, "current_polyline_field()")

    # allocate for special case treatment
    ntot = len(current)
    field_all = np.zeros((ntot, 3))

    # Check for zero-length segments (or discontinuous)
    mask_nan_start = np.isnan(segment_start).all(axis=1)
    mask_nan_end = np.isnan(segment_end).all(axis=1)
    mask_equal = np.all(segment_start == segment_end, axis=1)
    mask0 = mask_equal | mask_nan_start | mask_nan_end

    if np.all(mask0):
        return field_all

    # continue only with non-zero segments
    if np.any(mask0):
        not_mask0 = ~mask0  # avoid multiple computation of ~mask
        current = current[not_mask0]
        segment_start = segment_start[not_mask0]
        segment_end = segment_end[not_mask0]
        observers = observers[not_mask0]

    # rename
    p1, p2, po = segment_start, segment_end, observers

    # make dimensionless (avoid all large/small input problems) by introducing
    # the segment length as characteristic length scale.
    norm_12 = norm(p1 - p2, axis=1)
    p1 = (p1.T / norm_12).T
    p2 = (p2.T / norm_12).T
    po = (po.T / norm_12).T

    # p4 = projection of pos_obs onto line p1-p2
    t = np.sum((po - p1) * (p1 - p2), axis=1)
    p4 = p1 + (t * (p1 - p2).T).T

    # distance of observers from line
    norm_o4 = norm(po - p4, axis=1)

    # separate on-line cases (-> B=0)
    mask1 = norm_o4 < 1e-15  # account for numerical issues
    if np.all(mask1):
        return field_all

    # continue only with general off-line cases
    if np.any(mask1):
        not_mask1 = ~mask1
        po = po[not_mask1]
        p1 = p1[not_mask1]
        p2 = p2[not_mask1]
        p4 = p4[not_mask1]
        norm_12 = norm_12[not_mask1]
        norm_o4 = norm_o4[not_mask1]
        current = current[not_mask1]

    # determine field direction
    cros = np.cross(p2 - p1, po - p4)
    norm_cros = norm(cros, axis=1)
    eB = (cros.T / norm_cros).T

    # compute angles
    norm_o1 = norm(
        po - p1, axis=1
    )  # improve performance by computing all norms at once
    norm_o2 = norm(po - p2, axis=1)
    norm_41 = norm(p4 - p1, axis=1)
    norm_42 = norm(p4 - p2, axis=1)
    sinTh1 = norm_41 / norm_o1
    sinTh2 = norm_42 / norm_o2
    deltaSin = np.empty((len(po),))

    # determine how p1,p2,p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = (norm_41 > 1) * (norm_41 > norm_42)
    deltaSin[mask2] = abs(sinTh1[mask2] - sinTh2[mask2])
    # both points above
    mask3 = (norm_42 > 1) * (norm_42 > norm_41)
    deltaSin[mask3] = abs(sinTh2[mask3] - sinTh1[mask3])
    # one above one below or one equals p4
    mask4 = ~mask2 * ~mask3
    deltaSin[mask4] = abs(sinTh1[mask4] + sinTh2[mask4])

    field = (deltaSin / norm_o4 * eB.T / norm_12 * current / 10).T  # m->mm, T->mT

    # broadcast general case results into allocated vector
    mask0[~mask0] = mask1
    field_all[~mask0] = field

    # return B or H
    if bh:
        return field_all

    # H: mT -> kA/m
    return field_all * 10 / 4 / np.pi


def current_line_field(*args, **kwargs):
    """current_line_field is deprecated, see current_polyline_field"""

    warnings.warn(
        (
            "current_line_field is deprecated and will be removed in a future version, "
            "use current_polyline_field instead."
        ),
        MagpylibDeprecationWarning,
        stacklevel=2,
    )
    return current_polyline_field(*args, **kwargs)
