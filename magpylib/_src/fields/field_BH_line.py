"""
Implementations of analytical expressions of line current segments
"""
import numpy as np
from numpy.linalg import norm

from magpylib._src.input_checks import check_field_input


def field_BH_line_from_vert(
    field: str,
    observers: np.ndarray,
    current: np.ndarray,
    vertices: list,  # list of mix3 ndarrays
) -> np.ndarray:
    """
    This function accepts n (mi,3) shaped vertex-sets, creates a single long
    input array for field_BH_line(), computes, sums and returns a single field for each
    vertex-set at respective n observer positions.

    ### Args:
    - bh (boolean): True=B, False=H
    - current (ndarray n): current on line in units of [A]
    - vertex_sets (list of len n): n vertex sets (each of shape (mi,3))
    - pos_obs (ndarray nx3): n observer positions in units of [mm]

    ### Returns:
    - B-field (ndarray nx3): B-field vectors at pos_obs in units of mT
    """

    nv = len(vertices)  # number of input vertex_sets
    npp = int(observers.shape[0] / nv)  # number of position vectors
    nvs = [len(vset) - 1 for vset in vertices]  # length of vertex sets
    nseg = sum(nvs)  # number of segments

    # vertex_sets -> segments
    curr_tile = np.repeat(current, nvs)
    pos_start = np.concatenate([vert[:-1] for vert in vertices])
    pos_end = np.concatenate([vert[1:] for vert in vertices])

    # create input for vectorized computation in one go
    observers = np.reshape(observers, (nv, npp, 3))
    observers = np.repeat(observers, nvs, axis=0)
    observers = np.reshape(observers, (-1, 3))

    curr_tile = np.repeat(curr_tile, npp)
    pos_start = np.repeat(pos_start, npp, axis=0)
    pos_end = np.repeat(pos_end, npp, axis=0)

    # compute field
    field = current_line_field(field, observers, curr_tile, pos_start, pos_end)
    field = np.reshape(field, (nseg, npp, 3))

    # sum for each vertex set
    ns_cum = [sum(nvs[:i]) for i in range(nv + 1)]  # cumulative index positions
    field_sum = np.array(
        [np.sum(field[ns_cum[i - 1] : ns_cum[i]], axis=0) for i in range(1, nv + 1)]
    )

    return np.reshape(field_sum, (-1, 3))


# ON INTERFACE
def current_line_field(
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
        If `field='B'` return B-field in units of [mT], if `field='H'` return H-field
        in units of [kA/m].

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of [mm].

    current: ndarray, shape (n,)
        Electrical current in units of [A].

    start: ndarray, shape (n,3)
        Line start positions (x,y,z) in Cartesian coordinates in units of [mm].

    end: ndarray, shape (n,3)
        Line end positions (x,y,z) in Cartesian coordinates in units of [mm].

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of current in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

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
    >>> B = magpy.core.current_line_field('B', obs, curr, start, end)
    >>> print(B)
    [[ 0.         -0.14142136  0.        ]
     [ 0.          0.          0.        ]]

    Notes
    -----
    Field computation via law of Biot Savart. See also countless online ressources.
    eg. http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
    """
    # pylint: disable=too-many-statements

    bh = check_field_input(field, "current_line_field()")

    # allocate for special case treatment
    ntot = len(current)
    field_all = np.zeros((ntot, 3))

    # Check for zero-length segments
    mask0 = np.all(segment_start == segment_end, axis=1)
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
