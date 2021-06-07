"""
Implementations of analytical expressions of line current segments
"""

import numpy as np
from numpy.linalg import norm
from magpylib._lib.config import Config


def field_BH_line_from_vert(
    bh: bool,
    current: np.ndarray,
    vertex_sets: list,  # list of mix3 ndarrays
    pos_obs: np.ndarray
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

    nv = len(vertex_sets)           # number of input vertex_sets
    npp = int(pos_obs.shape[0]/nv)  # number of position vectors
    nvs = [len(vset)-1 for vset in vertex_sets] # length of vertex sets
    nseg = sum(nvs)                             # number of segments

    # vertex_sets -> segements
    curr_tile = np.repeat(current, nvs)
    pos_start = np.concatenate([vert[:-1] for vert in vertex_sets])
    pos_end = np.concatenate([vert[1:] for vert in vertex_sets])

    # create input for vectorized computation in one go
    pos_obs = np.tile(pos_obs[:npp], (nseg,1))
    curr_tile = np.repeat(curr_tile, npp)
    pos_start = np.repeat(pos_start, npp, axis=0)
    pos_end = np.repeat(pos_end, npp, axis=0)

    # compute field
    field = field_BH_line(bh, curr_tile, pos_start, pos_end, pos_obs)
    field = np.reshape(field, (nseg, npp, 3))

    # sum for each vertex set
    ns_cum = [sum(nvs[:i]) for i in range(nv+1)]   # cumulative index positions
    field_sum = np.array([np.sum(field[ns_cum[i-1]:ns_cum[i]], axis=0) for i in range(1,nv+1)])

    return np.reshape(field_sum, (-1,3))


def field_BH_line(
    bh: bool,
    current: np.ndarray,
    pos_start: np.ndarray,
    pos_end: np.ndarray,
    pos_obs: np.ndarray
    ) -> np.ndarray:
    """
    ### Args:
    - bh (boolean): True=B, False=H
    - current (float): current on line in units of [A]
    - pos_start (ndarray nx3) start position of line segments
    - pos_end (ndarray nx3) end positions of line segments
    - pos_obs (ndarray nx3): n observer positions in units of [mm]

    ### Returns:
    - B-field (ndarray nx3): B-field vectors at pos_obs in units of mT

    ### init_state:
    Line current flowing in a straight line from pos_start to pos_end.

    ### Computation info:
    Field computation via law of Biot Savart. See also countless online ressources.
    eg. http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf

    ### Numerical instabilities:
        - singularity at r=0, B set to 0 within Config.EDGESIZE
    """
    # pylint: disable=too-many-statements

    # Check for zero-length segments
    mask0 = np.all(pos_start==pos_end, axis=1)
    if np.all(mask0):
        n0 = len(pos_obs)
        return np.zeros((n0,3))

    any_zero_segments = np.any(mask0)

    # continue only with non-zero segments
    if any_zero_segments:
        not_mask0 = ~mask0     # avoid multiple computation of ~mask
        p1 = pos_start[not_mask0]
        p2 = pos_end[not_mask0]
        po = pos_obs[not_mask0]
        current = current[not_mask0]
    else:
        p1 = pos_start        # just renaming
        p2 = pos_end
        po = pos_obs

    # p4 = projection of pos_obs onto line p1-p2
    p1p2 = p1-p2
    norm_12 = norm(p1p2, axis=1)
    t = np.sum((po-p1)*p1p2, axis=1) / norm_12**2
    p4 = p1 + (t*p1p2.T).T

    # distance of observer from line
    norm_o4 = norm(po - p4, axis=1)

    # on-line cases (set B=0)
    mask1 = norm_o4 < Config.EDGESIZE
    if np.all(mask1):
        n0 = len(pos_obs)
        return np.zeros((n0,3))

    any_on_line_cases = np.any(mask1)

    # redefine to avoid tons of slices
    if any_on_line_cases:
        not_mask1 = ~mask1     # avoid multiple computation of ~mask
        po = po[not_mask1]
        p1 = p1[not_mask1]
        p2 = p2[not_mask1]
        p4 = p4[not_mask1]
        norm_12 = norm_12[not_mask1]
        norm_o4 = norm_o4[not_mask1]
        current = current[not_mask1]

    # determine field direction
    cros = np.cross(p2-p1, po-p4)
    norm_cros = norm(cros, axis=1)
    eB = (cros.T/norm_cros).T

    # compute angles
    norm_o1 = norm(po-p1, axis=1)
    norm_o2 = norm(po-p2, axis=1)
    norm_41 = norm(p4-p1, axis=1)
    norm_42 = norm(p4-p2, axis=1)
    sinTh1 = norm_41/norm_o1
    sinTh2 = norm_42/norm_o2
    deltaSin = np.empty((len(po),))

    # determine how p1,p2,p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = ((norm_41>norm_12) * (norm_41>norm_42))
    deltaSin[mask2] = abs(sinTh1[mask2]-sinTh2[mask2])
    # both points above
    mask3 = ((norm_42>norm_12) * (norm_42>norm_41))
    deltaSin[mask3] = abs(sinTh2[mask3]-sinTh1[mask3])
    # one above one below or one equals p4
    mask4 = ~mask2 * ~mask3
    deltaSin[mask4] = abs(sinTh1[mask4]+sinTh2[mask4])

    field = (deltaSin/norm_o4*eB.T* current/10).T # m->mm, T->mT

    if any_on_line_cases: # brodcast into np.zeros
        n1 = len(mask1)
        fields1 = np.zeros((n1,3))
        fields1[not_mask1] = field
        field = fields1

    if any_zero_segments: # brodcast into np.zeros
        n0 = len(mask0)
        fields0 = np.zeros((n0,3))
        fields0[not_mask0] = field
        field = fields0

    # return B
    if bh:
        return field

    # return H (mT -> kA/m)
    H = field*10/4/np.pi
    return H
