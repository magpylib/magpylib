"""
Implementations of analytical expressions of line current segments
"""

import numpy as np
from numpy.linalg import norm
from magpylib._lib.config import Config

def field_BH_line(
    bh: bool,
    current: float,
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

    # Check for zero-length segments
    mask0 = np.invert(np.all(pos_start==pos_end, axis=1))

    p1 = pos_start[mask0]
    p2 = pos_end[mask0]
    po = pos_obs[mask0]

    # p4 = projection of pos_obs onto line p1-p2
    p1p2 = p1-p2
    norm_12 = norm(p1p2, axis=1)
    t = np.sum((po-p1)*p1p2, axis=1) / norm_12**2
    p4 = p1 + (t*p1p2.T).T

    # on-line cases (set B=0)
    norm_o4 = norm(po - p4, axis=1) # distance of observer from line
    mask1 = np.invert(norm_o4 < Config.EDGESIZE)

    # redefine to avoid tons of slices
    if np.any(~mask1):
        po = po[mask1]
        p1 = p1[mask1]
        p2 = p2[mask1]
        p4 = p4[mask1]

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

    # mask1 and mask0 fields should be (0,0,0)
    n1 = len(po)
    fields1 = np.zeros((n1,3))

    fields1[mask1] = (deltaSin/norm_o4*eB.T* current/10).T # m->mm, T->mT

    n0 = len(pos_obs)
    fields0 = np.zeros((n0,3))
    fields0[mask0] = fields1

    # return B
    if bh:
        return fields0

    # return H (mT -> kA/m)
    H = fields0*10/4/np.pi
    return H
