"""
Implementations of analytical expressions of line current segments
"""

import numpy as np
from numpy.linalg import norm
from magpylib._lib.config import Config

def field_BH_line(
    bh: bool,
    pos_obs: np.ndarray,
    vertices: np.ndarray,
    current: float):
    """
    ### Args:
    - bh (boolean): True=B, False=H
    - pos_obs (ndarray nx3): n observer positions in units of [mm]
    - vertices (ndarray (m+1)x3): vertices of m line segements [mm]
    - current (float): current on line in units of [A]

    ### Returns:
    - B-field (ndarray nx3): B-field vectors at pos_obs in units of mT

    ### init_state:
    Line current flowing in a straight line (segment) from vertext to vertext, starting
    at the first vertex and ending at the last.

    ### Computation info:
    Field computation via law of Biot Savart. See also countless online ressources.
    eg. http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
    """
    # p1 are init positions, p2 end position of the line segments
    p1 = vertices[:-1]
    p2 = vertices[1:]

    # Check for zero-length segment and redefine vertices
    mask0 = np.invert(np.all(p1==p2, axis=1))
    p1, p2 = p1[mask0], p2[mask0]

    n = len(pos_obs)         # number of observer positions
    m = len(p1)              # number of segments

    # tile up inputs to shape --> obs_pos(n), segments(m), xyz(3)
    p1 = np.tile(p1, (n,1,1))
    p2 = np.tile(p2, (n,1,1))
    po = np.tile(pos_obs, (m, 1, 1))
    po = np.swapaxes(po, 0, 1)

    # ravel to shape (n*m, 3) - required for masking special cases
    p1 = np.reshape(p1, (n*m,3))
    p2 = np.reshape(p2, (n*m,3))
    po = np.reshape(po, (n*m,3))

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

    # mask1 fields should be (0,0,0)
    fields = np.zeros((n*m,3))
    fields[mask1] = (deltaSin/norm_o4*eB.T* current/10).T # m->mm, T->mT

    # reshape and sum over all sections
    fields = np.reshape(fields, (n,m,3))
    B = np.sum(fields, axis=1)

    if bh:
        return B

    # transform units mT -> kA/m
    H = B*10/4/np.pi
    return H
