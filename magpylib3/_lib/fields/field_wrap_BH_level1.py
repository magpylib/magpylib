import sys
import numpy as np
from magpylib3._lib.fields.field_BH_box import field_BH_box
from magpylib3._lib.fields.field_BH_cylinder import field_BH_cylinder


def getBH_level1(**kwargs:dict) -> np.ndarray:
    """ Vectorized field computation

    Args
    ----
    kwargs: dict of "1D"-input vectors that describes the computation.

    Returns
    -------
    field: ndarray, shape (l*m*n,3)

    Info
    ----
    - no input checks !
    - applys spatial transformations global CS <-> source CS
    - selects the correct Bfield_XXX function from input
    """

    # inputs
    src_type = kwargs['src_type']
    bh = kwargs['bh']  # True=B, False=H

    rot = kwargs['rot'] # only rotation object allowed as input
    pos = kwargs['pos']
    poso = kwargs['pos_obs']

    # transform obs_pos into source CS
    pos_rel = pos - poso                           # relative position
    pos_rel_rot = rot.apply(pos_rel, inverse=True) # rotate rel_pos into source CS

    # compute field
    if src_type == 'Box':
        mag = kwargs['mag']
        dim = kwargs['dim']
        B = field_BH_box(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Cylinder':
        mag = kwargs['mag']
        dim = kwargs['dim']
        niter = kwargs['niter']
        B = field_BH_cylinder(bh, mag, dim, pos_rel_rot, niter)
    else:
        sys.exit('ERROR: getBH() - bad src input type')

    # transform field back into global CS
    B = rot.apply(B)

    return B
