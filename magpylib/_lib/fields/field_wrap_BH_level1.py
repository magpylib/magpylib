import numpy as np
from magpylib._lib.fields.field_BH_box import field_BH_box
from magpylib._lib.fields.field_BH_cylinder import field_BH_cylinder
from magpylib._lib.fields.field_BH_cylinder_old import field_BH_cylinder_old
from magpylib._lib.fields.field_BH_sphere import field_BH_sphere
from magpylib._lib.fields.field_BH_dipole import field_BH_dipole
from magpylib._lib.fields.field_BH_circular import field_BH_circular
from magpylib._lib.fields.field_BH_line import field_BH_line, field_BH_line_from_vert
from magpylib._lib.exceptions import MagpylibInternalError


def getBH_level1(**kwargs:dict) -> np.ndarray:
    """ Vectorized field computation

    - applies spatial transformations global CS <-> source CS
    - selects the correct Bfield_XXX function from input

    Args
    ----
    kwargs: dict of shape (N,x) input vectors that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3)

    """

    # base inputs of all sources
    src_type = kwargs['source_type']
    bh = kwargs['bh']      # True=B, False=H

    rot = kwargs['orientation']    # only rotation object allowed as input
    pos = kwargs['position']
    poso = kwargs['observer']

    # transform obs_pos into source CS
    pos_rel = poso - pos                           # relative position
    pos_rel_rot = rot.apply(pos_rel, inverse=True) # rotate rel_pos into source CS

    # compute field
    if src_type == 'Box':
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        B = field_BH_box(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Cylinder':
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        # transform dim2 to dim6
        if len(dim[0]) == 2:
            n = len(dim)
            null = np.zeros(n)
            eins = np.ones(n)
            d, h = dim.T
            dim = np.array([null, d/2, null, eins*360, -h/2, h/2]).T
        B = field_BH_cylinder(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Cylinder_old':
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        B = field_BH_cylinder_old(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Sphere':
        mag = kwargs['magnetization']
        dia = kwargs['diameter']
        B = field_BH_sphere(bh, mag, dia, pos_rel_rot)
    elif src_type == 'Dipole':
        moment = kwargs['moment']
        B = field_BH_dipole(bh, moment, pos_rel_rot)
    elif src_type == 'Circular':
        current = kwargs['current']
        dia = kwargs['diameter']
        B = field_BH_circular(bh, current, dia, pos_rel_rot)
    elif src_type =='Line':
        current = kwargs['current']
        if 'vertices' in kwargs:
            vertices = kwargs['vertices']
            B = field_BH_line_from_vert(bh, current, vertices, pos_rel_rot)
        else:
            pos_start = kwargs['segment_start']
            pos_end = kwargs['segment_end']
            B = field_BH_line(bh, current, pos_start, pos_end, pos_rel_rot)
    else:
        raise MagpylibInternalError('Bad src input type in level1')

    # transform field back into global CS
    B = rot.apply(B)

    return B
