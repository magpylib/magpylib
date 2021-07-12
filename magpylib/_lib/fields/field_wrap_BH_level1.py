import numpy as np
from magpylib._lib.fields.field_BH_cuboid import field_BH_cuboid
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
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches

    # base inputs of all sources
    src_type = kwargs['source_type']
    bh = kwargs['bh']      # True=B, False=H

    rot = kwargs['orientation']    # only rotation object allowed as input
    pos = kwargs['position']
    poso = kwargs['observer']

    # transform obs_pos into source CS
    pos_rel = poso - pos                           # relative position
    pos_rel_rot = rot.apply(pos_rel, inverse=True) # rotate rel_pos into source CS

    # collect dictionary inputs and compute field
    if src_type == 'Cuboid':
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        B = field_BH_cuboid(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Cylinder':
        # Cylinder dimension input can be shape (2,), (5,) or (6,)
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        if len(dim[0]) == 2: # (d,h) type input
            n = len(dim)
            d, h = dim.T
            di = np.zeros(n)
            phi1 = np.zeros(n)
            phi2=np.ones(n)*360
        elif len(dim[0]) == 3: # (d,h,di) type input
            n = len(dim)
            d, h, di = dim.T
            phi1 = np.zeros(n)
            phi2=np.ones(n)*360
        else: # (d,h,di,phi1,phi2) type input
            d,h,di,phi1,phi2 = dim.T
        r1, r2 = di/2, d/2
        z1, z2 = -h/2, h/2
        dim = np.array([r1,r2,phi1,phi2,z1,z2]).T
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
