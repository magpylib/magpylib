import numpy as np
from magpylib._src.fields.field_BH_cuboid import field_BH_cuboid
from magpylib._src.fields.field_BH_cylinder import field_BH_cylinder
from magpylib._src.fields.field_BH_cylinder_tile import field_BH_cylinder_tile
from magpylib._src.fields.field_BH_sphere import field_BH_sphere
from magpylib._src.fields.field_BH_dipole import field_BH_dipole
from magpylib._src.fields.field_BH_loop import field_BH_loop
from magpylib._src.fields.field_BH_line import field_BH_line, field_BH_line_from_vert
from magpylib._src.exceptions import MagpylibInternalError


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
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        B = field_BH_cylinder(bh, mag, dim, pos_rel_rot)

    elif src_type == 'CylinderSegment':
        mag = kwargs['magnetization']
        dim = kwargs['dimension']
        r1,r2,h,phi1,phi2 = dim.T
        z1, z2 = -h/2, h/2
        dim = np.array([r1,r2,phi1,phi2,z1,z2]).T
        B = field_BH_cylinder_tile(bh, mag, dim, pos_rel_rot)

    elif src_type == 'Sphere':
        mag = kwargs['magnetization']
        dia = kwargs['diameter']
        B = field_BH_sphere(bh, mag, dia, pos_rel_rot)

    elif src_type == 'Dipole':
        moment = kwargs['moment']
        B = field_BH_dipole(bh, moment, pos_rel_rot)

    elif src_type == 'Loop':
        current = kwargs['current']
        dia = kwargs['diameter']
        B = field_BH_loop(bh, current, dia, pos_rel_rot)

    elif src_type =='Line':
        current = kwargs['current']
        if 'vertices' in kwargs:
            vertices = kwargs['vertices']
            B = field_BH_line_from_vert(bh, current, vertices, pos_rel_rot)
        else:
            pos_start = kwargs['segment_start']
            pos_end = kwargs['segment_end']
            B = field_BH_line(bh, current, pos_start, pos_end, pos_rel_rot)

    elif src_type == 'CustomSource':
        bh_key = 'B' if bh else 'H'
        if kwargs[f'field_{bh_key}_lambda'] is not None:
            B = kwargs[f'field_{bh_key}_lambda'](pos_rel_rot)
        else:
            raise MagpylibInternalError(
                f'{bh_key}-field calculation not implemented for CustomSource class'
            )

    else:
        raise MagpylibInternalError(f'Bad src input type "{src_type}" in level1')

    # transform field back into global CS
    B = rot.apply(B)

    return B
