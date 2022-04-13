import numpy as np

from magpylib._src.exceptions import MagpylibInternalError
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_field
from magpylib._src.fields.field_BH_cylinder_segment import (
    magnet_cylinder_segment_field_internal,
)
from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.fields.field_BH_line import current_line_field
from magpylib._src.fields.field_BH_line import field_BH_line_from_vert
from magpylib._src.fields.field_BH_loop import current_loop_field
from magpylib._src.fields.field_BH_sphere import magnet_sphere_field


def getBH_level1(**kwargs: dict) -> np.ndarray:
    """Vectorized field computation

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
    src_type = kwargs["source_type"]
    field = kwargs["field"]  # 'B' or 'H'

    rot = kwargs["orientation"]  # only rotation object allowed as input
    pos = kwargs["position"]
    poso = kwargs["observers"]

    # transform obs_pos into source CS
    pos_rel = poso - pos  # relative position
    pos_rel_rot = rot.apply(pos_rel, inverse=True)  # rotate rel_pos into source CS

    # collect dictionary inputs and compute field
    if src_type == "Cuboid":
        mag = kwargs["magnetization"]
        dim = kwargs["dimension"]
        B = magnet_cuboid_field(field, pos_rel_rot, mag, dim)

    elif src_type == "Cylinder":
        mag = kwargs["magnetization"]
        dim = kwargs["dimension"]
        B = magnet_cylinder_field(field, pos_rel_rot, mag, dim)

    elif src_type == "CylinderSegment":
        mag = kwargs["magnetization"]
        dim = kwargs["dimension"]
        B = magnet_cylinder_segment_field_internal(field, pos_rel_rot, mag, dim)

    elif src_type == "Sphere":
        mag = kwargs["magnetization"]
        dia = kwargs["diameter"]
        B = magnet_sphere_field(field, pos_rel_rot, mag, dia)

    elif src_type == "Dipole":
        moment = kwargs["moment"]
        B = dipole_field(field, pos_rel_rot, moment)

    elif src_type == "Loop":
        current = kwargs["current"]
        dia = kwargs["diameter"]
        B = current_loop_field(field, pos_rel_rot, current, dia)

    elif src_type == "Line":
        current = kwargs["current"]
        if "vertices" in kwargs:
            vertices = kwargs["vertices"]
            B = field_BH_line_from_vert(field, pos_rel_rot, current, vertices)
        else:
            pos_start = kwargs["segment_start"]
            pos_end = kwargs["segment_end"]
            B = current_line_field(field, pos_rel_rot, current, pos_start, pos_end)

    elif src_type == "CustomSource":
        if kwargs.get("field_func", None) is not None:
            B = kwargs["field_func"](field, pos_rel_rot)

    else:
        raise MagpylibInternalError(f'Bad src input type "{src_type}" in level1')

    # transform field back into global CS
    B = rot.apply(B)

    return B
