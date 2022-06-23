import numpy as np

from magpylib._src.exceptions import MagpylibInternalError
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_field
from magpylib._src.fields.field_BH_cylinder_segment import (
    magnet_cylinder_segment_field_internal,
)
from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.fields.field_BH_line import current_vertices_field
from magpylib._src.fields.field_BH_loop import current_loop_field
from magpylib._src.fields.field_BH_sphere import magnet_sphere_field
from magpylib._src.fields.field_BH_tetrahedron import magnet_tetrahedron_field

FIELD_FUNCTIONS = {
    "Cuboid": magnet_cuboid_field,
    "Cylinder": magnet_cylinder_field,
    "CylinderSegment": magnet_cylinder_segment_field_internal,
    "Sphere": magnet_sphere_field,
    "Tetrahedron": magnet_tetrahedron_field,
    "Dipole": dipole_field,
    "Loop": current_loop_field,
    "Line": current_vertices_field,
}


def getBH_level1(
    *,
    source_type: str,
    position: np.ndarray,
    orientation: np.ndarray,
    observers: np.ndarray,
    **kwargs: dict,
) -> np.ndarray:
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

    # transform obs_pos into source CS
    pos_rel_rot = orientation.apply(observers - position, inverse=True)

    # collect dictionary inputs and compute field
    field_func = FIELD_FUNCTIONS.get(source_type, None)

    if source_type == "CustomSource":
        field = kwargs["field"]
        if kwargs.get("field_func", None) is not None:
            BH = kwargs["field_func"](field, pos_rel_rot)
    elif field_func is not None:
        BH = field_func(observers=pos_rel_rot, **kwargs)
    else:
        raise MagpylibInternalError(f'Bad src input type "{source_type}" in level1')

    # transform field back into global CS
    BH = orientation.apply(BH)

    return BH
