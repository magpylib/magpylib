""" getBHv wrapper codes"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._lib.exceptions import MagpylibBadUserInput


def getBHv_level2(**kwargs: dict) -> np.ndarray:
    """ Direct access to vectorized computation

    Parameters
    ----------
    kwargs: dict that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3), field at obs_pos in [mT] or [kA/m]

    Info
    ----
    - check inputs

    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # generate dict of secured inputs for auto-tiling ---------------
    #  entries in this dict will be tested for input length, and then
    #  be automatically tiled up and stored back into kwargs for calling
    #  getBH_level1().
    #  To allow different input dimensions, the tdim argument is also given
    #  which tells the program which dimension it should tile up.
    tile_params = {}

    # mandatory general inputs ------------------
    try:
        src_type = kwargs['src_type']
        poso = np.array(kwargs['pos_obs'], dtype=float)
        tile_params['pos_obs'] = (poso,2)    # <-- (input,tdim)

        # optional general inputs -------------------
        # if no input set pos=0
        pos = np.array(kwargs.get('pos', (0,0,0)), dtype=float)
        tile_params['pos'] = (pos,2)
        # if no input set rot=unit
        rot = kwargs.get('rot', R.from_quat((0,0,0,1)))
        tile_params['rot'] = (rot.as_quat(),2)
        # if no input set squeeze=True
        squeeze = kwargs.get('squeeze', True)

        # mandatory class specific inputs -----------
        if src_type == 'Box':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,2)

        elif src_type == 'Cylinder':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,2)

        elif src_type == 'Sphere':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,1)

        elif src_type == 'Dipole':
            moment = np.array(kwargs['moment'], dtype=float)
            tile_params['moment'] = (moment,2)

        elif src_type == 'Circular':
            current = np.array(kwargs['current'], dtype=float)
            tile_params['current'] = (current,1)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,1)

        elif src_type == 'Line':
            current = np.array(kwargs['current'], dtype=float)
            tile_params['current'] = (current,1)
            pos_start = np.array(kwargs['pos_start'], dtype=float)
            tile_params['pos_start'] = (pos_start,2)
            pos_end = np.array(kwargs['pos_end'], dtype=float)
            tile_params['pos_end'] = (pos_end,2)

    except KeyError as kerr:
        msg = f'Missing input keys: {str(kerr)}'
        raise MagpylibBadUserInput(msg) from kerr

    # auto tile 1D parameters ---------------------------------------

    # evaluation vector length
    ns = [len(val) for val,tdim in tile_params.values() if val.ndim == tdim]
    if len(set(ns)) > 1:
        msg = f'getBHv() bad array input lengths: {str(set(ns))}'
        raise MagpylibBadUserInput(msg)
    n = max(ns, default=1)

    # tile 1D inputs and replace original values in kwargs
    for key,(val,tdim) in tile_params.items():
        if val.ndim<tdim:
            if tdim == 2:
                kwargs[key] = np.tile(val,(n,1))
            elif tdim == 1:
                kwargs[key] = np.array([val]*n)
        else:
            kwargs[key] = val

    # change rot to Rotation object
    kwargs['rot'] = R.from_quat(kwargs['rot'])

    # compute and return B
    B = getBH_level1(**kwargs)

    if squeeze:
        return np.squeeze(B)
    return B


# ON INTERFACE
def getBv(**kwargs):
    """
    B-Field computation from dictionary of input vectors.

    This function avoids the object-oriented Magpylib interface and gives direct
        access to the field implementations. It is the fastet way to compute fields
        with Magpylib.

    Inputs will automatically be tiled to shape (N,x) to fit with other inputs.

    Parameters
    ----------
    src_type: string
        Source type of computation. Must be either 'Box', 'Cylinder', 'Sphere', 'Dipole',
        'Circular' or 'Line'.

    pos: array_like, shape (3,) or (N,3), default=(0,0,0)
        Source positions in units of [mm].

    rot: scipy Rotation object, default=unit rotation
        Source rotations relative to the init_state.

    pos_obs: array_like, shape (3,) or (N,3)
        Observer positions in units of [mm].

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output are eliminated.

    Parameters - homogenous magnets
    -------------------------------
    mag: array_like, shape (3,) or (N,3)
        Homogeneous magnet magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape is src_type dependent
        Magnet dimension input in units of [mm].

    Parameters - Dipole
    -------------------
    moment:  array_like, shape (3,) or (N,3)
        Magnetic dipole moment in units of [mT*mm^3]. For homogeneous magnets the
        relation is moment = magnetization*volume.

    Parameters - Circular current loop
    ----------------------------------
    current: array_like, shape (N,)
        Current flowing in loop in units of [A].

    dim: array_like, shape (N,)
        Diameter of circular loop in units of [mm].

    Parameters - Line current
    -------------------------
    current: array_like, shape (N,)
        Current in units of [A]

    pos_start: array_like, shape (N,3)
        Start positions of line current segments.

    pos_end: array_like, shape (N,3)
        End positions of line current segments.

    Returns
    -------
    B-field: ndarray, shape (N,3)
        B-field generated by source at pos_obs in units of [mT].
    """
    return getBHv_level2(bh=True, **kwargs)


# ON INTERFACE
def getHv(**kwargs):
    """
    H-Field computation from dictionary of input vectors.

    This function avoids the object-oriented Magpylib interface and gives direct
        access to the field implementations. It is the fastet way to compute fields
        with Magpylib.

    Inputs will automatically be tiled to shape (N,x) to fit with other inputs.

    Parameters
    ----------
    src_type: string
        Source type of computation. Must be either 'Box', 'Cylinder', 'Sphere', 'Dipole',
        'Circular' or 'Line'.

    pos: array_like, shape (3,) or (N,3), default=(0,0,0)
        Source positions in units of [mm].

    rot: scipy Rotation object, default=unit rotation
        Source rotations relative to the init_state.

    pos_obs: array_like, shape (3,) or (N,3)
        Observer positions in units of [mm].

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output are eliminated.

    Parameters - homogenous magnets
    -------------------------------
    mag: array_like, shape (3,) or (N,3)
        Homogeneous magnet magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape is src_type dependent
        Magnet dimension input

    Parameters - Dipole
    -------------------
    moment:  array_like, shape (3,) or (N,3)
        Magnetic dipole moment in units of [mT*mm^3]. For homogeneous magnets the
        relation is moment = magnetization*volume.

    Parameters - Circular current loop
    ----------------------------------
    current: array_like, shape (N,)
        Current flowing in loop in units of [A].

    dim: array_like, shape (N,)
        Diameter of circular loop in units of [mm].

    Parameters - Line current
    -------------------------
    current: array_like, shape (N,)
        Current in units of [A]

    pos_start: array_like, shape (N,3)
        Start positions of line current segments.

    pos_end: array_like, shape (N,3)
        End positions of line current segments.

    Returns
    -------
    H-field: ndarray, shape (N,3)
        H-field generated by source at pos_obs in units of [kA/m].
    """
    return getBHv_level2(bh=False, **kwargs)
