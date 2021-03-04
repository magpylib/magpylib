import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.fields.field_wrap_BH_level1 import getBH_level1
from  magpylib._lib.math_utility import check_allowed_keys

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

    # unknown kwarg input ('user accident') -------------------------
    allowed_keys = ['bh', 'mag', 'dim', 'pos', 'rot', 'pos_obs', 'niter', 'src_type']
    check_allowed_keys(allowed_keys, kwargs, 'getBHv()')

    # generate dict of secured inputs for auto-tiling ---------------
    tile_params = {}

    # mandatory general inputs ------------------
    try:
        src_type = kwargs['src_type']
        poso = np.array(kwargs['pos_obs'], dtype=float)
    except KeyError as kerr:
        sys.exit('ERROR: getBHv() - missing input ' + str(kerr))
    tile_params['pos_obs'] = poso           # <-- tile

    # optional general inputs -------------------
    pos = np.array(kwargs.get('pos', (0,0,0)), dtype=float)
    tile_params['pos'] = pos                # <-- tile

    rot = kwargs.get('rot', R.from_quat((0,0,0,1)))
    tile_params['rot'] = rot.as_quat()      # <-- tile

    # mandatory class specific inputs -----------
    if src_type == 'Box':
        try:
            mag = np.array(kwargs['mag'],dtype=float)
            dim = np.array(kwargs['dim'],dtype=float)
        except KeyError as kerr:
            sys.exit('ERROR getBHv: missing input ' + str(kerr))
        tile_params['mag'] = mag            # <-- tile
        tile_params['dim'] = dim            # <-- tile
    elif src_type == 'Cylinder':
        try:
            mag = np.array(kwargs['mag'],dtype=float)
            dim = np.array(kwargs['dim'],dtype=float)
        except KeyError as kerr:
            sys.exit('ERROR: getBHv() - missing input ' + str(kerr))
        tile_params['mag'] = mag            # <-- tile
        tile_params['dim'] = dim            # <-- tile
        niter = kwargs.get('niter', 50)     # set niter
        kwargs['niter'] = niter

    # auto tile 1D parameters ---------------------------------------

    # evaluation vector length
    ns = [len(val) for val in tile_params.values() if val.ndim == 2]
    if len(set(ns)) > 1:
        sys.exit('ERROR: getBHv() - bad array input lengths: ' + str(set(ns)))
    n = max(ns, default=1)

    # tile 1D inputs and replace original values in kwargs
    for key,val in tile_params.items():
        if val.ndim == 1:
            kwargs[key] = np.tile(val,(n,1))
        else:
            kwargs[key] = val
    # change rot to Rotation object
    kwargs['rot'] = R.from_quat(kwargs['rot'])

    # compute and return B
    B = getBH_level1(**kwargs)

    if n==1: # remove highest level when n=1
        return B[0]
    return B


def getBv(**kwargs: dict) -> np.ndarray:
    """ B-Field computation from dictionary of vectors.

    ### Args:
    - src_type (string): source type must be either 'Box', 'Cylinder', ...
    - pos (ndarray): default=(0,0,0), source positions
    - rot (scipy..Rotation): default=unit_rotation: source rotations relative to init_state
    - pos_obs (ndarray): observer positions

    ### Args-magnets:
    - mag (vector): homogeneous magnet magnetization in units of mT
    - dim (vector): magnet dimension input

    ### Args-specific:
    - niter (int): default=50, for Cylinder sources diametral iteration

    ### Returns:
    - B-field (ndarray Nx3): B-field at pos_obs in units of mT

    ### Info:
    Inputs pos and rot will be set to default if not given. 3-vector inputs will
    automatically be tiled to Nx3 format. At least one input must be of the form
    Nx3 !

    A check for mandatory input information is performed.
    """

    return getBHv_level2(bh=True, **kwargs)


def getHv(**kwargs: dict) -> np.ndarray:
    """ H-Field computation from dictionary of vectors.

    ### Args:
    - src_type (string): source type must be either 'Box', 'Cylinder', ...
    - pos (ndarray): default=(0,0,0), source positions
    - rot (scipy..Rotation): default=unit_rotation: source rotations relative to init_state
    - pos_obs (ndarray): observer positions

    ### Args-magnets:
    - mag (vector): homogeneous magnet magnetization in units of mT
    - dim (vector): magnet dimension input

    ### Args-specific:
    - niter (int): default=50, for Cylinder sources diametral iteration

    ### Returns:
    - H-field (ndarray Nx3): H-field at pos_obs in units of kA/m

    ### Info:
    Inputs pos and rot will be set to default if not given. 3-vector inputs will
    automatically be tiled to Nx3 format. At least one input must be of the form
    Nx3 !

    A check for mandatory input information is performed.
    return getBHv(False,**kwargs)
    """

    return getBHv_level2(bh=False, **kwargs)
