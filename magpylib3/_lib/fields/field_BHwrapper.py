"""
Field computation structure:

level0 (vectorized core Field functions): 
    input arrays
    output B-field arr in local CS 
    - pure vectorized field computations from literature
    - all computations in source CS
    - in field_BH_xxx.py files

level1(getBH_level1): calls level0
    input: dict of arrays
    output B-field arr in global CS
    - apply transformation to global CS
    - select correct level0 src_type computation

level2(getBHv): calls level1
    input: dict
    output B-field arr in global CS
    - check input for mandatory information
    - set missing input variables to default values
    - tile 1D inputs

getBH_level2:
    - input dict checks (unknowns)
    - secure user inputs
    - group similar sources for combined computation
    - generate vector input format for getBH_level1
    - adjust Bfield output format to pos_obs, path, sources input format
    
level3(getB, getH, getBv, getHv):
    - user interface
    - docstrings
    - separated B and H
    - transform input into dict for level2

level4(src.getB, src.getH):
    - user interface
    - docstrings 
    - calling level3 directly from sources

"""

import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3 import _lib as _lib
from magpylib3._lib.fields.field_BH_box import field_BH_box
from magpylib3._lib.fields.field_BH_cylinder import field_BH_cylinder
from magpylib3._lib.math_utility.utility import format_src_input, same_path_length
from magpylib3._lib.config import config


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
    bh =  kwargs['bh']  # True=B, False=H

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


def getBHv(**kwargs: dict) -> np.ndarray:
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
    keys = kwargs.keys()
    complement = [i for i in keys if i not in allowed_keys]
    if complement:
        print('WARNING: getBHv() - unknown input kwarg, ', complement)

    # generate dict of secured inputs for auto-tiling ---------------
    tile_params = {} 

    # mandatory general inputs ------------------
    try: 
        src_type = kwargs['src_type']
        poso = np.array(kwargs['pos_obs'], dtype=float)
    except KeyError as ke:
        sys.exit('ERROR: getBHv() - missing input ' + str(ke))
    tile_params['pos_obs'] = poso           # <-- tile

    # optional general inputs -------------------
    pos = np.array(kwargs.get('pos', (0,0,0)), dtype=float)
    tile_params['pos'] = pos                # <-- tile
    
    rot = kwargs.get('rot', R.from_quat((0,0,0,1)))
    tile_params['rot'] = rot.as_quat()      # <-- tile

    # mandatory class specific inputs -----------
    if src_type is 'Box':
        try:
            mag = np.array(kwargs['mag'],dtype=float)
            dim = np.array(kwargs['dim'],dtype=float)
        except KeyError as ke:
            sys.exit('ERROR getBHv: missing input ' + str(ke))
        tile_params['mag'] = mag            # <-- tile
        tile_params['dim'] = dim            # <-- tile
    elif src_type is 'Cylinder':
        try:
            mag = np.array(kwargs['mag'],dtype=float)
            dim = np.array(kwargs['dim'],dtype=float)
        except KeyError as ke:
            sys.exit('ERROR: getBHv() - missing input ' + str(ke))
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


def scr_dict_homo_mag(group: list, poso_flat: np.ndarray) -> dict:
    """ Generates getBH_level1 input dict for homogeneous magnets

    Parameters
    ----------
    - group: list of sources
    - poso_flat: ndarray, shape (n,3), pos_obs flattened

    Returns
    -------
    dict for getBH_level1 input
    
    """
    l_group = len(group)    # sources in group
    m = len(group[0]._pos)  # path length
    n = len(poso_flat)      # no. observer pos
    len_dim = len(group[0]._dim)

    # prepare and fill arrays, shape: (l_group, m, n)
    magv = np.empty((l_group*m*n,3))
    dimv = np.empty((l_group*m*n,len_dim))
    posv = np.empty((l_group*m*n,3))
    rotv = np.empty((l_group*m*n,4))
    for i,s in enumerate(group):
        magv[i*m*n:(i+1)*m*n] = np.tile(s.mag, (m*n,1))
        dimv[i*m*n:(i+1)*m*n] = np.tile(s.dim, (m*n,1))
        posv[i*m*n:(i+1)*m*n] = np.tile(s._pos,n).reshape(m*n,3)
        rotv[i*m*n:(i+1)*m*n] = np.tile(s._rot.as_quat(),n).reshape(m*n,4)
    posov = np.tile(poso_flat, (l_group*m,1))
    rotobj = R.from_quat(rotv, normalized=True)
    src_dict = {'mag':magv, 'dim':dimv, 'pos':posv, 'pos_obs': posov, 'rot':rotobj}
    return src_dict


def getBH_level2(**kwargs: dict) -> np.ndarray:
    """Field computation (level2) for given sources

    Parameters
    ----------
    - bh (bool): True=getB, False=getH
    - sources (L list): a 1D list of L sources/collections with similar pathlength M
    - pos_obs (N1 x N2 x ... x 3 array_like): observer positions
    - sumup (bool): default=False returns [B1,B2,...], True returns sum(Bi)
    - niter (int): default=50, for Cylinder sources diametral iteration

    Returns
    -------
    field: ndarray, shape (L,M,N1,N2,...,3), field of L sources, M path positions and N observer positions

    Info:
    -----
    - input dict checks (unknowns)
    - secure user inputs
    - group similar sources for combined computation
    - generate vector input format for getBH_level1
    - adjust Bfield output format to pos_obs, path, sources input format

    """

    # make sure there is no unknown kwarg input ('user accident') ----
    allowed_keys = ['bh', 'sources', 'pos_obs', 'sumup', 'niter']
    keys = kwargs.keys()
    complement = [i for i in keys if i not in allowed_keys]
    if complement:
        print('WARNING: getBH() - unknown input kwarg, ', complement)

    # collect input ------------------------------------------------
    bh = kwargs['bh']
    sources = kwargs['sources']
    sumup = kwargs['sumup']
    poso = np.array(kwargs['pos_obs'], dtype=np.float)
    
    # formatting ----------------------------------------------------
    # flatten out Collections
    src_list = format_src_input(sources)

    # test if all sources have a similar path length and good path format
    if not same_path_length(src_list):
        sys.exit('ERROR: getBH() - all paths must be of good format and similar length !')

    # determine shape of positions and flatten into nx3 array
    poso_shape = poso.shape
    n = np.prod(poso_shape[:-1],dtype=int) # pylint: disable=unsubscriptable-object
    poso_flat = np.reshape(poso,(n,3))

    l = len(src_list)           # number of sources
    m = len(src_list[0]._pos)   # path length
    B = np.empty((l,m,n,3))     # store fields here

    # group similar source types-------------------------------------
    src_sorted = [[],[]]   # store groups here
    order = [[],[]]        # keep track of the source order
    for i,s in enumerate(src_list):
        if isinstance(s, _lib.obj_classes.Box):
            src_sorted[0] += [s]
            order[0] += [i]
        elif isinstance(s, _lib.obj_classes.Cylinder):
           src_sorted[1] += [s]
           order[1] += [i]
        else:
            sys.exit('WARNING getBH() - bad source input !')

    # evaluate each non-empty group in one go------------------------

    # Box group
    group = src_sorted[0]  
    if group:
        src_dict = scr_dict_homo_mag(group, poso_flat)              # compute array dict for level1
        B_group = getBH_level1(bh=bh, src_type='Box', **src_dict)   # compute field
        B_group = B_group.reshape((len(group),m,n,3))               # reshape
        for i in range(len(group)):                                 # move to dedicated positions in B
                B[order[0][i]] = B_group[i]

    # Cylinder group
    group = src_sorted[1]
    if group:
        niter = kwargs.get('niter', config.ITER_CYLINDER)
        src_dict = scr_dict_homo_mag(group, poso_flat)
        B_group = getBH_level1(bh=bh, src_type='Cylinder', niter=niter, **src_dict)
        B_group = B_group.reshape((len(group),m,n,3))
        for i in range(len(group)):
                B[order[1][i]] = B_group[i]

    # bring to correct shape (B.shape = pos_obs.shape)---------------
    B = B.reshape(np.r_[l,m,poso_shape])

    # pathlength = 1: reduce 2nd level
    if m == 1:
        B = np.squeeze(B, axis=1)

    # only one source: reduce highest level
    if l == 1:
        return B[0]
    
    if sumup:
        B = np.sum(B, axis=0)
        return B

    # rearrange B when there is at least one Collection with more than one source
    if len(src_list) > len(sources):
        for i,s in enumerate(sources):
            if isinstance(s, _lib.obj_classes.Collection):
                col_len = len(s._sources)
                B[i] = np.sum(B[i:i+col_len],axis=0)    # set B[i] to sum of slice
                B = np.delete(B,np.s_[i+1:i+col_len],0) # delete remaining part of slice
    
    return B

# INTERFACE FUNCTIONS -------------------------------------------
def getB(sources:list, pos_obs:np.ndarray, sumup:bool=False, **specs:dict) -> np.ndarray:
    """ Compute the B-field for a sequence of given sources.

    Parameters
    ----------
    sources: list 
        1D list of L sources/collections with similar path length M
    
    pos_obs: array_like, shape (N1,N2,...,3), unit [mm]
        observer position(s) in units of [mm].

    sumup: bool, default=False
        If False getB returns shape (L,M,N1,...), else sums up the fields of all sources 
        and returns shape (M,N1,...).
    
    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    B-field: ndarray, shape (L, M, N1, N2, ... ,3), unit [mT]
        B-field of each source at each path position and each observer position in units of mT

    Info
    ----
    This function automatically groups similar sources together for optimal vectorization 
    of the computation. For maximal performance call this function as little as possible, 
    do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(bh=True, sources=sources, pos_obs=pos_obs, sumup=sumup, **specs)


def getH(sources:Sequence, pos_obs:np.ndarray, sumup:bool=False, **specs:dict) -> np.ndarray:
    """ Compute the H-field for a sequence of given sources.

    Parameters
    ----------
    sources: list 
        1D list of L sources/collections with similar path length M
    
    pos_obs: array_like, shape (N1,N2,...,3), unit [mm]
        observer position(s) in units of [mm].

    sumup: bool, default=False
        If False getB returns shape (L,M,N1,...), else sums up the fields of all sources 
        and returns shape (M,N1,...).
    
    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    B-field: ndarray, shape (L, M, N1, N2, ... ,3), unit [mT]
        B-field of each source at each path position and each observer position in units of mT

    Info
    ----
    This function automatically groups similar sources together for optimal vectorization 
    of the computation. For maximal performance call this function as little as possible, 
    do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(bh=False, sources=sources, pos_obs=pos_obs, sumup=sumup, **specs)


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
    return getBHv(bh=True, **kwargs)

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
    return getBHv(bh=False, **kwargs)
