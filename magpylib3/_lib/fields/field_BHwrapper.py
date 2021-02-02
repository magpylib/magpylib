"""
Field computation structure:

level0 (vectorized core Field functions): 
    input arrays
    output B-field arr in local CS 
    - pure vectorized field computations from literature
    - all computations in source CS
    - in field_BH_xxx.py files

level1(getB_core): wraps level0
    input: dict of arrays
    output B-field arr in global CS
    - apply selected lev0 src_type computation
    - apply transformation to global CS

level2(getBv): wraps level1
    input: dict
    output B-field arr in global CS
    - check input for mandatory information
    - set missing input variables to default values
    - tile 1D inputs

level2(getB): wraps level1
    input: *sources + **kwargs (pos_obs, src_paths, etc...)
    output: B-field for each source at each pos
    - auto-generates correct vector input format from source attributes
    - groups similar sources for combined computation
    - adjust pos output shape to input shape

level3(src.getB): wraps level2 getB
"""

import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3 import _lib as _lib
from magpylib3._lib.fields.field_BH_box import field_BH_box
from magpylib3._lib.fields.field_BH_cylinder import field_BH_cylinder
from magpylib3._lib.math_utility.utility import format_src_input


def getBH_core(**kwargs:dict) -> np.ndarray:
    """ Field computation (level1) from input dict

    ### Args:
    - kwargs (dict): Input that describes the computation. See level0

    ### Returns:
    - B/H-field (ndarray Nx3): B(mT)H(kA/m) field at pos_obs

    ### Info (level1):
    This function wraps the level 0 core field computations.
    - selects the correct Bfield_XXX function from input
    - applys spatial transformations of fields to global CS
    - no input checks !
    """

    niter = 50 # bind to avoid Pylance PossiblyUnboundVariable Warning

    # inputs --------------------------------------------------------
    src_type = kwargs['src_type']
    bh =  kwargs['bh']  # True=B, False=H

    rot = kwargs['rot'] # only rotation object allowed as input
    pos = kwargs['pos']

    # class specific
    if src_type == 'Box' or 'Cylinder':
        mag = kwargs['mag']
        dim = kwargs['dim']
        poso = kwargs['pos_obs']

        if src_type == 'Cylinder':
            try:
                niter = kwargs['niter']
            except KeyError:
                pass # niter = 50 by default
    else:
        print('ERROR getBH_core: bad src input type')
        sys.exit()

    # field computation and spatial transformation ------------------
    # transform field sample position into CS of source
    pos_rel = pos - poso                          # relative position
    pos_rel_rot = rot.apply(pos_rel,inverse=True) # rotate rel_pos into source CS

    # compute field in source CS
    if src_type == 'Box':
        B = field_BH_box(bh, mag, dim, pos_rel_rot)
    elif src_type == 'Cylinder':
        B = field_BH_cylinder(bh, mag, dim, pos_rel_rot, niter)
    else:
        print('ERROR getBH_core: bad src input type')
        sys.exit()

    # transform field back into global CS
    
    B = rot.apply(B)

    return B


def getBHv(**kwargs: dict) -> np.ndarray:
    """ Field computation (level 2v) from dictionary of vectors.

    ### Args:
    - kwargs (dict): Input that describes the computation. See getBv() and getHv().

    ### Returns:
    - B/H-field (ndarray Nx3): B(mT)H(kA/m) field at pos_obs

    ### Info:
    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    
    tile_params = {} # collect all type secured inputs for auto-tiling
    n = 1  # set input vector length (in case there are only 1D inputs)

    # check mandatory general inputs
    try: 
        src_type = kwargs['src_type']
        poso = np.array(kwargs['pos_obs'],dtype=np.float)
    except KeyError as ke:
        print('ERROR getB_level1_user: missing input ', ke)
        sys.exit()
    tile_params['pos_obs'] = poso   # add for auto-tilting

    # check mandatory class specific inputs
    if src_type == 'Box' or 'Cylinder':
        try: 
            mag = np.array(kwargs['mag'],dtype=np.float)
            dim = np.array(kwargs['dim'],dtype=np.float)
        except KeyError as ke:
            print('ERROR getB_level1_user: missing input ', ke)
            sys.exit()
        tile_params['mag']=mag      # add for auto-tilting
        tile_params['dim']=dim      # add for auto-tilting

    # check optional general inputs
    try:
        pos = np.array(kwargs['pos'],dtype=np.float)
    except KeyError:
        pos = np.zeros(3)
    tile_params['pos'] = pos        # add for auto-tilting
    try:
        rot = kwargs['rot']
        n = len(rot.as_rotvec())    # input vector length from rot input
    except KeyError:
        # generate a unit rotation and add to dict
        kwargs['rot'] = R.from_rotvec((0,0,0))

    # determine input vector length
    for val in tile_params.values():
        if len(val.shape) == 2:
            n = len(val)
            break

    # tile 1D inputs and replace original values
    for key,val in tile_params.items():
        if len(val.shape) == 1:
            new_val = np.tile(val,(n,1))
            kwargs[key] = new_val

    # compute and return B
    B = getBH_core(**kwargs)
    if n==1: # remove highest level when n=1
        return B[0]
    return B


def getBH_box_group(bh: bool, group: list, poso_flat: np.ndarray) -> np.ndarray:
    """ Helper function that generates the vector-input for the
    Box magnet group of the getBH function and computes the field.

    ### Args:
    - bh (bool): True=getB, False=getH
    - group (list): list of sources in group
    - poso_flat (ndarray): pos_obs flattened

    ### Returns:
    - BH-field (ndarray): m_group*n x 3 arr of BH-vectors
    """
    m_group = len(group)
    n = len(poso_flat)
    magv = np.empty((m_group*n,3))
    dimv = np.empty((m_group*n,3))
    posv = np.empty((m_group*n,3))
    rotv = np.empty((m_group*n,3))
    for i,s in enumerate(group):
        magv[i*n:(i+1)*n] = np.tile(s._mag, (n,1))
        dimv[i*n:(i+1)*n] = np.tile(s._dim, (n,1))
        posv[i*n:(i+1)*n] = np.tile(s._pos, (n,1))
        rotv[i*n:(i+1)*n] = np.tile(s._rot.as_rotvec(), (n,1))
    posov = np.tile(poso_flat,(m_group,1))
    rotobj = R.from_rotvec(rotv)
    B = getBH_core(bh=bh, src_type='Box', mag=magv, dim=dimv, pos=posv, pos_obs=posov, rot=rotobj)
    return B


def getBH_cylinder_group(bh: bool, group: list, poso_flat: np.ndarray, niter: int) -> np.ndarray:
    """ Helper function that generates the vector-input for the
    Cylinder magnet group of the getBH function and computes the field.

    ### Args:
    - bh (bool): True=getB, False=getH
    - group (list): list of sources in group
    - poso_flat (ndarray): pos_obs flattened
    - niter (int): diametral iteration

    ### Returns:
    - BH-field (ndarray): m_group*n x 3 arr of BH-vectors
    """
    m_group = len(group)
    n = len(poso_flat)
    magv = np.empty((m_group*n,3))
    dimv = np.empty((m_group*n,2))
    posv = np.empty((m_group*n,3))
    rotv = np.empty((m_group*n,3))
    for i,s in enumerate(group):
        magv[i*n:(i+1)*n] = np.tile(s._mag, (n,1))
        dimv[i*n:(i+1)*n] = np.tile(s._dim, (n,1))
        posv[i*n:(i+1)*n] = np.tile(s._pos, (n,1))
        rotv[i*n:(i+1)*n] = np.tile(s._rot.as_rotvec(), (n,1))
    posov = np.tile(poso_flat,(m_group,1))
    rotobj = R.from_rotvec(rotv)
    B = getBH_core(bh=bh, src_type='Cylinder', mag=magv, dim=dimv, pos=posv, pos_obs=posov, rot=rotobj, niter=niter)
    return B


def getBH(**kwargs: dict) -> np.ndarray:
    """Field computation (level2) for given sources.

    ### Args:
    - bh (bool): True=getB, False=getH
    - sources (M sequence): a 1D tuple of M input sources/collections
    - pos_obs (N1 x N2 x ... x 3 vector): observer positions
    - sumup (bool): default=False returns [B1,B2,...Bm], True returns sum(Bi)
    - niter (int): default=50, for Cylinder sources diametral iteration

    ### Returns:
    - B/H-field (M x N1 x N2 x ... x 3 ndarray): B(mT)H(kA/m) field of each source at pos_obs

    ### Info (level2):
    This function wraps the level 1 field computations.
    - secures input types (list/tuple -> ndarray)
    - generates correct vector input format for getB_level1
    - groups similar sources for combined computation
    - adjust Bfield output format to pos_obs and sources input format
    - input checks
    """

    bh = kwargs['bh']
    sources = kwargs['sources']

    # input checks and type securing --------------------------------
    try:
        sumup = kwargs['sumup']
    except KeyError:
        sumup = False

    try: 
        poso = np.array(kwargs['pos_obs'], dtype=np.float)
    except KeyError:
        print('ERROR getBH: missing pos_obs input')
        sys.exit()
    
    # formatting ----------------------------------------------------
    # flatten out Collections
    src_list = format_src_input(sources)

    # determine shape of positions and flatten into Nx3 array
    poso_shape = poso.shape
    n = np.prod(poso_shape[:-1],dtype=int) # pylint: disable=unsubscriptable-object
    poso_flat = np.reshape(poso,(n,3))

    m = len(src_list)      # number of sources
    B = np.empty((m,n,3))  # store fields here

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
            print('WARNING getB: bad source input !')
            sys.exit()

    # each non-empty group is then evaluated in one go---------------
    # Box group <<<<<<<<<<<<<<<<<<<<<
    group = src_sorted[0]  
    if group: # is empty ?
        m_group = len(group)
        B_group = getBH_box_group(bh, group, poso_flat)
        for i in range(m_group):
            B[order[0][i]] = B_group[i*n:(i+1)*n]

    # Cylinder group <<<<<<<<<<<<<<<<
    group = src_sorted[1]  
    if group: # is empty ?
        try:
            niter = kwargs['niter']
        except KeyError:
            niter = 50
        m_group = len(group)
        B_group = getBH_cylinder_group(bh, group, poso_flat, niter)
        for i in range(m_group):
            B[order[1][i]] = B_group[i*n:(i+1)*n]

    # bring to correct shape (B.shape = pos_obs.shape)---------------
    B = B.reshape(np.r_[m,poso_shape])

    # only one source: reduce highest level
    if m == 1:
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
def getB(*sources: Sequence, **kwargs: dict) -> np.ndarray:
    """ Compute the B-field for a list of given sources.

    ### Args:
    - sources (sequence of M sources): can be sources or Collections
    - pos_obs (N1 x N2 x ... x 3 vector): observer positions
    - sumup (bool): default=False returns [B1,B2,...Bm], True returns sum(Bi)
    
    ### Args-specific:
    - niter (int): default=50, for Cylinder sources diametral iteration

    ### Returns:
    - B-field (M x N1 x N2 x ... x 3 ndarray): B-field of each source at pos_obs in units of mT

    ### Info:
    This function groups similar sources together for optimal vectorization 
    of the computation in one go. For performance call this function as little
    as possible, do not use it in a loop if not absolutely necessary.

    This function will be extended in the future to cover source paths, sensors and
    sensor paths.
    """
    return getBH(bh=True, sources=sources, **kwargs)


def getH(*sources: Sequence, **kwargs: dict) -> np.ndarray:
    """ Compute the H-field for a list of given sources.

    ### Args:
    - sources (sequence of M sources): can be sources or Collections
    - pos_obs (N1 x N2 x ... x 3 vector): observer positions
    - sumup (bool): default=False returns [H1,H2,...Hm], True returns sum(Hi)
    
    ### Args-specific:
    - niter (int): default=50, for Cylinder sources diametral iteration

    ### Returns:
    - H-field (M x N1 x N2 x ... x 3 ndarray): H-field of each source at pos_obs in units of kA/m

    ### Info:
    This function groups similar sources together for optimal vectorization 
    of the computation in one go. For performance call this function as little
    as possible, do not use it in a loop if not absolutely necessary.

    This function wil be extended in the future to cover source paths, sensors and
    sensor paths.
    """
    return getBH(bh=False, sources=sources, **kwargs)


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
    return getBHv(bh=True,**kwargs)

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
