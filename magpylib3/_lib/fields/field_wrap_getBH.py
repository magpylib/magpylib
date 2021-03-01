import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib3 as mag3
from magpylib3._lib.math_utility import format_src_input, same_path_length, check_allowed_keys
from magpylib3._lib.config import Config
from magpylib3._lib.fields.field_wrap_BH_level1 import getBH_level1


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
    # pylint: disable=protected-access

    l_group = len(group)    # sources in group
    m = len(group[0]._pos)  # path length
    n = len(poso_flat)      # no. observer pos
    len_dim = len(group[0]._dim)

    # prepare and fill arrays, shape: (l_group, m, n)
    magv = np.empty((l_group*m*n,3))
    dimv = np.empty((l_group*m*n,len_dim))
    posv = np.empty((l_group*m*n,3))
    rotv = np.empty((l_group*m*n,4))
    for i,src in enumerate(group):
        magv[i*m*n:(i+1)*m*n] = np.tile(src.mag, (m*n,1))
        dimv[i*m*n:(i+1)*m*n] = np.tile(src.dim, (m*n,1))
        posv[i*m*n:(i+1)*m*n] = np.tile(src._pos,n).reshape(m*n,3)
        rotv[i*m*n:(i+1)*m*n] = np.tile(src._rot.as_quat(),n).reshape(m*n,4)
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
    field: ndarray, shape (L,M,N1,N2,...,3), field of L sources, M path
    positions and N observer positions

    Info:
    -----
    - input dict checks (unknowns)
    - secure user inputs
    - group similar sources for combined computation
    - generate vector input format for getBH_level1
    - adjust Bfield output format to pos_obs, path, sources input format
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    #pylint: disable=too-many-statements

    # warning if key in kwargs that is not in allowed_keys ---------
    allowed_keys = ['bh', 'sources', 'pos_obs', 'sumup', 'niter']
    check_allowed_keys(allowed_keys, kwargs, 'getBH()')

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
    for i,src in enumerate(src_list):
        if isinstance(src, mag3.magnet.Box):
            src_sorted[0] += [src]
            order[0] += [i]
        elif isinstance(src, mag3.magnet.Cylinder):
            src_sorted[1] += [src]
            order[1] += [i]
        else:
            sys.exit('ERROR: getBH() - bad source input !')

    # evaluate each non-empty group in one go------------------------

    # Box group
    group = src_sorted[0]
    if group:
        src_dict = scr_dict_homo_mag(group, poso_flat)            # compute array dict for level1
        B_group = getBH_level1(bh=bh, src_type='Box', **src_dict) # compute field
        B_group = B_group.reshape((len(group),m,n,3))             # reshape
        for i in range(len(group)):                               # put at dedicated positions in B
            B[order[0][i]] = B_group[i]

    # Cylinder group
    group = src_sorted[1]
    if group:
        niter = kwargs.get('niter', Config.ITER_CYLINDER)
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
        for i,src in enumerate(sources):
            if isinstance(src, mag3.Collection):
                col_len = len(src._sources)
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
