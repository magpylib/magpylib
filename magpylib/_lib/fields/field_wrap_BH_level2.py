import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.utility import (all_same, check_static_sensor_orient,
    format_src_inputs, format_obs_inputs)
from magpylib._lib.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._lib.exceptions import MagpylibBadUserInput, MagpylibInternalError
from magpylib import _lib


def tile_mag(group: list, n_pp: int):
    """ tile up magnetizations of shape (3,)
    """
    magv = np.array([np.tile(src.magnetization, n_pp) for src in group])
    magv = magv.reshape((-1, 3))
    return magv


def tile_dim2(group: list, n_pp: int):
    """ tile up dimensions of shape (2,) and bigger
    """
    dimv = np.array([np.tile(src.dimension, n_pp) for src in group])
    dimv = dimv.reshape((-1,len(group[0].dimension)))
    return dimv


def tile_dim1(group: list, n_pp: int):
    """ tile up 1 dimensional dimensions (Sphere, Circular, ...)
    """
    dimv = np.array([np.tile(src.dimension, n_pp) for src in group])
    dimv = dimv.flatten()
    return dimv


def tile_moment(group: list, n_pp: int):
    """ tile up moments of shape (3,)
    """
    momv = np.array([np.tile(src.moment, n_pp) for src in group])
    momv = momv.reshape((-1, 3))
    return momv


def tile_current(group: list, n_pp: int):
    """ tile up 1 dimensional dimensions (Sphere, Circular, ...)
    """
    currv = np.array([np.tile(src.current, n_pp) for src in group])
    currv = currv.flatten()
    return currv


def get_src_dict(group: list, n_pix: int, n_pp: int, poso: np.ndarray) -> dict:
    """
    create dictionary for level1 input
    """
    # pylint: disable=protected-access

    # tile up basic attributes that all sources have
    # position
    posv = np.array([np.tile(src._position, n_pix).reshape(n_pp,3) for src in group])
    posv = posv.reshape((-1, 3))
    # orientation
    rotv = np.array([np.tile(src._orientation.as_quat(),n_pix).reshape(n_pp,4) for src in group])
    rotv = rotv.reshape((-1, 4))
    rotobj = R.from_quat(rotv)
    # pos_obs
    posov = np.tile(poso, (len(group),1))

    # determine which group we are dealing with
    src_type = group[0].object_type

    if src_type == 'Sphere':
        magv = tile_mag(group, n_pp)
        dimv = tile_dim1(group, n_pp)
        return {'src_type':src_type, 'magnetization':magv, 'dimension':dimv, 'position':posv,
            'pos_obs': posov, 'orientation':rotobj}

    if src_type in {'Box', 'Cylinder'}:
        magv = tile_mag(group, n_pp)
        dimv = tile_dim2(group, n_pp)
        return {'src_type':src_type, 'magnetization':magv, 'dimension':dimv, 'position':posv,
            'pos_obs': posov, 'orientation':rotobj}

    if src_type == 'Dipole':
        momv = tile_moment(group, n_pp)
        return {'src_type':src_type, 'moment':momv, 'position':posv,
            'pos_obs': posov, 'orientation':rotobj}

    if src_type == 'Circular':
        currv = tile_current(group, n_pp)
        dimv = tile_dim1(group, n_pp)
        return {'src_type':src_type, 'current':currv, 'dimension':dimv, 'position':posv,
            'pos_obs': posov, 'orientation':rotobj}

    if src_type == 'Line':
        # get_BH_line_from_vert function tiles internally !
        #currv = tile_current(group, n_pp)
        currv = np.array([src.current for src in group])
        vert_list = [src.vertices for src in group]
        return {'src_type':src_type, 'current':currv, 'vertices':vert_list,
            'position':posv, 'pos_obs': posov, 'orientation':rotobj}

    raise MagpylibInternalError('Bad src_type in get_src_dict')


def getBH_level2(bh, sources, observers, sumup, squeeze) -> np.ndarray:
    """...

    Parameters
    ----------
    - bh (bool): True=getB, False=getH
    - sources (src_obj or list): source object or 1D list of L sources/collections with similar
        pathlength M and/or 1.
    - observers (sens_obj or list or pos_obs): pos_obs or sensor object or 1D list of K sensors with
        similar pathlength M and/or 1 and sensor pos_pix of shape (N1,N2,...,3).
    - sumup (bool): default=False returns [B1,B2,...] for every source, True returns sum(Bi)
        for all sources.
    - squeeze (bool): default=True, If True output is squeezed (axes of length 1 are eliminated)

    Returns
    -------
    field: ndarray, shape squeeze((L,M,K,N1,N2,...,3)), field of L sources, M path
    positions, K sensors and N1xN2x.. observer positions and 3 field components.

    Info:
    -----
    - generates a 1D list of sources (collections flattened) and a 1D list of sensors from input
    - tile all paths of static (path_length=1) objects
    - combine all sensor pixel positions for joint evaluation
    - group similar source types for joint evaluation
    - compute field and store in allocated array
    - rearrange the array in the shape squeeze((L, M, K, N1, N2, ...,3))
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # avoid circular imports --------------------------------------------------
    Box = _lib.obj_classes.Box
    Cylinder = _lib.obj_classes.Cylinder
    Sphere = _lib.obj_classes.Sphere
    Collection = _lib.obj_classes.Collection
    Dipole = _lib.obj_classes.Dipole
    Circular = _lib.obj_classes.Circular
    Line = _lib.obj_classes.Line

    # CHECK AND FORMAT INPUT ---------------------------------------------------

    # format sources input:
    #   input: allow only bare src objects or 1D lists/tuple of src and col
    #   out: sources = ordered list of sources
    #   out: src_list = ordered list of sources with flattened collections
    sources, src_list = format_src_inputs(sources)

    # format observer inputs:
    #   allow only bare sensor, possis, or a list/tuple thereof
    #   create a list of sensor instances where possi inputs are moved to pos_pix
    sensors = format_obs_inputs(observers)

    # check if all sensor pixel shapes are similar.
    #   Cannot accept different obs pos input shapes as this would lead to incomplete
    #   axes in the return arrays.
    pix_shapes = [sens.pos_pix.shape for sens in sensors]
    if not all_same(pix_shapes):
        raise MagpylibBadUserInput('Different observer input shapes not allowed.'+
            ' All pos_pix and pos_obs input shapes must be similar !')
    pix_shape = pix_shapes[0]

    # check which sensors have unit roation
    #   so that they dont have to be rotated back later (performance issue)
    #   this check is made now when sensor paths are not yet tiled.
    unitQ = np.array([0,0,0,1.])
    unrotated_sensors = [all(all(r==unitQ) for r in sens._orientation.as_quat()) for sens in sensors]

    # check which sensors have a static orientation
    #   either static sensor or translation path
    #   later such sensors require less back-rotation effort (performance issue)
    static_sensor_rot = check_static_sensor_orient(sensors)

    # some important quantities -------------------------------------------------
    obj_list = set(src_list + sensors) # unique obj entries only !!!
    l0 = len(sources)
    l = len(src_list)
    k = len(sensors)

    # tile up paths -------------------------------------------------------------
    #   all obj paths that are shorter than max-length are filled up with the last
    #   postion/orientation of the object (static paths)
    path_lengths = [len(obj._position) for obj in obj_list]
    m = max(path_lengths)

    # objects to tile up and reset below
    mask_reset = [m!=pl for pl in path_lengths]
    reset_obj = [obj for obj,mask in zip(obj_list,mask_reset) if mask]
    reset_obj_m0 = [pl for pl,mask in zip(path_lengths,mask_reset) if mask]

    if m>1:
        for obj,m0 in zip(reset_obj, reset_obj_m0):
            # length to be tiled
            m_tile = m-m0
            # tile up position
            tile_pos = np.tile(obj._position[-1], (m_tile,1))
            obj.position = np.concatenate((obj._position, tile_pos))
            # tile up orientation
            tile_orient = np.tile(obj._orientation.as_quat()[-1], (m_tile,1))
            tile_orient = np.concatenate((obj._orientation.as_quat(), tile_orient))
            obj.orientation = R.from_quat(tile_orient)

    # combine information form all sensors to generate pos_obs with-------------
    #   shape (m * concat all sens flat pos_pix, 3)
    #   allows sensors with different pos_pix shapes <- relevant?
    poso =[[r.apply(sens.pos_pix.reshape(-1,3)) + p
            for r,p in zip(sens._orientation, sens._position)]
           for sens in sensors]
    poso = np.concatenate(poso,axis=1).reshape(-1,3)
    n_pp = len(poso)
    n_pix = int(n_pp/m)

    # group similar source types----------------------------------------------
    src_sorted = [[],[],[],[],[],[]]   # store groups here
    order = [[],[],[],[],[],[]]        # keep track of the source order
    for i,src in enumerate(src_list):
        if isinstance(src, Box):
            src_sorted[0] += [src]
            order[0] += [i]
        elif isinstance(src, Cylinder):
            src_sorted[1] += [src]
            order[1] += [i]
        elif isinstance(src, Sphere):
            src_sorted[2] += [src]
            order[2] += [i]
        elif isinstance(src, Dipole):
            src_sorted[3] += [src]
            order[3] += [i]
        elif isinstance(src, Circular):
            src_sorted[4] += [src]
            order[4] += [i]
        elif isinstance(src, Line):
            src_sorted[5] += [src]
            order[5] += [i]

    # evaluate each group in one vectorized step -------------------------------
    B = np.empty((l,m,n_pix,3))                               # allocate B
    for iii, group in enumerate(src_sorted):
        if group:                                             # is group empty ?
            lg = len(group)
            src_dict = get_src_dict(group, n_pix, n_pp, poso) # compute array dict for level1
            B_group = getBH_level1(bh=bh, **src_dict)         # compute field
            B_group = B_group.reshape((lg,m,n_pix,3))         # reshape (2% slower for large arrays)
            for i in range(lg):                               # put into dedicated positions in B
                B[order[iii][i]] = B_group[i]

    # reshape output ----------------------------------------------------------------
    # rearrange B when there is at least one Collection with more than one source
    if l > l0:
        for i,src in enumerate(sources):
            if isinstance(src, Collection):
                col_len = len(src.sources)
                B[i] = np.sum(B[i:i+col_len],axis=0)    # set B[i] to sum of slice
                B = np.delete(B,np.s_[i+1:i+col_len],0) # delete remaining part of slice

    # apply sensor rotations (after summation over collections to reduce rot.apply operations)
    #   note: replace by math.prod with python 3.8 or later
    k_pixel = int(np.product(pix_shape[:-1])) # total number of pixel positions
    for i,sens in enumerate(sensors):         # cylcle through all sensors
        if not unrotated_sensors[i]:          # apply operations only to rotated sensors
            if static_sensor_rot[i]:          # special case: same rotation along path
                # select part where rot is applied
                Bpart = B[:,:,i*k_pixel:(i+1)*k_pixel]
                # change shape from (l0,m,k_pixel,3) to (P,3) for rot package
                Bpart_flat = np.reshape(Bpart, (k_pixel*l0*m,3))
                # apply sensor rotation
                Bpart_flat_rot = sens._orientation[0].inv().apply(Bpart_flat)
                # overwrite Bpart in B
                B[:,:,i*k_pixel:(i+1)*k_pixel] = np.reshape(Bpart_flat_rot, (l0,m,k_pixel,3))
            else:                         # general case: different rotations along path
                for j in range(m): # THIS LOOP IS EXTREMELY SLOW !!!! github issue #283
                    Bpart = B[:,j,i*k_pixel:(i+1)*k_pixel]           # select part
                    Bpart_flat = np.reshape(Bpart, (k_pixel*l0,3))   # flatten for rot package
                    Bpart_flat_rot = sens._orientation[j].inv().apply(Bpart_flat)  # apply rotation
                    B[:,j,i*k_pixel:(i+1)*k_pixel] = np.reshape(Bpart_flat_rot, (l0,k_pixel,3)) # ov

    # rearrange sensor-pixel shape
    sens_px_shape = (k,) + pix_shape
    B = B.reshape((l0,m)+sens_px_shape)

    #
    if sumup:
        B = np.sum(B, axis=0)

    # reduce all size-1 levels
    if squeeze:
        B = np.squeeze(B)

    # reset tiled objects
    for obj,m0 in zip(reset_obj, reset_obj_m0):
        obj.position = obj.position[:m0]
        obj.orientation = obj.orientation[:m0]

    return B
