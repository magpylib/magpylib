import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.utility import (
    check_static_sensor_orient,
    format_src_inputs,
)
from magpylib._src.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._src.fields.field_wrap_BH_level2_dict import getBH_dict_level2
from magpylib._src.exceptions import MagpylibBadUserInput, MagpylibInternalError
from magpylib._src.input_checks import (
    check_excitations,
    check_dimensions,
    check_format_input_observers,
)


def tile_group_property(group: list, n_pp: int, prop_name: str):
    """ tile up group property"""
    prop0 = getattr(group[0], prop_name)
    out = np.array([getattr(src, prop_name) for src in group])
    out = np.tile(out, n_pp)
    if np.isscalar(prop0):
        return out.flatten()
    return out.reshape((-1, prop0.shape[0]))


def get_src_dict(group: list, n_pix: int, n_pp: int, poso: np.ndarray) -> dict:
    """ create dictionaries for level1 input
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-return-statements

    # tile up basic attributes that all sources have
    # position
    poss = np.array([src._position for src in group])
    posv = np.tile(poss, n_pix).reshape((-1, 3))

    # orientation
    rots = np.array([src._orientation.as_quat() for src in group])
    rotv = np.tile(rots, n_pix).reshape((-1, 4))
    rotobj = R.from_quat(rotv)

    # pos_obs
    posov = np.tile(poso, (len(group),1))

    # determine which group we are dealing with and tile up dim and excitation
    src_type = group[0]._object_type

    kwargs = {'source_type': src_type, 'position': posv, 'observer': posov, 'orientation': rotobj}

    if src_type in ('Sphere', 'Cuboid', 'Cylinder', 'CylinderSegment'):
        magv = tile_group_property(group, n_pp, 'magnetization')
        kwargs.update(magnetization=magv)
        if src_type=="Sphere":
            diav = tile_group_property(group, n_pp, 'diameter')
            kwargs.update(diameter=diav)
        else:
            dimv = tile_group_property(group, n_pp, 'dimension')
            kwargs.update(dimension=dimv)

    elif src_type == 'Dipole':
        momv = tile_group_property(group, n_pp, 'moment')
        kwargs.update({'moment':momv})

    elif src_type == 'Loop':
        currv = tile_group_property(group, n_pp, 'current')
        diav = tile_group_property(group, n_pp, 'diameter')
        kwargs.update({'current':currv, 'diameter':diav})

    elif src_type == 'Line':
        # get_BH_line_from_vert function tiles internally !
        #currv = tile_current(group, n_pp)
        currv = np.array([src.current for src in group])
        vert_list = [src.vertices for src in group]
        kwargs.update({'current':currv, 'vertices':vert_list})

    elif src_type == 'CustomSource':
        kwargs.update({
            'field_B_lambda': group[0].field_B_lambda,
            'field_H_lambda': group[0].field_H_lambda})

    else:
        raise MagpylibInternalError('Bad source_type in get_src_dict')

    return kwargs


def getBH_level2(sources, observers, **kwargs) -> np.ndarray:
    """...

    Parameters
    ----------
    - bh (bool): True=getB, False=getH
    - sources (src_obj or list): source object or 1D list of L sources/collections with similar
        pathlength M and/or 1.
    - observers (sens_obj or list or pos_obs): pos_obs or sensor object or 1D list of K
        sensors with similar pathlength M and/or 1 and sensor pixel of shape (N1,N2,...,3).
    kwargs:
    - 'sumup' (bool): False returns [B1,B2,...] for every source, True returns sum(Bi)
        for all sources.
    - 'squeeze' (bool): True output is squeezed (axes of length 1 are eliminated)
    - 'pixel_agg' : str: A compatible numpy aggregator string (e.g. `'min', 'max', 'mean'`)
       which applies on pixel output values.
    - 'field' (str): 'B' computes B field, 'H' computes H-field
    - getBH_dict inputs

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

    # CHECK AND FORMAT INPUT ---------------------------------------------------
    if isinstance(sources, str):
        return getBH_dict_level2(
            source_type=sources,
            observer=observers,
            **kwargs
        )

    # bad user inputs mixing getBH_dict kwargs with object oriented interface
    kwargs_check = kwargs.copy()
    for popit in ['field', 'sumup', 'squeeze', 'pixel_agg']:
        kwargs_check.pop(popit, None)
    if kwargs_check:
        raise MagpylibBadUserInput(
            f"Keyword arguments {tuple(kwargs_check.keys())} are only allowed when the source "
            "is defined by a string (e.g. sources='Cylinder')"
        )

    # format sources input:
    #   input: allow only bare src objects or 1D lists/tuple of src and col
    #   out: sources = ordered list of sources
    #   out: src_list = ordered list of sources with flattened collections
    sources, src_list = format_src_inputs(sources)

    # test if all source dimensions and excitations are initialized
    check_dimensions(sources)
    check_excitations(sources)

    # format observer inputs:
    #   allow only bare sensor, collection, pos_vec or list thereof
    #   transform input into an ordered list of sensors (pos_vec->pixel)
    #   check if all pixel shapes are similar.
    sensors = check_format_input_observers(observers)
    pix_shape = sensors[0].pixel.shape
    if pix_shape == (3,):
        pix_shape = (1,3)

    # check which sensors have unit roation
    #   so that they dont have to be rotated back later (performance issue)
    #   this check is made now when sensor paths are not yet tiled.
    unitQ = np.array([0,0,0,1.])
    unrotated_sensors = [all(all(r==unitQ)
        for r in sens._orientation.as_quat())
        for sens in sensors]

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
    max_path_len = max(path_lengths)

    # objects to tile up and reset below
    mask_reset = [max_path_len!=pl for pl in path_lengths]
    reset_obj = [obj for obj,mask in zip(obj_list,mask_reset) if mask]
    reset_obj_m0 = [pl for pl,mask in zip(path_lengths,mask_reset) if mask]

    if max_path_len>1:
        for obj,m0 in zip(reset_obj, reset_obj_m0):
            # length to be tiled
            m_tile = max_path_len-m0
            # tile up position
            tile_pos = np.tile(obj._position[-1], (m_tile,1))
            obj._position = np.concatenate((obj._position, tile_pos))
            # tile up orientation
            tile_orient = np.tile(obj._orientation.as_quat()[-1], (m_tile,1))
            # FUTURE use Rotation.concatenate() requires scipy>=1.8 and python 3.8
            tile_orient = np.concatenate((obj._orientation.as_quat(), tile_orient))
            obj._orientation = R.from_quat(tile_orient)

    # combine information form all sensors to generate pos_obs with-------------
    #   shape (m * concat all sens flat pixel, 3)
    #   allows sensors with different pixel shapes <- relevant?
    poso =[[r.apply(sens.pixel.reshape(-1,3)) + p
            for r,p in zip(sens._orientation, sens._position)]
           for sens in sensors]
    poso = np.concatenate(poso,axis=1).reshape(-1,3)
    n_pp = len(poso)
    n_pix = int(n_pp/max_path_len)

    # group similar source types----------------------------------------------
    groups = {}
    for ind,src in enumerate(src_list):
        if src._object_type=='CustomSource':
            group_key = src.field_B_lambda if kwargs['field']=='B' else src.field_H_lambda
        else:
            group_key = src._object_type
        if group_key not in groups:
            groups[group_key] = {'sources':[], 'order':[], 'source_type': src._object_type}
        groups[group_key]['sources'].append(src)
        groups[group_key]['order'].append(ind)

    # evaluate each group in one vectorized step -------------------------------
    B = np.empty((l,max_path_len,n_pix,3))                         # allocate B
    for group in groups.values():
        lg = len(group['sources'])
        gr = group['sources']
        src_dict = get_src_dict(gr, n_pix, n_pp, poso)  # compute array dict for level1
        B_group = getBH_level1(field=kwargs['field'], **src_dict)  # compute field
        B_group = B_group.reshape((lg,max_path_len,n_pix,3))  # reshape (2% slower for large arrays)
        for si in range(lg):                             # put into dedicated positions in B
            B[group['order'][si]] = B_group[si]

    # reshape output ----------------------------------------------------------------
    # rearrange B when there is at least one Collection with more than one source
    if l > l0:
        for si,src in enumerate(sources):
            if src._object_type == 'Collection':
                col_len = len(src.sources)
                B[si] = np.sum(B[si:si+col_len],axis=0)    # set B[i] to sum of slice
                B = np.delete(B,np.s_[si+1:si+col_len],0) # delete remaining part of slice

    # apply sensor rotations (after summation over collections to reduce rot.apply operations)
    #   note: replace by math.prod with python 3.8 or later
    pix_tot = int(np.product(pix_shape[:-1])) # total number of pixel positions
    for si,sens in enumerate(sensors):         # cycle through all sensors
        if not unrotated_sensors[si]:          # apply operations only to rotated sensors
            # select part where rot is applied
            Bpart = B[:,:,si*pix_tot:(si+1)*pix_tot]
            # change shape from (l0,m,k_pixel,3) to (P,3) for rot package
            Bpart_flat = np.reshape(Bpart, (pix_tot*l0*max_path_len,3))
            # apply sensor rotation
            if static_sensor_rot[si]:          # special case: same rotation along path
                sens_orient = sens._orientation[0]
            else:
                # FUTURE use R.concatenate() requires scipy>=1.8 and python 3.8
                # sens_orient = R.concatenate([sens._orientation]*l0)
                sens_orient = R.from_quat(np.concatenate([sens._orientation.as_quat()]*l0))
            Bpart_flat_rot = sens_orient.inv().apply(Bpart_flat)
            # overwrite Bpart in B
            B[:,:,si*pix_tot:(si+1)*pix_tot] = np.reshape(
                Bpart_flat_rot, (l0,max_path_len,pix_tot,3)
            )

    # rearrange sensor-pixel shape
    sens_px_shape = (k,) + pix_shape
    B = B.reshape((l0,max_path_len)+sens_px_shape)

    # sumup over sources
    if kwargs['sumup']:
        B = np.sum(B, axis=0, keepdims=True)

    # aggregate pixel values
    pixel_agg = kwargs['pixel_agg']
    if pixel_agg is not None:
        B = getattr(np, pixel_agg)(B, axis=tuple(range(3-B.ndim,-1)))
        if not kwargs['squeeze']:
            # add missing dimension since `pixel_agg` reduces pixel
            # dimensions to zero. Only needed if `squeeze is False``
            B = np.expand_dims(B, axis=-2)

    # reduce all size-1 levels
    if kwargs['squeeze']:
        B = np.squeeze(B)

    # reset tiled objects
    for obj,m0 in zip(reset_obj, reset_obj_m0):
        obj._position = obj._position[:m0]
        obj._orientation = obj._orientation[:m0]

    return B
