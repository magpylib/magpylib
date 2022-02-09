""" getBHv wrapper codes"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.utility import LIBRARY_BH_DICT_SOURCE_STRINGS

def getBH_dict_level2(**kwargs: dict) -> np.ndarray:
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
        src_type = kwargs['source_type']
        if src_type not in LIBRARY_BH_DICT_SOURCE_STRINGS:
            msg = f'sources input string must be one of {LIBRARY_BH_DICT_SOURCE_STRINGS}'
            raise MagpylibBadUserInput(msg)

        poso = np.array(kwargs['observer'], dtype=float)
        tile_params['observer'] = (poso,2)    # <-- (input,tdim)

        # optional general inputs -------------------
        # if no input set pos=0
        pos = np.array(kwargs.get('position', (0,0,0)), dtype=float)
        tile_params['position'] = (pos,2)
        # if no input set rot=unit
        rot = kwargs.get('orientation', R.from_quat((0,0,0,1)))
        tile_params['orientation'] = (rot.as_quat(),2)
        # if no input set squeeze=True
        squeeze = kwargs.get('squeeze', True)

        # mandatory class specific inputs -----------
        if src_type == 'Cuboid':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,2)

        elif src_type == 'Cylinder':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,2)

        elif src_type == 'CylinderSegment':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dim = np.array(kwargs['dimension'], dtype=float)
            tile_params['dimension'] = (dim,2)

        elif src_type == 'Sphere':
            mag = np.array(kwargs['magnetization'], dtype=float)
            tile_params['magnetization'] = (mag,2)
            dia = np.array(kwargs['diameter'], dtype=float)
            tile_params['diameter'] = (dia,1)

        elif src_type == 'Dipole':
            moment = np.array(kwargs['moment'], dtype=float)
            tile_params['moment'] = (moment,2)

        elif src_type == 'Loop':
            current = np.array(kwargs['current'], dtype=float)
            tile_params['current'] = (current,1)
            dia = np.array(kwargs['diameter'], dtype=float)
            tile_params['diameter'] = (dia,1)

        elif src_type == 'Line':
            current = np.array(kwargs['current'], dtype=float)
            tile_params['current'] = (current,1)
            pos_start = np.array(kwargs['segment_start'], dtype=float)
            tile_params['segment_start'] = (pos_start,2)
            pos_end = np.array(kwargs['segment_end'], dtype=float)
            tile_params['segment_end'] = (pos_end,2)

    except KeyError as kerr:
        msg = f'Missing input keys: {str(kerr)}'
        raise MagpylibBadUserInput(msg) from kerr
    except TypeError as terr:
        msg1='Bad user input type. When sources argument is a string,'
        msg2=' all other inputs must be scalar or array_like.'
        raise MagpylibBadUserInput(msg1+msg2) from terr

    # auto tile 1D parameters ---------------------------------------

    # evaluation vector length
    ns = [len(val) for val,tdim in tile_params.values() if val.ndim == tdim]
    if len(set(ns)) > 1:
        msg = f'Input array lengths must be similar. Instead received {str(set(ns))}'
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
    kwargs['orientation'] = R.from_quat(kwargs['orientation'])

    # compute and return B
    B = getBH_level1(**kwargs)

    if squeeze:
        return np.squeeze(B)
    return B


# def getB_dict(**kwargs):
#     """
#     B-Field computation in units of [mT] from a dictionary of input vectors of
#     length N.

#     This function avoids the object-oriented Magpylib interface and gives direct
#     access to the field implementations. It is the fastest way to compute fields
#     with Magpylib.

#     "Static" inputs of shape (x,) will automatically be tiled to shape (N,x) to
#     fit with other inputs.

#     Required inputs depend on chosen source_type!

#     Parameters
#     ----------
#     source_type: string
#         Source type for computation. Must be either 'Cuboid', 'Cylinder', 'Cylinder_old',
#           'Sphere',
#         'Dipole', 'Loop' or 'Line'. Expected input parameters depend on source_type.

#     position: array_like, shape (3,) or (N,3), default=(0,0,0)
#         Source positions in units of [mm].

#     orientation: scipy Rotation object, default=unit rotation
#         Source rotations relative to the initial state (see object docstrings).

#     observer: array_like, shape (3,) or (N,3)
#         Observer positions in units of [mm].

#     squeeze: bool, default=True
#         If True, the output is squeezed, i.e. all axes of length 1 in the output are eliminated.

#     magnetization: array_like, shape (3,) or (N,3)
#         Only `source_type in ('Cuboid', 'Cylinder', 'Sphere')`! Magnetization vector (mu0*M) or
#         remanence field of homogeneous magnet magnetization in units of [mT].

#     moment:  array_like, shape (3,) or (N,3)
#         Only `source_type = 'Moment'`! Magnetic dipole moment in units of [mT*mm^3]. For
#         homogeneous magnets the relation is moment = magnetization*volume.

#     current: array_like, shape (N,)
#         Only `source_type in ('Line', 'Loop')`! Current flowing in loop in units of [A].

#     dimension: array_like
#         Only `source_type in ('Cuboid', 'Cylinder', 'CylinderSegment')`! Magnet dimension
#         input in units of [mm]. Dimension format of sources similar as in object oriented
#         interface.

#     diameter: array_like, shape (N)
#         Only `source_type in (Sphere, Loop)`! Diameter of source in units of [mm].

#     segment_start: array_like, shape (N,3)
#         Only `source_type = 'Line'`! Start positions of line current segments in units of [mm].

#     segment_end: array_like, shape (N,3)
#         Only `source_type = 'Line'`! End positions of line current segments in units of [mm].

#     Returns
#     -------
#     B-field: ndarray, shape (N,3)
#         B-field generated by sources at observer positions in units of [mT].

#     Examples
#     --------

#     Three-fold evaluation of the dipole field. For each computation the moment is (100,100,100).

#     >>> import magpylib as magpy
#     >>> B = magpy.getB_dict(
#     >>>     source_type='Dipole',
#     >>>     position=[(1,2,3), (2,3,4), (3,4,5)],
#     >>>     moment=(100,100,100),
#     >>>     observer=[(1,1,1), (2,2,2), (3,3,3)])
#     >>> print(B)
#     [[-0.71176254  0.56941003  1.85058261]
#      [-0.71176254  0.56941003  1.85058261]
#      [-0.71176254  0.56941003  1.85058261]]

#     Six-fold evaluation of a Cuboid magnet field with increasing size and magnetization
#     of the magnet. Position and orientation are by default (0,0,0) and unit-orientation,
#     respectively. The observer position is (1,2,3) for each evaluation.

#     >>> import numpy as np
#     >>> import magpylib as magpy
#     >>> B = magpy.getB_dict(
#     >>>     source_type='Cuboid',
#     >>>     magnetization = [(0,0,m) for m in np.linspace(500,1000,6)],
#     >>>     dimension = [(a,a,a) for a in np.linspace(1,2,6)],
#     >>>     observer=(1,2,3))
#     >>> print(B)
#     [[ 0.48818967  0.97689261  0.70605984]
#      [ 1.01203491  2.02636222  1.46575704]
#      [ 1.87397714  3.756164    2.72063422]
#      [ 3.19414311  6.41330652  4.65485356]
#      [ 5.10909461 10.2855981   7.4881383 ]
#      [ 7.76954697 15.70382556 11.48192812]]


#     """
#     return getBH_dict_level2(field='B', **kwargs)


# # ON INTERFACE
# def getH_dict(**kwargs):
#     """
#     H-Field computation in units of [kA/m] from a dictionary of input vectors of
#     length N.

#     This function avoids the object-oriented Magpylib interface and gives direct
#     access to the field implementations. It is the fastest way to compute fields
#     with Magpylib.

#     "Static" inputs of shape (x,) will automatically be tiled to shape (N,x) to
#     fit with other inputs.

#     Required inputs depend on chosen source_type!

#     Parameters
#     ----------
#     source_type: string
#         Source type for computation. Must be either 'Cuboid', 'Cylinder','Cylinder_old',
#           'Sphere',
#         'Dipole', 'Loop' or 'Line'. Expected input parameters depend on source_type.

#     position: array_like, shape (3,) or (N,3), default=(0,0,0)
#         Source positions in units of [mm].

#     orientation: scipy Rotation object, default=unit rotation
#         Source rotations relative to the initial state (see object docstrings).

#     observer: array_like, shape (3,) or (N,3)
#         Observer positions in units of [mm].

#     squeeze: bool, default=True
#         If True, the output is squeezed, i.e. all axes of length 1 in the output are eliminated.

#     magnetization: array_like, shape (3,) or (N,3)
#         Only `source_type in ('Cuboid', 'Cylinder', 'Sphere')`! Magnetization vector (mu0*M) or
#         remanence field of homogeneous magnet magnetization in units of [mT].

#     moment:  array_like, shape (3,) or (N,3)
#         Only `source_type = 'Moment'`! Magnetic dipole moment in units of [mT*mm^3]. For
#         homogeneous magnets the relation is moment = magnetization*volume.

#     current: array_like, shape (N,)
#         Only `source_type in ('Line', 'Loop')`! Current flowing in loop in units of [A].

#     dimension: array_like
#         Only `source_type in ('Cuboid', 'Cylinder', 'CylinderSegment')`! Magnet dimension
#         input in units of [mm]. Dimension format of sources similar as in object oriented
#         interface.

#     diameter: array_like, shape (N)
#         Only `source_type in (Sphere, Loop)`! Diameter of source in units of [mm].

#     segment_start: array_like, shape (N,3)
#         Only `source_type = 'Line'`! Start positions of line current segments in units of [mm].

#     segment_end: array_like, shape (N,3)
#         Only `source_type = 'Line'`! End positions of line current segments in units of [mm].

#     Returns
#     -------
#     H-field: ndarray, shape (N,3)
#         H-field generated by sources at observer positions in units of [kA/m].

#     Examples
#     --------

#     Three-fold evaluation of the dipole field. For each computation the moment is (100,100,100).

#     >>> import magpylib as magpy
#     >>> H = magpy.getH_dict(
#     >>>     source_type='Dipole',
#     >>>     position=[(1,2,3), (2,3,4), (3,4,5)],
#     >>>     moment=(100,100,100),
#     >>>     observer=[(1,1,1), (2,2,2), (3,3,3)])
#     >>> print(H)
#     [[-0.56640264  0.45312211  1.47264685]
#      [-0.56640264  0.45312211  1.47264685]
#      [-0.56640264  0.45312211  1.47264685]]

#     Six-fold evaluation of a Cuboid magnet field with increasing size and magnetization
#     of the magnet. Position and orientation are (0,0,0) and unit-orientation, respectively,
#     by default. The observer position is (1,2,3) for each evaluation.

#     >>> import numpy as np
#     >>> import magpylib as magpy
#     >>> H = magpy.getH_dict(
#     >>>     source_type='Cuboid',
#     >>>     magnetization = [(0,0,m) for m in np.linspace(500,1000,6)],
#     >>>     dimension = [(a,a,a) for a in np.linspace(1,2,6)],
#     >>>     observer=(1,2,3))
#     >>> print(H)
#     [[ 0.388489    0.77738644  0.56186457]
#      [ 0.80535179  1.61252782  1.16641239]
#      [ 1.49126363  2.98906034  2.16501192]
#      [ 2.54181833  5.10354717  3.70421476]
#      [ 4.06568831  8.1850189   5.95887113]
#      [ 6.18280903 12.49670731  9.13702808]]

#     """
#     return getBH_dict_level2(field='H', **kwargs)
