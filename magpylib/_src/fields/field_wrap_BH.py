"""Field computation structure:

level0:(field_BH_XXX.py files)
    - pure vectorized field computations from literature
    - all computations in source CS
    - distinguish B/H

level1(getBH_level1):
    - apply transformation to global CS
    - select correct level0 src_type computation
    - input dict, no input checks !

level2(getBHv_level2):  <--- DIRECT ACCESS TO FIELD COMPUTATION FORMULAS, INPUT = DICT OF ARRAYS
    - input dict checks (unknowns)
    - secure user inputs
    - check input for mandatory information
    - set missing input variables to default values
    - tile 1D inputs

level2(getBH_level2):   <--- COMPUTE FIELDS FROM SOURCES
    - input dict checks (unknowns)
    - secure user inputs
    - group similar sources for combined computation
    - generate vector input format for getBH_level1
    - adjust Bfield output format to (pos_obs, path, sources) input format

level3(getB, getH, getB_dict, getH_dict): <--- USER INTERFACE
    - docstrings
    - separated B and H
    - transform input into dict for level2

level4(src.getB, src.getH):       <--- USER INTERFACE
    - docstrings
    - calling level3 getB, getH directly from sources

level3(getBH_from_sensor):
    - adjust output format to (senors, path, sources) input format

level4(getB_from_sensor, getH_from_sensor): <--- USER INTERFACE

level5(sens.getB, sens.getH): <--- USER INTERFACE
"""
# pylint: disable=cyclic-import
import numbers
import warnings
from itertools import product
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.input_checks import check_dimensions
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_format_input_observers
from magpylib._src.input_checks import check_format_pixel_agg
from magpylib._src.input_checks import check_getBH_output_type
from magpylib._src.utility import check_static_sensor_orient
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import format_src_inputs
from magpylib._src.utility import get_registered_sources


def tile_group_property(group: list, n_pp: int, prop_name: str):
    """tile up group property"""
    out = [getattr(src, prop_name) for src in group]
    if not np.isscalar(out[0]) and any(o.shape != out[0].shape for o in out):
        out = np.asarray(out, dtype="object")
    else:
        out = np.array(out)
    return np.repeat(out, n_pp, axis=0)


def get_src_dict(group: list, n_pix: int, n_pp: int, poso: np.ndarray) -> dict:
    """create dictionaries for level1 input"""
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
    posov = np.tile(poso, (len(group), 1))

    # determine which group we are dealing with and tile up properties

    kwargs = {
        "position": posv,
        "observers": posov,
        "orientation": rotobj,
    }

    src_props = group[0]._field_func_kwargs_ndim

    for prop in src_props:
        if hasattr(group[0], prop) and prop not in (
            "position",
            "orientation",
            "observers",
        ):
            kwargs[prop] = tile_group_property(group, n_pp, prop)

    return kwargs


def getBH_level1(
    *,
    field_func: Callable,
    field: str,
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

    # transform obs_pos into source CS
    pos_rel_rot = orientation.apply(observers - position, inverse=True)

    # compute field
    BH = field_func(field=field, observers=pos_rel_rot, **kwargs)

    # transform field back into global CS
    if BH is not None:  # catch non-implemented field_func a level above
        BH = orientation.apply(BH)

    return BH


def getBH_level2(
    sources, observers, *, field, sumup, squeeze, pixel_agg, output, **kwargs
) -> np.ndarray:
    """Compute field for given sources and observers.

    Parameters
    ----------
    sources : src_obj or list
        source object or 1D list of L sources/collections with similar
        pathlength M and/or 1.
    observers : sens_obj or list or pos_obs
        pos_obs or sensor object or 1D list of K sensors with similar pathlength M
        and/or 1 and sensor pixel of shape (N1,N2,...,3).
    sumup : bool, default=False
        returns [B1,B2,...] for every source, True returns sum(Bi) sfor all sources.
    squeeze : bool, default=True:
        If True output is squeezed (axes of length 1 are eliminated)
    pixel_agg : str
        A compatible numpy aggregator string (e.g. `'min', 'max', 'mean'`)
        which applies on pixel output values.
    field : {'B', 'H'}
        'B' computes B field, 'H' computes H-field
    output: str, default='ndarray'
        Output type, which must be one of `('ndarray', 'dataframe')`. By default a multi-
        dimensional array ('ndarray') is returned. If 'dataframe' is chosen, the function
        returns a 2D-table as a `pandas.DataFrame` object (the Pandas library must be
        installed).

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
    # pylint: disable=import-outside-toplevel

    from magpylib._src.obj_classes.class_Collection import Collection
    from magpylib._src.obj_classes.class_magnet_TriangularMesh import TriangularMesh

    # CHECK AND FORMAT INPUT ---------------------------------------------------
    if isinstance(sources, str):
        return getBH_dict_level2(
            source_type=sources,
            observers=observers,
            field=field,
            squeeze=squeeze,
            **kwargs,
        )

    # bad user inputs mixing getBH_dict kwargs with object oriented interface
    if kwargs:
        raise MagpylibBadUserInput(
            f"Keyword arguments {tuple(kwargs.keys())} are only allowed when the source "
            "is defined by a string (e.g. sources='Cylinder')"
        )

    # format sources input:
    #   input: allow only one bare src object or a 1D list/tuple of src and col
    #   out: sources = ordered list of sources
    #   out: src_list = ordered list of sources with flattened collections
    sources, src_list = format_src_inputs(sources)

    # test if all source dimensions and excitations are initialized
    check_dimensions(src_list)
    check_excitations(src_list)

    # make sure that TriangularMesh sources have a closed mesh when getB is called - warn if not
    if field == "B":
        for src in src_list:
            if isinstance(src, TriangularMesh):
                # unchecked mesh status - may be open
                if src.status_open is None:
                    warnings.warn(
                        f"Unchecked mesh status of {src} detected before B-field computation. "
                        "An open mesh may return bad results."
                    )
                elif src.status_open:  # mesh is open
                    warnings.warn(
                        f"Open mesh of {src} detected before B-field computation. "
                        "An open mesh may return bad results."
                    )

    # format observers input:
    #   allow only bare sensor, collection, pos_vec or list thereof
    #   transform input into an ordered list of sensors (pos_vec->pixel)
    #   check if all pixel shapes are similar - or else if pixel_agg is given
    pixel_agg_func = check_format_pixel_agg(pixel_agg)
    sensors, pix_shapes = check_format_input_observers(observers, pixel_agg)
    pix_nums = [
        int(np.prod(ps[:-1])) for ps in pix_shapes
    ]  # number of pixel for each sensor
    pix_inds = np.cumsum([0] + pix_nums)  # cumulative indices of pixel for each sensor
    pix_all_same = len(set(pix_shapes)) == 1

    # check which sensors have unit rotation
    #   so that they dont have to be rotated back later (performance issue)
    #   this check is made now when sensor paths are not yet tiled.
    unitQ = np.array([0, 0, 0, 1.0])
    unrotated_sensors = [
        all(all(r == unitQ) for r in sens._orientation.as_quat()) for sens in sensors
    ]

    # check which sensors have a static orientation
    #   either static sensor or translation path
    #   later such sensors require less back-rotation effort (performance issue)
    static_sensor_rot = check_static_sensor_orient(sensors)

    # some important quantities -------------------------------------------------
    obj_list = set(src_list + sensors)  # unique obj entries only !!!
    num_of_sources = len(sources)
    num_of_src_list = len(src_list)
    num_of_sensors = len(sensors)

    # tile up paths -------------------------------------------------------------
    #   all obj paths that are shorter than max-length are filled up with the last
    #   position/orientation of the object (static paths)
    path_lengths = [len(obj._position) for obj in obj_list]
    max_path_len = max(path_lengths)

    # objects to tile up and reset below
    mask_reset = [max_path_len != pl for pl in path_lengths]
    reset_obj = [obj for obj, mask in zip(obj_list, mask_reset) if mask]
    reset_obj_m0 = [pl for pl, mask in zip(path_lengths, mask_reset) if mask]

    if max_path_len > 1:
        for obj, m0 in zip(reset_obj, reset_obj_m0):
            # length to be tiled
            m_tile = max_path_len - m0
            # tile up position
            tile_pos = np.tile(obj._position[-1], (m_tile, 1))
            obj._position = np.concatenate((obj._position, tile_pos))
            # tile up orientation
            tile_orient = np.tile(obj._orientation.as_quat()[-1], (m_tile, 1))
            # FUTURE use Rotation.concatenate() requires scipy>=1.8 and python 3.8
            tile_orient = np.concatenate((obj._orientation.as_quat(), tile_orient))
            obj._orientation = R.from_quat(tile_orient)

    # combine information form all sensors to generate pos_obs with-------------
    #   shape (m * concat all sens flat pixel, 3)
    #   allows sensors with different pixel shapes <- relevant?
    poso = [
        [
            r.apply(sens.pixel.reshape(-1, 3)) + p
            for r, p in zip(sens._orientation, sens._position)
        ]
        for sens in sensors
    ]
    poso = np.concatenate(poso, axis=1).reshape(-1, 3)
    n_pp = len(poso)
    n_pix = int(n_pp / max_path_len)

    # group similar source types----------------------------------------------
    field_func_groups = {}
    for ind, src in enumerate(src_list):
        group_key = src.field_func
        if group_key is None:
            raise MagpylibMissingInput(
                f"Cannot compute {field}-field because "
                f"`field_func` of {src} has undefined {field}-field computation."
            )
        if group_key not in field_func_groups:
            field_func_groups[group_key] = {
                "sources": [],
                "order": [],
            }
        field_func_groups[group_key]["sources"].append(src)
        field_func_groups[group_key]["order"].append(ind)

    # evaluate each group in one vectorized step -------------------------------
    B = np.empty((num_of_src_list, max_path_len, n_pix, 3))  # allocate B
    for field_func, group in field_func_groups.items():
        lg = len(group["sources"])
        gr = group["sources"]
        src_dict = get_src_dict(gr, n_pix, n_pp, poso)  # compute array dict for level1
        B_group = getBH_level1(
            field_func=field_func, field=field, **src_dict
        )  # compute field
        if B_group is None:
            raise MagpylibMissingInput(
                f"Cannot compute {field}-field because "
                f"`field_func` {field_func} has undefined {field}-field computation."
            )
        B_group = B_group.reshape(
            (lg, max_path_len, n_pix, 3)
        )  # reshape (2% slower for large arrays)
        for gr_ind in range(lg):  # put into dedicated positions in B
            B[group["order"][gr_ind]] = B_group[gr_ind]

    # reshape output ----------------------------------------------------------------
    # rearrange B when there is at least one Collection with more than one source
    if num_of_src_list > num_of_sources:
        for src_ind, src in enumerate(sources):
            if isinstance(src, Collection):
                col_len = len(format_obj_input(src, allow="sources"))
                # set B[i] to sum of slice
                B[src_ind] = np.sum(B[src_ind : src_ind + col_len], axis=0)
                B = np.delete(
                    B, np.s_[src_ind + 1 : src_ind + col_len], 0
                )  # delete remaining part of slice

    # apply sensor rotations (after summation over collections to reduce rot.apply operations)
    for sens_ind, sens in enumerate(sensors):  # cycle through all sensors
        pix_slice = slice(pix_inds[sens_ind], pix_inds[sens_ind + 1])
        if not unrotated_sensors[sens_ind]:  # apply operations only to rotated sensors
            # select part where rot is applied
            Bpart = B[:, :, pix_slice]
            # change shape to (P,3) for rot package
            Bpart_orig_shape = Bpart.shape
            Bpart_flat = np.reshape(Bpart, (-1, 3))
            # apply sensor rotation
            if static_sensor_rot[sens_ind]:  # special case: same rotation along path
                sens_orient = sens._orientation[0]
            else:
                sens_orient = R.from_quat(
                    np.tile(  # tile for each source from list
                        np.repeat(  # same orientation path index for all indices
                            sens._orientation.as_quat(), pix_nums[sens_ind], axis=0
                        ),
                        (num_of_sources, 1),
                    )
                )
            Bpart_flat_rot = sens_orient.inv().apply(Bpart_flat)
            # overwrite Bpart in B
            B[:, :, pix_slice] = np.reshape(Bpart_flat_rot, Bpart_orig_shape)
        if sens.handedness == "left":
            B[..., pix_slice, 0] *= -1

    # rearrange sensor-pixel shape
    if pix_all_same:
        B = B.reshape((num_of_sources, max_path_len, num_of_sensors, *pix_shapes[0]))
        # aggregate pixel values
        if pixel_agg is not None:
            B = pixel_agg_func(B, axis=tuple(range(3 - B.ndim, -1)))
    else:  # pixel_agg is not None when pix_all_same, checked with
        Bsplit = np.split(B, pix_inds[1:-1], axis=2)
        Bagg = [np.expand_dims(pixel_agg_func(b, axis=2), axis=2) for b in Bsplit]
        B = np.concatenate(Bagg, axis=2)

    # reset tiled objects
    for obj, m0 in zip(reset_obj, reset_obj_m0):
        obj._position = obj._position[:m0]
        obj._orientation = obj._orientation[:m0]

    # sumup over sources
    if sumup:
        B = np.sum(B, axis=0, keepdims=True)

    output = check_getBH_output_type(output)

    if output == "dataframe":
        # pylint: disable=import-outside-toplevel
        # pylint: disable=no-member
        import pandas as pd

        if sumup and len(sources) > 1:
            src_ids = [f"sumup ({len(sources)})"]
        else:
            src_ids = [s.style.label if s.style.label else f"{s}" for s in sources]
        sens_ids = [s.style.label if s.style.label else f"{s}" for s in sensors]
        num_of_pixels = np.prod(pix_shapes[0][:-1]) if pixel_agg is None else 1
        df = pd.DataFrame(
            data=product(src_ids, range(max_path_len), sens_ids, range(num_of_pixels)),
            columns=["source", "path", "sensor", "pixel"],
        )
        df[[field + k for k in "xyz"]] = B.reshape(-1, 3)
        return df

    # reduce all size-1 levels
    if squeeze:
        B = np.squeeze(B)
    elif pixel_agg is not None:
        # add missing dimension since `pixel_agg` reduces pixel
        # dimensions to zero. Only needed if `squeeze is False``
        B = np.expand_dims(B, axis=-2)

    return B


def getBH_dict_level2(
    source_type,
    observers,
    *,
    field: str,
    position=(0, 0, 0),
    orientation=R.identity(),
    squeeze=True,
    **kwargs: dict,
) -> np.ndarray:
    """Direct interface access to vectorized computation

    Parameters
    ----------
    kwargs: dict that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3), field at obs_pos in mT or kA/m

    Info
    ----
    - check inputs

    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # generate dict of secured inputs for auto-tiling ---------------
    #  entries in this dict will be tested for input length, and then
    #  be automatically tiled up and stored back into kwargs for calling
    #  getBH_level1().
    #  To allow different input dimensions, the tdim argument is also given
    #  which tells the program which dimension it should tile up.

    # pylint: disable=import-outside-toplevel

    try:
        source_classes = get_registered_sources()
        field_func = source_classes[source_type]._field_func
        field_func_kwargs_ndim = {"position": 2, "orientation": 2, "observers": 2}
        field_func_kwargs_ndim.update(
            source_classes[source_type]._field_func_kwargs_ndim
        )
    except KeyError as err:
        raise MagpylibBadUserInput(
            f"Input parameter `sources` must be one of {list(source_classes)}"
            " when using the direct interface."
        ) from err

    kwargs["observers"] = observers
    kwargs["position"] = position

    # change orientation to Rotation numpy array for tiling
    kwargs["orientation"] = orientation.as_quat()

    # evaluation vector lengths
    vec_lengths = {}
    ragged_seq = {}
    for key, val in kwargs.items():
        try:
            if (
                not isinstance(val, numbers.Number)
                and not isinstance(val[0], numbers.Number)
                and any(len(o) != len(val[0]) for o in val)
            ):
                ragged_seq[key] = True
                val = np.array([np.array(v, dtype=float) for v in val], dtype="object")
            else:
                ragged_seq[key] = False
                val = np.array(val, dtype=float)
        except TypeError as err:
            raise MagpylibBadUserInput(
                f"{key} input must be array-like.\n" f"Instead received {val}"
            ) from err
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim == expected_dim or ragged_seq[key]:
            if len(val) == 1:
                val = np.squeeze(val)
            else:
                vec_lengths[key] = len(val)

        kwargs[key] = val

    if len(set(vec_lengths.values())) > 1:
        raise MagpylibBadUserInput(
            "Input array lengths must be 1 or of a similar length.\n"
            f"Instead received lengths {vec_lengths}"
        )
    vec_len = max(vec_lengths.values(), default=1)
    # tile 1D inputs and replace original values in kwargs
    for key, val in kwargs.items():
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim < expected_dim and not ragged_seq[key]:
            kwargs[key] = np.tile(val, (vec_len, *[1] * (expected_dim - 1)))

    # change orientation back to Rotation object
    kwargs["orientation"] = R.from_quat(kwargs["orientation"])

    # compute and return B
    B = getBH_level1(field=field, field_func=field_func, **kwargs)

    if B is not None and squeeze:
        return np.squeeze(B)
    return B


def getB(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    **kwargs,
):
    """Compute B-field in units of mT for given sources and observers.

    Field implementations can be directly accessed (avoiding the object oriented
    Magpylib interface) by providing a string input `sources=source_type`, array_like
    positions as `observers` input, and all other necessary input parameters (see below)
    as kwargs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l source and/or collection objects.

        Direct interface: input must be one of (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`,
        `'Sphere'`, `'Dipole'`, `'Circle'` or `'Polyline'`).

    observers: array_like or (list of) `Sensor` objects
        Can be array_like positions of shape (n1, n2, ..., 3) where the field
        should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
        of such sensor objects (must all have similar pixel shapes). All positions
        are given in units of mm.

        Direct interface: Input must be array_like with shape (3,) or (n,3) corresponding
        positions to observer positions in units of mm.

    sumup: bool, default=`False`
        If `True`, the fields of all sources are summed up.

    squeeze: bool, default=`True`
        If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    pixel_agg: str, default=`None`
        Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
        which is applied to observer output values, e.g. mean of all sensor pixel outputs.
        With this option, observers input with different (pixel) shapes is allowed.

    output: str, default='ndarray'
        Output type, which must be one of `('ndarray', 'dataframe')`. By default a
        `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
        object is returned (the Pandas library must be installed).

    See Also
    --------
    *Direct-interface

    position: array_like, shape (3,) or (n,3), default=`(0,0,0)`
        Source position(s) in the global coordinates in units of mm.

    orientation: scipy `Rotation` object with length 1 or n, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation.

    magnetization: array_like, shape (3,) or (n,3)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`, `'Sphere'`)!
        Magnetization vector(s) (mu0*M, remanence field) in units of kA/m given in
        the local object coordinates (rotates with object).

    moment: array_like, shape (3) or (n,3), unit mT*mm^3
        Only source_type == `'Dipole'`!
        Magnetic dipole moment(s) in units of mT*mm^3 given in the local object coordinates
        (rotates with object). For homogeneous magnets the relation moment=magnetization*volume
        holds.

    current: array_like, shape (n,)
        Only source_type == `'Circle'` or `'Polyline'`!
        Electrical current in units of A.

    dimension: array_like, shape (x,) or (n,x)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`)!
        Magnet dimension input in units of mm and deg. Dimension format x of sources is similar
        as in object oriented interface.

    diameter: array_like, shape (n,)
        Only source_type == `'Sphere'` or `'Circle'`!
        Diameter of source in units of mm.

    segment_start: array_like, shape (n,3)
        Only source_type == `'Polyline'`!
        Start positions of line current segments in units of mm.

    segment_end: array_like, shape (n,3)
        Only source_type == `'Polyline'`!
        End positions of line current segments in units of mm.

    Returns
    -------
    B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
        B-field at each path position (m) for each sensor (k) and each sensor pixel
        position (n1, n2, ...) in units of mT. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than m will be
        considered as static beyond their end.

    Direct interface: ndarray, shape (n,3)
        B-field for every parameter set in units of mT.

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    Examples
    --------
    In this example we compute the B-field in units of mT of a spherical magnet and a current loop
    at the observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Circle(current=100, diameter=2)
    >>> src2 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1)
    >>> B = magpy.getB([src1, src2], (1,1,1))
    >>> print(B)
    [[6.23597388e+00 6.23597388e+00 2.66977810e+00]
     [8.01875374e-01 8.01875374e-01 1.48029737e-16]]

    We can also use sensor objects as observers input:

    >>> sens1 = magpy.Sensor(position=(1,1,1))
    >>> sens2 = sens1.copy(position=(1,1,-1))
    >>> B = magpy.getB([src1, src2], [sens1, sens2])
    >>> print(B)
    [[[ 6.23597388e+00  6.23597388e+00  2.66977810e+00]
      [-6.23597388e+00 -6.23597388e+00  2.66977810e+00]]
    <BLANKLINE>
     [[ 8.01875374e-01  8.01875374e-01  1.48029737e-16]
      [-8.01875374e-01 -8.01875374e-01  1.48029737e-16]]]

    Through the direct interface we can compute the same fields for the loop as:

    >>> obs = [(1,1,1), (1,1,-1)]
    >>> B = magpy.getB('Circle', obs, current=100, diameter=2)
    >>> print(B)
    [[ 6.23597388  6.23597388  2.6697781 ]
     [-6.23597388 -6.23597388  2.6697781 ]]

    But also for a set of four completely different instances:

    >>> B = magpy.getB(
    ...     'Circle',
    ...     observers=((1,1,1), (1,1,-1), (1,2,3), (2,2,2)),
    ...     current=(11, 22, 33, 44),
    ...     diameter=(1, 2, 3, 4),
    ...     position=((0,0,0), (0,0,1), (0,0,2), (0,0,3)),
    ... )
    >>> print(B)
    [[ 0.17111325  0.17111325  0.01705189]
     [-0.38852048 -0.38852048  0.49400758]
     [ 1.14713551  2.29427102 -0.22065346]
     [-2.48213467 -2.48213467 -0.79683487]]
    """
    return getBH_level2(
        sources,
        observers,
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        field="B",
        **kwargs,
    )


def getH(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    **kwargs,
):
    """Compute H-field in kA/m for given sources and observers.

    Field implementations can be directly accessed (avoiding the object oriented
    Magpylib interface) by providing a string input `sources=source_type`, array_like
    positions as `observers` input, and all other necessary input parameters (see below)
    as kwargs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l source and/or collection objects.

        Direct interface: input must be one of (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`,
        `'Sphere'`, `'Dipole'`, `'Circle'` or `'Polyline'`).

    observers: array_like or (list of) `Sensor` objects
        Can be array_like positions of shape (n1, n2, ..., 3) where the field
        should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
        of such sensor objects (must all have similar pixel shapes). All positions
        are given in units of mm.

        Direct interface: Input must be array_like with shape (3,) or (n,3) corresponding
        positions to observer positions in units of mm.

    sumup: bool, default=`False`
        If `True`, the fields of all sources are summed up.

    squeeze: bool, default=`True`
        If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    pixel_agg: str, default=`None`
        Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
        which is applied to observer output values, e.g. mean of all sensor pixel outputs.
        With this option, observers input with different (pixel) shapes is allowed.

    output: str, default='ndarray'
        Output type, which must be one of `('ndarray', 'dataframe')`. By default a
        `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
        object is returned (the Pandas library must be installed).

    See Also
    --------
    *Direct-interface

    position: array_like, shape (3,) or (n,3), default=`(0,0,0)`
        Source position(s) in the global coordinates in units of mm.

    orientation: scipy `Rotation` object with length 1 or n, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation.

    magnetization: array_like, shape (3,) or (n,3)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`, `'Sphere'`)!
        Magnetization vector(s) (mu0*M, remanence field) in units of kA/m given in
        the local object coordinates (rotates with object).

    moment: array_like, shape (3) or (n,3), unit mT*mm^3
        Only source_type == `'Dipole'`!
        Magnetic dipole moment(s) in units of mT*mm^3 given in the local object coordinates
        (rotates with object). For homogeneous magnets the relation moment=magnetization*volume
        holds.

    current: array_like, shape (n,)
        Only source_type == `'Circle'` or `'Polyline'`!
        Electrical current in units of A.

    dimension: array_like, shape (x,) or (n,x)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`)!
        Magnet dimension input in units of mm and deg. Dimension format x of sources is similar
        as in object oriented interface.

    diameter: array_like, shape (n,)
        Only source_type == `'Sphere'` or `'Circle'`!
        Diameter of source in units of mm.

    segment_start: array_like, shape (n,3)
        Only source_type == `'Polyline'`!
        Start positions of line current segments in units of mm.

    segment_end: array_like, shape (n,3)
        Only source_type == `'Polyline'`!
        End positions of line current segments in units of mm.

    Returns
    -------
    H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
        H-field at each path position (m) for each sensor (k) and each sensor pixel
        position (n1, n2, ...) in units of kA/m. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than m will be
        considered as static beyond their end.

    Direct interface: ndarray, shape (n,3)
        H-field for every parameter set in units of kA/m.

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    Examples
    --------
    In this example we compute the H-field kA/m of a spherical magnet and a current loop
    at the observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Circle(current=100, diameter=2)
    >>> src2 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1)
    >>> H = magpy.getH([src1, src2], (1,1,1))
    >>> print(H)
    [[4.96243034e+00 4.96243034e+00 2.12454191e+00]
     [6.38112147e-01 6.38112147e-01 1.17798322e-16]]

    We can also use sensor objects as observers input:

    >>> sens1 = magpy.Sensor(position=(1,1,1))
    >>> sens2 = sens1.copy(position=(1,1,-1))
    >>> H = magpy.getH([src1, src2], [sens1, sens2])
    >>> print(H)
    [[[ 4.96243034e+00  4.96243034e+00  2.12454191e+00]
      [-4.96243034e+00 -4.96243034e+00  2.12454191e+00]]
    <BLANKLINE>
     [[ 6.38112147e-01  6.38112147e-01  1.17798322e-16]
      [-6.38112147e-01 -6.38112147e-01  1.17798322e-16]]]

    Through the direct interface we can compute the same fields for the loop as:

    >>> obs = [(1,1,1), (1,1,-1)]
    >>> H = magpy.getH('Circle', obs, current=100, diameter=2)
    >>> print(H)
    [[ 4.96243034  4.96243034  2.12454191]
     [-4.96243034 -4.96243034  2.12454191]]

    But also for a set of four completely different instances:

    >>> H = magpy.getH(
    ...     'Circle',
    ...     observers=((1,1,1), (1,1,-1), (1,2,3), (2,2,2)),
    ...     current=(11, 22, 33, 44),
    ...     diameter=(1, 2, 3, 4),
    ...     position=((0,0,0), (0,0,1), (0,0,2), (0,0,3)),
    ... )
    >>> print(H)
    [[ 0.1361676   0.1361676   0.01356947]
     [-0.30917477 -0.30917477  0.39311875]
     [ 0.91286143  1.82572286 -0.17559045]
     [-1.97522001 -1.97522001 -0.63410104]]
    """
    return getBH_level2(
        sources,
        observers,
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        field="H",
        **kwargs,
    )
