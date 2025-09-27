"""Magnetuic field computation structure:

level_core:(field_BH_XXX.py files)   <--- CORE INTERFACE
    - pure vectorized field computations from literature
    - all computations in source CS

level0a:(BHJM_XX)
    - distinguish between B, H, J and M

level0b:(BHJM_internal_XX)
    - connect BHJM-level to level1

level1(_getBH_level1):
    - apply transformation to global CS
    - select correct level0 src_type computation
    - input dict, no input checks !

level2(_getBH_level2):
    - input dict checks (unknowns)
    - secure user inputs
    - group similar sources for combined computation
    - generate vector input format for _getBH_level1
    - adjust Bfield output format to (pos_obs, path, sources) input format

level 2b (func.getB_cuboid, ...)  <---FUNCTIONAL INTERFACE
    - prepares vectorized input
    - calls directly into level1

level3(magpylib.getB/H):   <--- USER INTERFACE
    - docstrings
    - separated B and H
    - transform input into dict for level2

level4(src.getB/H, sensor.getB/H, collection.getB/H):   <--- USER INTERFACE
    - docstrings
    - calling level3 getB, getH directly from sources

"""

# pylint: disable=cyclic-import
# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments

import warnings
from collections.abc import Callable
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BHfunc import _getBH_dict_level2
from magpylib._src.input_checks import (
    check_dimensions,
    check_excitations,
    check_format_input_observers,
    check_format_pixel_agg,
    check_getBH_output_type,
)
from magpylib._src.utility import (
    check_static_sensor_orient,
    format_obj_input,
    format_src_inputs,
    has_parameter,
)


def _tile_group_property(group: list, n_pp: int, prop_name: str):
    """tile up group property"""
    out = [getattr(src, prop_name) for src in group]
    if not np.isscalar(out[0]) and any(o.shape != out[0].shape for o in out):
        out = np.asarray(out, dtype="object")
    else:
        out = np.array(out)
    return np.repeat(out, n_pp, axis=0)


def _get_src_dict(group: list, n_pix: int, n_pp: int, poso: np.ndarray) -> dict:
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
            kwargs[prop] = _tile_group_property(group, n_pp, prop)

    return kwargs


def _getBH_level1(
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
    kwargs: dict of shape (i, x) input vectors that describes the computation.

    Returns
    -------
    field: ndarray, shape (i, 3)

    """

    # transform obs_pos into source CS
    pos_rel_rot = orientation.apply(observers - position, inverse=True)

    # filter arguments
    if not has_parameter(field_func, "in_out"):  # in_out passed only to magnets
        kwargs.pop("in_out", None)

    # compute field
    BH = field_func(field=field, observers=pos_rel_rot, **kwargs)

    # transform field back into global CS
    if BH is not None:  # catch non-implemented field_func a level above
        BH = orientation.apply(BH)

    return BH


def _getBH_level2(
    sources, observers, *, field, sumup, squeeze, pixel_agg, output, in_out, **kwargs
) -> np.ndarray:
    """Compute field for given sources and observers.
    Info:
    -----
    - generates a 1D list of sources (collections flattened) and a 1D list of sensors from input
    - tile all paths of static (path_length=1) objects
    - combine all sensor pixel positions for joint evaluation
    - group similar source types for joint evaluation
    - compute field and store in allocated array
    - rearrange the array in the shape squeeze((L, M, K, N1, N2, ..., 3))
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=import-outside-toplevel

    from magpylib._src.obj_classes.class_Collection import Collection  # noqa: I001, PLC0415
    from magpylib._src.obj_classes.class_magnet_TriangularMesh import TriangularMesh  # noqa: PLC0415

    # DEPRECATED CALL TO FUNCTIONAL INTERFACE
    if isinstance(sources, str):
        msg = (
            "Calling the functional interface (string input for sources) from magpy.getB() "
            "is deprecated and will be removed in future versions. The functional "
            "interface was moved to the magpy.func subpackage."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return _getBH_dict_level2(
            source_type=sources,
            observers=observers,
            field=field,
            squeeze=squeeze,
            in_out=in_out,
            **kwargs,
        )

    # CHECK AND FORMAT INPUT ---------------------------------------------------

    # format sources input:
    #   input: allow only one bare src object or a 1D list/tuple of src and col
    #   out: sources = ordered list of sources
    #   out: src_list = ordered list of sources with flattened collections
    sources, src_list = format_src_inputs(sources)

    # test if all source dimensions and excitations are initialized
    check_dimensions(src_list)
    check_excitations(src_list)

    # make sure that given in_out there is a Tetrahedron class or a TriangularMesh
    #   class in sources. Else throw a warning
    if in_out != "auto":
        from magpylib._src.obj_classes.class_magnet_Tetrahedron import Tetrahedron  # noqa: I001, PLC0415

        if not any(isinstance(src, Tetrahedron | TriangularMesh) for src in src_list):
            warnings.warn(
                "Parameter in_out was explicitly set but is ignored in the computation. "
                "It applies only to classes Tetrahedron and TriangularMesh.",
                UserWarning,
                stacklevel=2,
            )

    # make sure that TriangularMesh sources have a closed mesh when getB is called - warn if not
    if field == "B":
        for src in src_list:
            if isinstance(src, TriangularMesh):
                # unchecked mesh status - may be open
                if src.status_open is None:
                    warnings.warn(
                        f"Unchecked open mesh status in {src!r} detected before B-field computation. "
                        "An open mesh may return bad results.",
                        stacklevel=2,
                    )
                elif src.status_open:  # mesh is open
                    warnings.warn(
                        f"Open mesh detected in {src!r} before B-field computation. "
                        "An open mesh may return bad results.",
                        stacklevel=2,
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
    pix_inds = np.cumsum([0, *pix_nums])  # cumulative indices of pixel for each sensor
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
    reset_obj = [obj for obj, mask in zip(obj_list, mask_reset, strict=False) if mask]
    reset_obj_m0 = [
        pl for pl, mask in zip(path_lengths, mask_reset, strict=False) if mask
    ]

    if max_path_len > 1:
        for obj, m0 in zip(reset_obj, reset_obj_m0, strict=False):
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
            (
                np.array([[0, 0, 0]])
                if sens.pixel is None
                else r.apply(sens.pixel.reshape(-1, 3))
            )
            + p
            for r, p in zip(sens._orientation, sens._position, strict=False)
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
            msg = (
                f"Cannot compute {field}-field because input "
                f"field_func of {src} has undefined {field}-field computation."
            )
            raise MagpylibMissingInput(msg)
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
        src_dict = _get_src_dict(gr, n_pix, n_pp, poso)  # compute array dict for level1
        # compute field
        B_group = _getBH_level1(
            field_func=field_func, field=field, in_out=in_out, **src_dict
        )
        if B_group is None:
            msg = (
                f"Cannot compute {field}-field because input "
                f"field_func {field_func} has undefined {field}-field computation."
            )
            raise MagpylibMissingInput(msg)
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
            # change shape to (P, 3) for rot package
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
    for obj, m0 in zip(reset_obj, reset_obj_m0, strict=False):
        obj._position = obj._position[:m0]
        obj._orientation = obj._orientation[:m0]

    # sumup over sources
    if sumup:
        B = np.sum(B, axis=0, keepdims=True)

    output = check_getBH_output_type(output)

    if output == "dataframe":
        # pylint: disable=import-outside-toplevel
        # pylint: disable=no-member
        import pandas as pd  # noqa: PLC0415

        if sumup and len(sources) > 1:
            src_ids = [f"sumup ({len(sources)})"]
        else:
            src_ids = [s.style.label if s.style.label else f"{s}" for s in sources]
        sens_ids = [s.style.label if s.style.label else f"{s}" for s in sensors]
        num_of_pixels = np.prod(pix_shapes[0][:-1]) if pixel_agg is None else 1
        df_field = pd.DataFrame(
            data=product(src_ids, range(max_path_len), sens_ids, range(num_of_pixels)),
            columns=["source", "path", "sensor", "pixel"],
        )
        df_field[[field + k for k in "xyz"]] = B.reshape(-1, 3)
        return df_field

    # reduce all size-1 levels
    if squeeze:
        B = np.squeeze(B)
    elif pixel_agg is not None:
        # add missing dimension since `pixel_agg` reduces pixel
        # dimensions to zero. Only needed if `squeeze is False``
        B = np.expand_dims(B, axis=-2)

    return B


def getB(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    in_out="auto",
    **kwargs,
):
    """Return B-field (T) of s sources at o observers.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    sources : Source | list
        Sources that generate the magnetic field. Can be a single source
        or a 1D list of s sources.
    observers : Sensor | list[Sensor] | array-like, shape (o1, o2, ..., 3)
        Input specifying where the field is evaluated. Multiple objects in a list
        must have identical pixel shape unless ``pixel_agg`` is used.
        All positions given in units (m)
    sumup : bool, default False
        If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
    pixel_agg : str | None, default None
        Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
        pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
    output : {'ndarray', 'dataframe'}, default 'ndarray'
        Output container type. ``'dataframe'`` returns a pandas DataFrame.
    in_out : {'auto', 'inside', 'outside'}, default 'auto'
        Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
        observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
        all outside (faster).

    Returns
    -------
    ndarray | DataFrame
        B-field (T) with squeezed shape (s, p, o, o1, o2, ..., 3) where s is the number
        of sources, p is the number of paths, o the number of observers with (pixel)shape
        o1, o2, ... .

    Examples
    --------
    B-field of a current loop and a spherical magnet at one observer:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> loop = magpy.current.Circle(current=100, diameter=0.002)
    >>> sph = magpy.magnet.Sphere(polarization=(0.0, 0.0, 0.1), diameter=0.001)
    >>> B = magpy.getB([loop, sph], (0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 6.054e-06  6.054e-06  2.357e-08]
    [ 8.019e-07  8.019e-07 -9.056e-23]]

    With two sensors:

    >>> s1 = magpy.Sensor(position=(0.01, 0.01, 0.01))
    >>> s2 = s1.copy(position=(0.01, 0.01, -0.01))
    >>> B = magpy.getB([loop, sph], [s1, s2])
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[[ 6.054e-06  6.054e-06  2.357e-08]
    [-6.054e-06 -6.054e-06  2.357e-08]]
    <BLANKLINE>
    [[ 8.019e-07  8.019e-07 -9.056e-23]
    [-8.019e-07 -8.019e-07 -9.056e-23]]]
    """
    return _getBH_level2(
        sources,
        observers,
        field="B",
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getH(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    in_out="auto",
    **kwargs,
):
    """Return H-field (A/m) of s sources at o observers.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    sources : Source | list
        Sources that generate the magnetic field. Can be a single source
        or a 1D list of s sources.
    observers : Sensor | list[Sensor] | array-like, shape (o1, o2, ..., 3)
        Input specifying where the field is evaluated. Multiple objects in a list
        must have identical pixel shape unless ``pixel_agg`` is used.
        All positions given in units (m)
    sumup : bool, default False
        If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
    pixel_agg : str | None, default None
        Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
        pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
    output : {'ndarray', 'dataframe'}, default 'ndarray'
        Output container type. ``'dataframe'`` returns a pandas DataFrame.
    in_out : {'auto', 'inside', 'outside'}, default 'auto'
        Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
        observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
        all outside (faster).

    Returns
    -------
    ndarray | DataFrame
        H-field (A/m) with squeezed shape (s, p, o, o1, o2, ..., 3) where s is the number
        of sources, p is the number of paths, o the number of observers with (pixel)shape
        o1, o2, ... .

    Examples
    --------
    H-field of a current loop and a spherical magnet at one observer:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> loop = magpy.current.Circle(current=100.0, diameter=0.002)
    >>> sph = magpy.magnet.Sphere(polarization=(0.0, 0.0, 0.1), diameter=0.001)
    >>> H = magpy.getH([loop, sph], (0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...    print(H)
    [[ 4.818e+00  4.818e+00  1.875e-02]
    [ 6.381e-01  6.381e-01 -7.207e-17]]

    With two sensors:

    >>> sens1 = magpy.Sensor(position=(0.01, 0.01, 0.01))
    >>> sens2 = sens1.copy(position=(0.01, 0.01, -0.01))
    >>> H = magpy.getH([src1, src2], [sens1, sens2])
    >>> with np.printoptions(precision=3):
    ...    print(H)
    [[[ 4.818e+00  4.818e+00  1.875e-02]
    [-4.818e+00 -4.818e+00  1.875e-02]]
    <BLANKLINE>
    [[ 6.381e-01  6.381e-01 -7.207e-17]
    [-6.381e-01 -6.381e-01 -7.207e-17]]]
    """
    return _getBH_level2(
        sources,
        observers,
        field="H",
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getM(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    in_out="auto",
    **kwargs,
):
    """Return magnetization (A/m) of s sources at o observers.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    sources : Source | list
        Sources that generate the magnetic field. Can be a single source
        or a 1D list of s sources.
    observers : Sensor | list[Sensor] | array-like, shape (o1, o2, ..., 3)
        Input specifying where the field is evaluated. Multiple objects in a list
        must have identical pixel shape unless ``pixel_agg`` is used.
        All positions given in units (m)
    sumup : bool, default False
        If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
    pixel_agg : str | None, default None
        Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
        pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
    output : {'ndarray', 'dataframe'}, default 'ndarray'
        Output container type. ``'dataframe'`` returns a pandas DataFrame.
    in_out : {'auto', 'inside', 'outside'}, default 'auto'
        Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
        observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
        all outside (faster).

    Returns
    -------
    ndarray | DataFrame
        Magnetization (A/m) with squeezed shape (s, p, o, o1, o2, ..., 3) where s is the number
        of sources, p is the number of paths, o the number of observers with (pixel)shape
        o1, o2, ... .

    Examples
    --------
    Magnetization at one point (inside the magnet):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> cube = magpy.magnet.Cuboid(
    ...     dimension=(10, 1, 1),
    ...     polarization=(1, 0, 0)
    ... ).rotate_from_angax(45, 'z')
    >>> M = cube.getM((3, 3, 0))
    >>> with np.printoptions(precision=0):
    ...    print(M)
    [562698. 562698.      0.]
    """
    return _getBH_level2(
        sources,
        observers,
        field="M",
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getJ(
    sources=None,
    observers=None,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    in_out="auto",
    **kwargs,
):
    """Return magnetic polarization (T) of s sources at o observers.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    sources : Source | list
        Sources that generate the magnetic field. Can be a single source
        or a 1D list of s sources.
    observers : Sensor | list[Sensor] | array-like, shape (o1, o2, ..., 3)
        Input specifying where the field is evaluated. Multiple objects in a list
        must have identical pixel shape unless ``pixel_agg`` is used.
        All positions given in units (m)
    sumup : bool, default False
        If ``True``, sum the fields from all sources. If ``False``, keep the source axis.
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (e.g. a single source or a single sensor).
    pixel_agg : str | None, default None
        Name of a NumPy aggregation function (e.g. ``'mean'``, ``'min'``) applied over the
        pixel axis of each sensor. Allows mixing sensors with different pixel shapes.
    output : {'ndarray', 'dataframe'}, default 'ndarray'
        Output container type. ``'dataframe'`` returns a pandas DataFrame.
    in_out : {'auto', 'inside', 'outside'}, default 'auto'
        Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
        observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
        all outside (faster).

    Returns
    -------
    ndarray | DataFrame
        Polarization (T) with squeezed shape (s, p, o, o1, o2, ..., 3) where s is the number
        of sources, p is the number of paths, o the number of observers with (pixel)shape
        o1, o2, ... .

    Examples
    --------
    Polarization at one point (inside the magnet):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> cube = magpy.magnet.Cuboid(
    ...     dimension=(10, 1, 1),
    ...     polarization=(1, 0, 0)
    ... ).rotate_from_angax(45, 'z')
    >>> J = cube.getJ((3, 3, 0))
    >>> with np.printoptions(precision=3):
    ...    print(J)
    [0.707 0.707 0.   ]
    """
    return _getBH_level2(
        sources,
        observers,
        field="J",
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )
