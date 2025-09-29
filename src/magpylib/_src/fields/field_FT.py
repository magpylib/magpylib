"""Force implementation."""

# pylint: disable=import-outside-toplevel

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.fields.field_BH import getB
from magpylib._src.input_checks import check_dimensions, check_excitations
from magpylib._src.utility import format_src_inputs

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _check_format_input_targets(targets):
    """
    Check and format targets input
    - flatten Collections
    - check if allowed instance
    - check if meshing parameter is set

    Returns
    -------
    flat list of targets , list of collection indices for summation
    """
    if not isinstance(targets, list):
        targets = [targets]

    # flatten out collections, keep indices for later summation in getFT
    flat_targets = []
    coll_idx = []
    idx = 0
    for t in targets:
        from magpylib._src.obj_classes.class_Collection import (  # noqa: PLC0415
            Collection,
        )

        if isinstance(t, Collection):
            len_sources = len(t.sources_all)
            if len_sources == 0:
                msg = f"Given target Collection {t} has no target sources."
                raise ValueError(msg)
            flat_targets.extend(t.sources_all)
            coll_idx.append(idx)
            idx += len_sources
        else:
            flat_targets.append(t)
            coll_idx.append(idx)
            idx += 1

    # check if all flat_targets are valid

    # check if dimensions and excitations are initialized
    check_dimensions(flat_targets)
    check_excitations(flat_targets)

    for t in flat_targets:
        # exclude Dipole from check
        from magpylib._src.obj_classes.class_magnet_Sphere import (  # noqa: PLC0415
            Sphere,
        )
        from magpylib._src.obj_classes.class_misc_Dipole import Dipole  # noqa: PLC0415

        if not isinstance(t, Dipole | Sphere):
            if not hasattr(t, "meshing"):
                msg = (
                    "Input targets must be allowed Magpylib target objects; "
                    f"instead received type {type(t)}."
                )
                raise ValueError(msg)

            # check if meshing parameter is explicitly set
            if t.meshing is None:
                msg = (
                    f"Missing meshing input for target {t}. "
                    "All targets must have the meshing parameter explicitly set."
                )
                raise ValueError(msg)

    return flat_targets, coll_idx


def _create_eps_vector(eps):
    """
    Create a vector of finite difference steps based on the input eps.

    Parameters
    ----------
    eps : float
        The finite difference step size.

    Returns
    -------
    np.ndarray
        A vector of shape (7, 3) for finite difference calculations.
    """
    return np.array(
        [
            (0, 0, 0),
            (eps, 0, 0),
            (-eps, 0, 0),
            (0, eps, 0),
            (0, -eps, 0),
            (0, 0, eps),
            (0, 0, -eps),
        ]
    )


def _check_format_input_pivot(pivot, targets, n_path):
    """
    Check and format pivot input

    Returns
    -------
    array of pivots with shape (t, p, 3) or None
    """
    msg = (
        "Input pivot must be 'centroid', None, or array-like with shape "
        "(3,), (t, 3), or (t, p, 3) when t targets are present and path "
        f"length p are present; instead received {pivot!r}."
    )

    if pivot is None:
        return None

    if isinstance(pivot, str) and pivot == "centroid":
        return np.array(
            [
                np.pad(t._centroid, ((0, n_path - len(t._centroid)), (0, 0)), "edge")
                for t in targets
            ]
        )

    if isinstance(pivot, list | tuple | np.ndarray):
        try:
            pivot = np.array(pivot, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(msg) from e

        n_tgt = len(targets)

        if pivot.shape == (3,):
            return np.tile(pivot, (n_tgt, n_path, 1))
        if pivot.shape == (n_tgt, 3):
            return np.repeat(pivot[:, np.newaxis, :], n_path, axis=1)
        if pivot.shape == (n_tgt, n_path, 3):
            return pivot

    raise ValueError(msg)


def _check_eps(eps):
    """
    check FD step
    """
    msg = f"Input eps must be a positive float; instead received {eps!r}."
    if not isinstance(eps, float):
        raise ValueError(msg)
    if eps <= 0:
        raise ValueError(msg)


def getFT(
    sources,
    targets,
    pivot="centroid",
    eps=1e-5,
    squeeze=True,
    meshreport=False,
    return_mesh=False,
):
    """Compute magnetic force and torque on t targets from s sources.

    The computation uses meshing and finite differences. SI units are assumed
    for all inputs and outputs.

    Parameters
    ----------
    sources : Source | list[Source]
        Sources that generate the magnetic field. Can be a single source or a
        1D list of s source objects.
    targets : Target | list[Target]
        Objects on which the magnetic field acts, generating force and torque.
        Can be a 1D list of t target objects. All targets (except Dipoles
        and Spheres) must have a valid ``meshing`` parameter set.
    pivot : 'centroid' | None | array-like, shape (3,) or (t, 3) or (t, p, 3), default 'centroid'
        Pivot point through which the force contributes to the torque. If
        ``'centroid'``, each target's centroid is used. If ``None``, no pivot
        is applied (may yield nonphysical results). If an array of shape
        (3,), the same pivot is used for all targets. Shapes (t, 3)
        and (t, p, 3) provide per-target or per-target-per-path pivots.
    eps : float, default 1e-5
        Finite-difference step size for gradient-field computation for magnet
        targets. A good value is 1e-5 * characteristic_system_size (e.g.,
        magnet size or source-target distance).
    squeeze : bool, default True
        If ``True``, dimensions of size 1 in the output are removed.
    meshreport : bool, default False
        If ``True``, prints a brief report of the mesh used for each target.
    return_mesh : bool, default False
        If ``True``, returns the meshes as a list of dictionaries instead of
        force and torque.

    Returns
    -------
    tuple[ndarray, ndarray]
        Force and torque with shapes (s, p, t, 3), where s is the
        number of sources, p the path length, and t the number of
        targets. If ``squeeze`` is ``True``, dimensions of size 1 are removed.
        If ``return_mesh`` is ``True``, returns the meshes list instead.

    Notes
    -----
    The force and torque are computed via
    F = (gradB) · MOM and T = B x MOM + r x F. The gradient field is
    obtained using finite differences on the meshed targets.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> cube = magpy.magnet.Cuboid(
    ...     dimension=(1.0, 1.0, 1.0),
    ...     polarization=(0.1, 0.2, 0.3),
    ... )
    >>> circ = magpy.current.Circle(
    ...     diameter=2.0,
    ...     current=1e3,
    ...     position=(0.0, 0.0, 1.0),
    ...     meshing=50,
    ... )
    >>> F, T = magpy.getFT(cube, circ)
    >>> print(f'force: {np.round(F, decimals=2)} N')
    force: [ 13.65  27.31 -81.93] N
    >>> print(f'torque: {np.round(T, decimals=2)} N*m')
    torque: [-8.55  4.27 -0.  ] N*m
    """
    # COMPUTATION SCHEME
    #   STEP1: Collect all inputs
    #   STEP2: Broadcast into large computation arrays DB, B, MOM, ...
    #   STEP3: Apply the equations
    #   STEP4: Sum up and reshape

    # INPUT CHECKS ############################################################
    #  - check and format all inputs

    # Input check targets
    #  - check if all targets are allowed and parameters are set (meshing, ...)
    #  - return 1D list of targets (flatten collections)
    targets, coll_idx = _check_format_input_targets(targets)
    n_tgt = len(targets)

    # Input checks sources: from getB
    sources, _ = format_src_inputs(sources)
    n_src = len(sources)

    # Determine path length
    tgt_path_lengths = [len(tgt._position) for tgt in targets]
    src_path_lengths = [len(src._position) for src in sources]
    n_path = max(tgt_path_lengths + src_path_lengths)

    # Input check pivot
    #  - return shape (n_tgt, n_path, 3)
    pivot = _check_format_input_pivot(pivot, targets, n_path)

    # Input check eps
    _check_eps(eps)

    # RUN MESHING FUNCTIONS ###################################################
    #  - collect all meshing infos
    #  - prepare masks and idx for later

    meshes = [tgt._generate_mesh() for tgt in targets]

    # Mesh sizes and Masks
    mask_magnet = np.array(["moments" in m for m in meshes])
    mask_current = ~mask_magnet

    mesh_sizes_all = np.array([len(m["pts"]) for m in meshes])
    mesh_sizes_all7 = mesh_sizes_all * np.where(mask_magnet, 7, 1)  # with FD steps

    mask_mesh_magnet = np.repeat(mask_magnet, mesh_sizes_all)
    mask_mesh_current = ~mask_mesh_magnet

    # Total number of instances
    n_magnet = sum(mask_magnet)
    n_current = sum(mask_current)
    n_mesh_all = sum(mesh_sizes_all)
    n_mesh_all7 = sum(mesh_sizes_all7)

    # Index lists
    idx_mag = np.where(mask_magnet)[0]
    idx_cur = np.where(mask_current)[0]
    idx_ends_all = np.cumsum(mesh_sizes_all)  # All meshs are tiled to one big
    idx_starts_all = np.r_[0, idx_ends_all[:-1]]  #  array. These are start and
    idx_ends_all7 = np.cumsum(mesh_sizes_all7)  #  end indices of respective
    idx_starts_all7 = np.r_[0, idx_ends_all7[:-1]]  #  meshes.

    # Meshreport - maybe add optional breakpoint here so that one can look at the
    #  mesh parameters before going into eval.
    if meshreport:
        logging.info("Mesh report:")
        for t, m in zip(targets, mesh_sizes_all, strict=False):
            logging.info("  Target %s: %d points", t, m)
        logging.info("")

    # OBSERVER ARRAY ##########################################################
    # determine observer points for B-field evaluation
    #  - path padding
    #  - apply path transform to meshes
    #  - add FD steps
    OBS7 = np.empty((n_path, n_mesh_all7, 3))

    # For later rotations of moments etc.
    tgt_ori = np.empty((n_tgt, n_path, 4))

    # Create transformed meshes and broadcast into OBS7
    eps_vec = _create_eps_vector(eps)
    for i, tgt in enumerate(targets):
        # Path padding
        n_pad = n_path - len(tgt._position)
        if n_pad == 0:  # np.pad creates a relevant overhead
            pos = tgt._position
            ori = tgt._orientation.as_quat()
        else:
            pos = np.pad(tgt._position, ((0, n_pad), (0, 0)), "edge")
            ori = np.pad(tgt._orientation.as_quat(), ((0, n_pad), (0, 0)), "edge")

        # Store orientations
        tgt_ori[i] = ori

        # Indexing for broadcasting
        start7 = idx_starts_all7[i]
        end7 = idx_ends_all7[i]

        n_mesh = mesh_sizes_all[i]
        n_mesh7 = mesh_sizes_all7[i]

        # Apply path to meshes
        pos = np.repeat(pos, n_mesh, axis=0)
        ori = R.from_quat(np.repeat(ori, n_mesh, axis=0))
        mesh = np.tile(meshes[i]["pts"], (n_path, 1))
        mesh = ori.apply(mesh) + pos
        mesh = mesh.reshape(n_path, n_mesh, 3)

        # Extend meshes by FD steps
        if mask_magnet[i]:
            mesh = mesh[:, :, np.newaxis, :] + eps_vec[np.newaxis, np.newaxis, :, :]

        # Broadcast into OBS7
        OBS7[:, start7:end7] = mesh.reshape((n_path, n_mesh7, 3))

    # Return mesh for analysis
    if return_mesh:
        return meshes

    # B-FIELD COMPUTATION ############################################################
    # B-field computation limited by getB no ragged inputs, and that sources must be
    #    path-wise associated with observers.
    # Chosen approach: loop over path iff there is a source_path
    #  - most cases covered with single getB
    #  - general case annoying because source path must be extracted, cycled through, and reset
    # Alternatives:
    #  - create sensor objects from meshes is limited by ragged inputs
    #  - compute relative paths between all sources and targets requires source by source eval

    # Source path length
    n_src_path = max(len(src._position) for src in sources)

    # No source path (95% of cases)
    if n_src_path == 1:
        B_all = getB(sources, OBS7, squeeze=False)[:, 0, 0, :]

    # Annoying general case - cycle through path
    else:
        # Store original paths
        src_pos0 = [src._position for src in sources]
        src_ori0 = [src._orientation for src in sources]

        # Extract an pad source paths
        src_pos = np.empty((n_src, n_path, 3))
        src_ori = np.empty((n_src, n_path, 4))

        for i, src in enumerate(sources):
            n_pad = n_path - len(src._position)
            if n_pad == 0:  # np.pad creates a relevant overhead
                src_pos[i] = src._position
                src_ori[i] = src._orientation.as_quat()
            else:
                src_pos[i] = np.pad(src._position, ((0, n_pad), (0, 0)), "edge")
                src_ori[i] = np.pad(
                    src._orientation.as_quat(), ((0, n_pad), (0, 0)), "edge"
                )

        # Allocate
        B_all = np.empty((n_src, n_path, n_mesh_all7, 3))

        # Compute for each path and Broadcast
        for j in range(n_path):
            for i, src in enumerate(sources):
                src.position = src_pos[i, j]
                src.orientation = R.from_quat(src_ori[i, j])
            B_all[:, j] = getB(sources, OBS7[j], squeeze=True)

        # Restore original paths
        for i, src in enumerate(sources):
            src._position = src_pos0[i]
            src._orientation = src_ori0[i]

    F_all = np.zeros((n_src, n_path, n_mesh_all, 3))
    T_all = np.zeros((n_src, n_path, n_mesh_all, 3))

    # MAGNETS ########################################################################
    if n_magnet > 0:
        # Computation array allocation
        mesh_sizes = mesh_sizes_all[mask_magnet]
        n_mesh = sum(mesh_sizes)

        B = np.empty((n_src, n_path, n_mesh, 3))  # B-field at cell location
        DB = np.empty(
            (n_src, n_path, n_mesh, 3, 3)
        )  # B-field gradient at cell location
        MOM = np.empty((n_src, n_path, n_mesh, 3))  # magnetic moment of cells

        # Compute and Broadcast
        idx_ends = np.cumsum(mesh_sizes)
        idx_starts = np.r_[0, idx_ends[:-1]]

        for start, end, ii in zip(idx_starts, idx_ends, idx_mag, strict=False):
            start_all = idx_starts_all7[ii]
            end_all = idx_ends_all7[ii]

            # B - broadcast (better memory and speed than concatenate)
            B[:, :, start:end] = B_all[:, :, start_all:end_all:7]

            # DB - compute and broadcast
            for i in range(3):  # ∂B/∂x, ∂B/∂x, ∂B/∂z
                DB[:, :, start:end, i] = B_all[
                    :, :, start_all + 2 * i + 1 : end_all : 7
                ]
                DB[:, :, start:end, i] -= B_all[
                    :, :, start_all + 2 * i + 2 : end_all : 7
                ]
            DB[:, :, start:end] /= 2 * eps  # div all by FD length

            # MOM - apply rot and broadcast
            n = mesh_sizes_all[ii]
            mom = np.tile(meshes[ii]["moments"], (n_path, 1))
            rot = np.repeat(tgt_ori[ii], n, axis=0)
            mom_rot = R.from_quat(rot).apply(mom).reshape(n_path, n, 3)
            MOM[:, :, start:end] = mom_rot[np.newaxis, :]

        # Force and Torque computation
        #  - !!Performance!!: cross (torque) few time faster than einsum (force) few
        #    times faster than array API force. Also, the torque computation relies on
        #    the force computation. Therefore it makes no sense to separate the force
        #    and torque computation.

        F = np.einsum("abijk, abik->abij", DB, MOM)  # NumPy only
        # F = np.sum(DB * MOM[:, :, :, np.newaxis, :], axis=4) # array API
        T = np.cross(MOM, B)

        # broadcast into B (OBS7 needed below) to save memory? !!!!!!!!!!!!!!!!!!!!!!!!
        F_all[:, :, mask_mesh_magnet] = F
        T_all[:, :, mask_mesh_magnet] = T

    # CURRENT COMPUTATION #########################################################
    if n_current > 0:
        # Computation array allocation
        mesh_sizes = mesh_sizes_all[mask_current]
        n_mesh = sum(mesh_sizes)

        B = np.empty((n_src, n_path, n_mesh, 3))  # B-field at cell location
        CVEC = np.empty((n_src, n_path, n_mesh, 3))  # current vectors

        # Compute and Broadcast
        idx_ends = np.cumsum(mesh_sizes)
        idx_starts = np.r_[0, idx_ends[:-1]]

        for start, end, ii in zip(idx_starts, idx_ends, idx_cur, strict=False):
            start_all = idx_starts_all7[ii]
            end_all = idx_ends_all7[ii]

            # B - broadcast (better memory and speed than concatenate)
            B[:, :, start:end] = B_all[:, :, start_all:end_all]

            # TVEC - apply rot and broadcast
            n = mesh_sizes_all[ii]
            cvec = np.tile(meshes[ii]["cvecs"], (n_path, 1))
            rot = np.repeat(tgt_ori[ii], n, axis=0)

            cvec_rot = R.from_quat(rot).apply(cvec).reshape(n_path, n, 3)
            # print(cvec_rot)
            CVEC[:, :, start:end] = cvec_rot[np.newaxis, :]

        # Force and Torque computation
        F = np.cross(CVEC, B)
        T = np.zeros_like(F)

        # broadcast into B (OBS7 needed below) to save memory? !!!!!!!!!!!!!!!!!!!!!!!!
        F_all[:, :, mask_mesh_current] = F
        T_all[:, :, mask_mesh_current] = T

    # PIVOT ############################################################################
    #  - force adds to torque via Pivot
    if pivot is not None:
        POS_PIV = np.zeros((n_src, n_path, n_mesh_all, 3))

        for j, piv in enumerate(pivot):
            start, end = idx_starts_all[j], idx_ends_all[j]
            start7, end7 = idx_starts_all7[j], idx_ends_all7[j]

            if mask_magnet[j]:
                diff = OBS7[:, start7:end7:7] - piv[:, np.newaxis, :]
            else:
                diff = OBS7[:, start7:end7] - piv[:, np.newaxis, :]

            POS_PIV[:, :, start:end] = diff

        T_all += np.cross(POS_PIV, F_all)

    # REDUCE AND SQUEEZE OUTPUT ###########################################################

    # Reduce mesh to targets (sumover) -> (n_src, n_path, n_tgt, 3)
    F_all = np.add.reduceat(F_all, idx_starts_all, axis=2)
    T_all = np.add.reduceat(T_all, idx_starts_all, axis=2)

    # Sumup targets that are in Collections
    F_all = np.add.reduceat(F_all, coll_idx, axis=2)
    T_all = np.add.reduceat(T_all, coll_idx, axis=2)

    if squeeze:
        return np.squeeze(F_all), np.squeeze(T_all)

    return F_all, T_all
