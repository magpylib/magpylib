"""Force implementation."""

# pylint: disable=import-outside-toplevel

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.fields.field_BH import _preserve_paths, getB
from magpylib._src.input_checks import check_dimensions, check_excitations
from magpylib._src.obj_classes.class_BaseTransform import (
    pad_path_properties,
    pad_path_property,
)
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


def _generate_path_meshes(targets, n_path, eps):
    """
    Generate meshes for each target at each path step.

    Parameters
    ----------
    targets : list
        List of target objects
    n_path : int
        Path length
    eps : float
        Finite difference step size

    Returns
    -------
    list of dict
        Each dict contains:
        - 'observers': (n_path, n_mesh, 3) or (n_path, n_mesh, 7, 3) for magnets
        - 'type': 'magnet' or 'current'
        - 'moments': (n_path, n_mesh, 3) for magnets
        - 'cvecs': (n_path, n_mesh, 3) for currents
        - 'n_mesh': number of mesh points
    """
    meshes = []
    eps_vec = _create_eps_vector(eps)

    for tgt in targets:
        # Get base mesh (path-independent geometry)
        base_mesh = tgt._generate_mesh()

        # Check for path-varying meshing (not yet supported)
        if path_vars := base_mesh.get("path_vars", ()):
            msg = (
                f"Detected path-varying meshing for target {tgt!r}: {path_vars}. "
                "which are not yet supported in getFT. "
            )
            raise NotImplementedError(msg)

        n_mesh = len(base_mesh["pts"])

        # Get path-varying transforms
        positions = pad_path_property(tgt._position, n_path)  # (n_path, 3)
        orientations = pad_path_property(tgt._orientation, n_path)  # Rotation object

        # Transform mesh for each path step (vectorized)
        is_magnet = "moments" in base_mesh

        # Broadcast base mesh to all path steps and apply rotations
        base_pts_flat = np.tile(base_mesh["pts"], (n_path, 1))  # (n_path*n_mesh, 3)
        orientations_repeated = R.from_quat(
            np.repeat(orientations.as_quat(), n_mesh, axis=0)
        )  # n_path*n_mesh rotations
        obs_all = orientations_repeated.apply(base_pts_flat).reshape(n_path, n_mesh, 3)
        obs_all += positions[:, np.newaxis, :]  # Add positions

        if is_magnet:
            # Add finite difference steps
            observers = obs_all[:, :, np.newaxis, :] + eps_vec[np.newaxis, np.newaxis, :, :]

            # Transform moments
            base_moments_flat = np.tile(base_mesh["moments"], (n_path, 1))
            moments = orientations_repeated.apply(base_moments_flat).reshape(
                n_path, n_mesh, 3
            )
        else:
            observers = obs_all

            # Transform current vectors
            base_cvecs_flat = np.tile(base_mesh["cvecs"], (n_path, 1))
            cvecs = orientations_repeated.apply(base_cvecs_flat).reshape(
                n_path, n_mesh, 3
            )

        if is_magnet:
            mesh_data = {
                "observers": observers,
                "moments": moments,
                "type": "magnet",
                "n_mesh": n_mesh,
            }
        else:
            mesh_data = {
                "observers": observers,
                "cvecs": cvecs,
                "type": "current",
                "n_mesh": n_mesh,
            }

        meshes.append(mesh_data)

    return meshes


def _flatten_observers(target_meshes):
    """
    Flatten all observer points into a single array.

    Parameters
    ----------
    target_meshes : list of dict
        Mesh data from _generate_path_meshes

    Returns
    -------
    obs_flat : ndarray
        Shape (n_path, n_total_obs, 3)
    mesh_info : dict
        Metadata for unflattening:
        - 'starts': start indices for each target
        - 'ends': end indices for each target
        - 'starts_7': start indices including FD steps
        - 'ends_7': end indices including FD steps
        - 'is_magnet': boolean array
        - 'mesh_sizes': number of mesh points per target
    """
    n_path = target_meshes[0]["observers"].shape[0]

    # Calculate total observer points
    mesh_sizes = np.array([m["n_mesh"] for m in target_meshes])
    is_magnet = np.array([m["type"] == "magnet" for m in target_meshes])
    mesh_sizes_7 = mesh_sizes * np.where(is_magnet, 7, 1)

    n_total = np.sum(mesh_sizes_7)

    # Allocate flat array
    obs_flat = np.empty((n_path, n_total, 3))

    # Fill flat array
    starts_7 = np.r_[0, np.cumsum(mesh_sizes_7)[:-1]]
    ends_7 = np.cumsum(mesh_sizes_7)

    for i, mesh in enumerate(target_meshes):
        start = starts_7[i]
        end = ends_7[i]

        if mesh["type"] == "magnet":
            # Flatten the (n_mesh, 7, 3) to (n_mesh*7, 3)
            obs_flat[:, start:end, :] = mesh["observers"].reshape(n_path, -1, 3)
        else:
            obs_flat[:, start:end, :] = mesh["observers"]

    # Create index arrays for unflattening
    starts = np.r_[0, np.cumsum(mesh_sizes)[:-1]]
    ends = np.cumsum(mesh_sizes)

    mesh_info = {
        "starts": starts,
        "ends": ends,
        "starts_7": starts_7,
        "ends_7": ends_7,
        "is_magnet": is_magnet,
        "mesh_sizes": mesh_sizes,
        "mesh_sizes_7": mesh_sizes_7,
    }

    return obs_flat, mesh_info


def _compute_B_with_paths(sources, obs_flat, n_path):
    """
    Compute B-field with native path support.

    Parameters
    ----------
    sources : list
        List of source objects
    obs_flat : ndarray
        Observer points with shape (n_path, n_obs, 3)
    n_path : int
        Path length

    Returns
    -------
    B_flat : ndarray
        B-field with shape (n_src, n_path, n_obs, 3)
    """
    n_src = len(sources)
    n_obs = obs_flat.shape[1]

    # Check if any source has a path
    src_path_lengths = [len(src._position) for src in sources]
    n_src_path = max(src_path_lengths)

    # Simple case: no source paths
    if n_src_path == 1:
        return getB(sources, obs_flat, squeeze=False)[:, 0, 0, :]

    # General case: sources have paths
    B_flat = np.empty((n_src, n_path, n_obs, 3))

    # Preserve original source paths and pad them to n_path
    with _preserve_paths(sources, path_properties=None, copy=False):
        # Pad all source paths to n_path
        for src in sources:
            pad_path_properties(
                src,
                n_path,
                path_properties=["position", "orientation", *src._path_properties],
            )

        # Store padded path properties before modification
        src_path_data = []
        for src in sources:
            src_data = {
                "position": src._position.copy(),
                "orientation": src._orientation.as_quat().copy(),
            }
            for prop in src._path_properties:
                src_data[prop] = getattr(src, f"_{prop}").copy()
            src_path_data.append(src_data)

        # Compute for each path step
        for j in range(n_path):
            # Set each source to its j-th path position
            for i, src in enumerate(sources):
                src.position = src_path_data[i]["position"][j]
                src.orientation = R.from_quat(src_path_data[i]["orientation"][j])

                # Set all path properties to single values for this path step
                for prop in src._path_properties:
                    prop_val = src_path_data[i][prop]
                    setattr(src, f"_{prop}", np.array([prop_val[j]]))

            # Compute B-field at this path step
            B_flat[:, j] = getB(sources, obs_flat[j], squeeze=True)

    return B_flat


def _compute_force_torque(B_flat, target_meshes, mesh_info, pivot, eps, n_src, n_path):
    """
    Compute forces and torques from B-field.

    Parameters
    ----------
    B_flat : ndarray
        B-field with shape (n_src, n_path, n_obs_total, 3)
    target_meshes : list of dict
        Mesh data from _generate_path_meshes
    mesh_info : dict
        Metadata from _flatten_observers
    pivot : ndarray or None
        Pivot points with shape (n_tgt, n_path, 3) or None
    eps : float
        Finite difference step size
    n_src : int
        Number of sources
    n_path : int
        Path length

    Returns
    -------
    F_all : ndarray
        Forces with shape (n_src, n_path, n_tgt, 3)
    T_all : ndarray
        Torques with shape (n_src, n_path, n_tgt, 3)
    """
    n_tgt = len(target_meshes)
    F_all = np.zeros((n_src, n_path, n_tgt, 3))
    T_all = np.zeros((n_src, n_path, n_tgt, 3))

    for tgt_idx, mesh in enumerate(target_meshes):
        start_7 = mesh_info["starts_7"][tgt_idx]
        end_7 = mesh_info["ends_7"][tgt_idx]
        n_mesh = mesh["n_mesh"]

        # Extract B-field for this target
        B_tgt = B_flat[:, :, start_7:end_7, :]  # (n_src, n_path, n_mesh*7 or n_mesh, 3)

        if mesh["type"] == "magnet":
            # Reshape to separate FD steps
            B_tgt = B_tgt.reshape(n_src, n_path, n_mesh, 7, 3)

            # Compute gradient using finite differences
            dB = np.empty((n_src, n_path, n_mesh, 3, 3))
            for i in range(3):  # ∂B/∂x, ∂B/∂y, ∂B/∂z
                dB[:, :, :, i, :] = (
                    B_tgt[:, :, :, 2 * i + 1, :] - B_tgt[:, :, :, 2 * i + 2, :]
                )
            dB /= 2 * eps

            # B at center points
            B_center = B_tgt[:, :, :, 0, :]  # (n_src, n_path, n_mesh, 3)

            # Broadcast moments
            moments = mesh["moments"]  # (n_path, n_mesh, 3)
            moments = moments[np.newaxis, :, :, :]  # (1, n_path, n_mesh, 3)

            # Force: F = (∇B) · m
            F = np.einsum("spijk,spik->spij", dB, moments)  # (n_src, n_path, n_mesh, 3)

            # Torque: T = m x B
            T = np.cross(moments, B_center, axis=-1)  # (n_src, n_path, n_mesh, 3)

        else:  # current
            # B at mesh points
            B_center = B_tgt  # (n_src, n_path, n_mesh, 3)

            # Broadcast current vectors
            cvecs = mesh["cvecs"]  # (n_path, n_mesh, 3)
            cvecs = cvecs[np.newaxis, :, :, :]  # (1, n_path, n_mesh, 3)

            # Force: F = I*dl x B
            F = np.cross(cvecs, B_center, axis=-1)  # (n_src, n_path, n_mesh, 3)
            T = np.zeros_like(F)

        # Add pivot contribution to torque
        if pivot is not None:
            # Get observer positions at centers
            if mesh["type"] == "magnet":
                obs_center = mesh["observers"][:, :, 0, :]  # (n_path, n_mesh, 3)
            else:
                obs_center = mesh["observers"]  # (n_path, n_mesh, 3)

            # Pivot for this target: (n_path, 3)
            piv = pivot[tgt_idx]  # (n_path, 3)

            # Position relative to pivot
            r = obs_center - piv[:, np.newaxis, :]  # (n_path, n_mesh, 3)
            r = r[np.newaxis, :, :, :]  # (1, n_path, n_mesh, 3)

            # T += r x F
            T += np.cross(r, F, axis=-1)

        # Sum over mesh points
        F_all[:, :, tgt_idx, :] = np.sum(F, axis=2)
        T_all[:, :, tgt_idx, :] = np.sum(T, axis=2)

    return F_all, T_all


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
    # INPUT VALIDATION & NORMALIZATION ########################################

    # Validate and flatten targets (including collections)
    targets, coll_idx = _check_format_input_targets(targets)

    # Validate sources
    sources, _ = format_src_inputs(sources)
    n_src = len(sources)

    # Determine path length (max across all sources and targets)
    tgt_path_lengths = [len(tgt._position) for tgt in targets]
    src_path_lengths = [len(src._position) for src in sources]
    n_path = max(tgt_path_lengths + src_path_lengths)

    # Validate and broadcast pivot to (n_tgt, n_path, 3) or None
    pivot = _check_format_input_pivot(pivot, targets, n_path)

    # Validate eps
    _check_eps(eps)

    # GENERATE PATH-AWARE MESHES ##############################################

    target_meshes = _generate_path_meshes(targets, n_path, eps)

    # Mesh report
    if meshreport:
        logging.info("Mesh report:")
        for t, mesh in zip(targets, target_meshes, strict=False):
            logging.info("  Target %s: %d points", t, mesh["n_mesh"])
        logging.info("")

    # Return mesh for analysis
    if return_mesh:
        # Return old-style mesh format for compatibility
        return [
            {
                "pts": m["observers"][0, :, 0, :]
                if m["type"] == "magnet"
                else m["observers"][0],
                **(
                    {"moments": m["moments"][0]}
                    if m["type"] == "magnet"
                    else {"cvecs": m["cvecs"][0]}
                ),
            }
            for m in target_meshes
        ]

    # FLATTEN OBSERVERS FOR B-FIELD COMPUTATION ###############################

    obs_flat, mesh_info = _flatten_observers(target_meshes)

    # COMPUTE B-FIELD #########################################################

    B_flat = _compute_B_with_paths(sources, obs_flat, n_path)

    # COMPUTE FORCES & TORQUES ################################################

    F_all, T_all = _compute_force_torque(
        B_flat, target_meshes, mesh_info, pivot, eps, n_src, n_path
    )

    # REDUCE COLLECTIONS ######################################################

    # Sum up targets that are in Collections
    F_all = np.add.reduceat(F_all, coll_idx, axis=2)
    T_all = np.add.reduceat(T_all, coll_idx, axis=2)

    # SQUEEZE OUTPUT ##########################################################

    if squeeze:
        return np.squeeze(F_all), np.squeeze(T_all)

    return F_all, T_all
