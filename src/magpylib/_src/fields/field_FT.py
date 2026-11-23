"""Force implementation."""

# pylint: disable=import-outside-toplevel

import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.fields.field_BH import _check_sources, _summed_field_on_path
from magpylib._src.fields.field_FT_func import _cell_force_torque, _check_eps
from magpylib._src.input_checks import check_dimensions, check_excitations
from magpylib._src.obj_classes.class_BaseTransform import pad_path_property
from magpylib._src.utility import format_src_inputs


def _needs_mesh(t):
    """Whether a target is cut into cells for the force computation."""
    from magpylib._src.obj_classes.class_magnet_Sphere import Sphere  # noqa: PLC0415
    from magpylib._src.obj_classes.class_misc_Dipole import Dipole  # noqa: PLC0415

    return not isinstance(t, Dipole | Sphere)


def _resolve_meshing(meshing, flat_targets, given, keyable):
    """Return the validated meshing value of every flat target.

    Parameters
    ----------
    meshing : int | tuple | dict | None
        The ``meshing`` input of getFT.
    flat_targets : list
        Targets with Collections flattened.
    given : list
        The target each flat target was given as (itself, or its Collection).
    keyable : dict
        id -> object for everything a dict key may name: the targets that take a
        mesh, and the given Collections and their nested Collections.
    """
    from magpylib._src.obj_classes.class_magnet_Cuboid import Cuboid  # noqa: PLC0415
    from magpylib._src.obj_classes.class_magnet_Cylinder import (  # noqa: PLC0415
        Cylinder,
    )
    from magpylib._src.obj_classes.class_magnet_CylinderSegment import (  # noqa: PLC0415
        CylinderSegment,
    )

    needs = [_needs_mesh(t) for t in flat_targets]
    if isinstance(meshing, dict):
        exact = [
            k
            for k in meshing
            if any(k is t for t, n in zip(flat_targets, needs, strict=True) if not n)
        ]
        if exact:
            msg = f"Input meshing has a value for {exact[0]}, which needs no meshing: it is exact."
            raise ValueError(msg)
        unknown = [k for k in meshing if id(k) not in keyable]
        if unknown:
            msg = (
                "Input meshing has keys that are not targets of this getFT call: "
                f"{unknown}."
            )
            raise ValueError(msg)

    values = []
    for t, g, need in zip(flat_targets, given, needs, strict=True):
        if not need:
            values.append(None)
            continue
        if isinstance(meshing, dict):
            # the target itself wins, then the Collections it sits in; a key can
            # only be a given target or a Collection under one, so the walk ends
            # at the outermost given Collection at the latest
            value = None
            key = t
            while key is not None:
                if key in meshing:
                    value = meshing[key]
                    break
                key = getattr(key, "parent", None)
        else:
            value = meshing
        if value is None:
            where = "" if t is g else f" (in {g})"
            msg = (
                f"Missing meshing input for target {t}{where}. Give the number of "
                "cells to getFT, e.g. getFT(sources, targets, meshing=100), or a "
                "dict with one value per target."
            )
            raise ValueError(msg)
        if isinstance(value, list | tuple | np.ndarray) and not isinstance(
            t, Cuboid | Cylinder | CylinderSegment
        ):
            msg = (
                f"Input meshing {value!r} for {t}: a triple only cuts a Cuboid, "
                "Cylinder or CylinderSegment; give an integer, or a dict with one "
                "value per target."
            )
            raise ValueError(msg)  # noqa: TRY004  # every meshing error is a ValueError
        values.append(t._validate_meshing(value))
    return values


def _check_format_input_targets(targets, meshing):
    """
    Check and format targets input
    - flatten Collections
    - check if allowed instance
    - resolve and validate the meshing value of every target

    Returns
    -------
    the targets as given (a list), the flat list of targets, the list of
    collection indices for summation, and the meshing value of every flat target
    """
    if not isinstance(targets, list):
        targets = [targets]

    from magpylib._src.obj_classes.class_Collection import Collection  # noqa: PLC0415

    # flatten out collections, keep indices for later summation in getFT
    flat_targets = []
    given = []
    keyable = {}
    coll_idx = []
    idx = 0
    for t in targets:
        if isinstance(t, Collection):
            len_sources = len(t.sources_all)
            if len_sources == 0:
                msg = f"Given target Collection {t} has no target sources."
                raise ValueError(msg)
            flat_targets.extend(t.sources_all)
            given.extend([t] * len_sources)
            keyable[id(t)] = t
            keyable.update({id(c): c for c in t.collections_all})
            coll_idx.append(idx)
            idx += len_sources
        else:
            flat_targets.append(t)
            given.append(t)
            coll_idx.append(idx)
            idx += 1

    # check if dimensions and excitations are initialized
    check_dimensions(flat_targets)
    check_excitations(flat_targets)

    for t in flat_targets:
        if not hasattr(t, "_generate_mesh"):
            msg = (
                "Input targets must be allowed Magpylib target objects; "
                f"instead received type {type(t)}."
            )
            raise ValueError(msg)

    keyable.update({id(t): t for t in flat_targets if _needs_mesh(t)})
    meshings = _resolve_meshing(meshing, flat_targets, given, keyable)

    return targets, flat_targets, coll_idx, meshings


def _target_centroids(targets, n_path):
    """Centroid path of each target as given, shape (t, p, 3).

    A Collection target gives its own volume-weighted centroid, so that the torque
    on it is the torque on one rigid body.
    """
    return np.array([pad_path_property(t._centroid, n_path) for t in targets])


def _check_format_input_pivot(pivot, targets, n_members, n_path):
    """
    Check and format pivot input

    Parameters
    ----------
    pivot : 'centroid' | None | array-like
        Pivot input of getFT.
    targets : list
        Targets as given, a Collection counting as one.
    n_members : list of int
        Number of flat targets each given target expands to.
    n_path : int
        Path length.

    Returns
    -------
    array of pivots with shape (t_flat, p, 3) or None
    """
    n_tgt = len(targets)
    msg = (
        "Input pivot must be 'centroid', None, or array-like with shape "
        "(3,), (t, 3), (t, p, 3), or (p, 3) for one target, for t targets "
        f"(a Collection counts as one) and path length p; instead received {pivot!r}."
    )

    if pivot is None:
        return None

    if isinstance(pivot, str) and pivot == "centroid":
        pivot = _target_centroids(targets, n_path)
    elif isinstance(pivot, list | tuple | np.ndarray):
        try:
            pivot = np.array(pivot, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(msg) from e

        if pivot.shape == (3,):
            pivot = np.tile(pivot, (n_tgt, n_path, 1))
        elif pivot.shape == (n_tgt, 3):
            pivot = np.repeat(pivot[:, np.newaxis, :], n_path, axis=1)
        elif n_tgt == 1 and pivot.shape == (n_path, 3):
            pivot = pivot[np.newaxis]  # one target on a path: its own centroid path
        elif pivot.shape != (n_tgt, n_path, 3):
            raise ValueError(msg)
    else:
        raise ValueError(msg)

    # every member of a Collection target shares the Collection's pivot
    return np.repeat(pivot, n_members, axis=0)


def _generate_path_meshes(targets, meshings, n_path):
    """
    Generate meshes for each target at each path step.

    Parameters
    ----------
    targets : list
        List of target objects
    meshings : list
        Validated meshing value per target, None where the target needs none
    n_path : int
        Path length

    Returns
    -------
    list of dict
        Each dict contains:
        - 'observers': (n_path, n_mesh, 3)
        - 'moments': (n_path, n_mesh, 3) for magnets
        - 'cvecs': (n_path, n_mesh, 3) for currents
        - 'n_mesh': number of mesh points
    """
    meshes = []

    for tgt, meshing in zip(targets, meshings, strict=True):
        # Get mesh (potentially path-varying)
        base_mesh = tgt._generate_mesh(meshing)

        # Check if mesh is path-varying by dimensionality
        # Path-varying: shape (p_mesh, n_mesh, 3)
        # Path-independent: shape (n_mesh, 3)
        has_path_varying_mesh = base_mesh["pts"].ndim == 3

        # Determine mesh size from mesh
        if has_path_varying_mesh:
            # Mesh already includes path: shape (p_mesh, n_mesh, 3)
            n_mesh = base_mesh["pts"].shape[1]
        else:
            # Mesh is path-independent: shape (n_mesh, 3)
            n_mesh = len(base_mesh["pts"])

        # Get path-varying transforms
        positions = pad_path_property(tgt._position, n_path)  # (n_path, 3)
        orientations = pad_path_property(tgt._orientation, n_path)  # Rotation object

        # Transform mesh for each path step
        is_magnet = "moments" in base_mesh

        # Prepare mesh data: either pad path-varying mesh or tile static mesh
        if has_path_varying_mesh:
            # Mesh already has path dimension: (p_mesh, n_mesh, 3) -> pad to n_path
            base_pts = pad_path_property(base_mesh["pts"], n_path)
        else:
            # Mesh is path-independent: (n_mesh, 3) -> tile to all path steps
            base_pts = np.tile(base_mesh["pts"], (n_path, 1, 1))

        # Apply rotations (vectorized across all path steps)
        base_pts_flat = base_pts.reshape(-1, 3)  # (n_path*n_mesh, 3)
        orientations_repeated = R.from_quat(
            np.repeat(orientations.as_quat(), n_mesh, axis=0)
        )
        obs_all = orientations_repeated.apply(base_pts_flat).reshape(n_path, n_mesh, 3)
        obs_all += positions[:, np.newaxis, :]

        if is_magnet:
            # Prepare and transform moments
            if has_path_varying_mesh:
                base_moments = pad_path_property(base_mesh["moments"], n_path)
            else:
                base_moments = np.tile(base_mesh["moments"], (n_path, 1, 1))
            base_moments_flat = base_moments.reshape(-1, 3)
            moments = orientations_repeated.apply(base_moments_flat).reshape(
                n_path, n_mesh, 3
            )
        else:
            # Prepare and transform current vectors
            if has_path_varying_mesh:
                base_cvecs = pad_path_property(base_mesh["cvecs"], n_path)
            else:
                base_cvecs = np.tile(base_mesh["cvecs"], (n_path, 1, 1))
            base_cvecs_flat = base_cvecs.reshape(-1, 3)
            cvecs = orientations_repeated.apply(base_cvecs_flat).reshape(
                n_path, n_mesh, 3
            )

        if is_magnet:
            mesh_data = {
                "observers": obs_all,
                "moments": moments,
                "n_mesh": n_mesh,
            }
        else:
            mesh_data = {
                "observers": obs_all,
                "cvecs": cvecs,
                "n_mesh": n_mesh,
            }

        meshes.append(mesh_data)

    return meshes


def _compute_force_torque(sources, target_meshes, pivot, eps, n_path):
    """
    Compute forces and torques on the meshed targets.

    Parameters
    ----------
    sources : list
        List of source objects
    target_meshes : list of dict
        Mesh data from _generate_path_meshes
    pivot : ndarray or None
        Pivot points with shape (n_tgt, n_path, 3) or None
    eps : float
        Finite difference step size
    n_path : int
        Path length, at least every source's path length. The cells of path
        step j see the sources at their path step j.

    Returns
    -------
    F_all : ndarray
        Forces with shape (n_src, n_path, n_tgt, 3)
    T_all : ndarray
        Torques with shape (n_src, n_path, n_tgt, 3)
    """
    # all path steps in one computation, on the chunks and threads of getB. The
    # sources are only read, so threads can compute with them at the same time
    # (#1074). Warnings point at the caller of getFT.
    sources, src_list = _check_sources(sources, "B", "auto", stacklevel=4)

    def field(observers):
        return _summed_field_on_path(sources, src_list, observers, "B", "auto")

    n_tgt = len(target_meshes)
    F_all = np.zeros((len(sources), n_path, n_tgt, 3))
    T_all = np.zeros((len(sources), n_path, n_tgt, 3))

    # the cells of all magnets in one field computation, then of all currents
    for kind in ("moments", "cvecs"):
        inds = [i for i, mesh in enumerate(target_meshes) if kind in mesh]
        if not inds:
            continue
        meshes = [target_meshes[i] for i in inds]
        F, T = _cell_force_torque(
            field,
            np.concatenate([mesh["observers"] for mesh in meshes], axis=1),
            **{kind: np.concatenate([mesh[kind] for mesh in meshes], axis=1)},
            pivot=None
            if pivot is None
            else np.repeat(
                pivot[inds], [mesh["n_mesh"] for mesh in meshes], axis=0
            ).swapaxes(0, 1),
            eps=eps,
        )
        ends = np.cumsum([mesh["n_mesh"] for mesh in meshes])
        for i, start, end in zip(
            inds, ends - np.diff(ends, prepend=0), ends, strict=True
        ):
            F_all[:, :, i] = np.sum(F[:, :, start:end], axis=2)
            T_all[:, :, i] = np.sum(T[:, :, start:end], axis=2)

    return F_all, T_all


def getFT(
    sources,
    targets,
    pivot="centroid",
    eps=1e-5,
    squeeze=True,
    *,
    meshing=None,
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
        Can be a 1D list of t target objects. A Collection counts as one target
        and gives the force and torque on it as one body.
    meshing : int | tuple | dict | None, default None, keyword-only
        How finely each target is cut for the computation: the target number of
        cells (magnets) or segments (currents). An integer applies to every
        target, including each magnet inside a Collection. A dict keyed by
        target gives one value per target, ``meshing={ring: 100, coil: 40}``;
        a Collection key covers its members. A Cuboid value may be a triple
        ``(n1, n2, n3)`` of divisions along its sides, and a Cylinder or
        CylinderSegment value a triple ``(nphi, nr, nh)`` of divisions along the
        arc, the radius and the height. A bare triple applies when every meshed
        target takes one. Required for every target but ``Dipole``
        and ``Sphere``, which are exact. Each class documents what a value means
        for it and its minimum, such as 4 for a ``Circle``.
    pivot : 'centroid' | None | array-like, shape (3,), (t, 3), (t, p, 3) or (p, 3), default 'centroid'
        Pivot point through which the force contributes to the torque. If
        ``'centroid'``, each target's centroid is used; for a Collection target
        this is the Collection's centroid, volume-weighted over its magnets or,
        for a Collection without volume such as a coil, the mean of its members'
        centroids, so that the torque is the torque on one rigid body. If
        ``None``, no pivot is applied (may yield nonphysical
        results). If an array of shape (3,), the same pivot is used for all
        targets. Shapes (t, 3) and (t, p, 3) provide per-target or
        per-target-per-path pivots, where t counts the targets as given, a
        Collection counting as one, like the output does; a single target on a
        path also takes shape (p, 3), such as its own ``centroid``.
    eps : float, default 1e-5
        Finite-difference step size for gradient-field computation for magnet
        targets. A good value is 1e-5 * characteristic_system_size (e.g.,
        magnet size or source-target distance).
    squeeze : bool, default True
        If ``True``, dimensions of size 1 in the output are removed.
    return_mesh : bool, default False, keyword-only
        If ``True``, returns the force mesh of each target instead of force and
        torque, as a list of dictionaries with keys ``'pts'``, and ``'moments'``
        (magnets) or ``'cvecs'`` (currents).

        .. deprecated:: 6.0.0
            A function returning a target's cells is planned.

    Returns
    -------
    tuple[ndarray, ndarray]
        Force and torque with shapes (s, p, t, 3), where s is the
        number of sources, p the path length, and t the number of
        targets. If ``squeeze`` is ``True``, dimensions of size 1 are removed.

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
    ... )
    >>> F, T = magpy.getFT(cube, circ, meshing=50)
    >>> print(f'force: {np.round(F, decimals=2)} N')
    force: [ 13.65  27.31 -81.93] N
    >>> print(f'torque: {np.round(T, decimals=2) + 0.0} N*m')
    torque: [-8.55  4.27  0.  ] N*m
    """
    # INPUT VALIDATION & NORMALIZATION ########################################

    # Validate and flatten targets (including collections)
    targets_in, targets, coll_idx, meshings = _check_format_input_targets(
        targets, meshing
    )
    n_members = np.diff([*coll_idx, len(targets)])  # flat targets per given target

    # Validate sources
    sources, src_list = format_src_inputs(sources)

    # Determine path length (max across all sources and targets), with the
    # sources of a Collection, which may have paths of their own
    tgt_path_lengths = [tgt._get_path_len() for tgt in targets]
    src_path_lengths = [src._get_path_len() for src in sources + src_list]
    n_path = max(tgt_path_lengths + src_path_lengths)

    # Validate and broadcast pivot to (n_tgt_flat, n_path, 3) or None
    pivot = _check_format_input_pivot(pivot, targets_in, n_members, n_path)

    # Validate eps
    _check_eps(eps)

    # GENERATE PATH-AWARE MESHES ##############################################

    target_meshes = _generate_path_meshes(targets, meshings, n_path)

    if return_mesh:
        warnings.warn(
            "getFT(..., return_mesh=True) is deprecated and will be removed in a "
            "future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            {
                "pts": m["observers"][0],
                **{k: m[k][0] for k in ("moments", "cvecs") if k in m},
            }
            for m in target_meshes
        ]

    # COMPUTE FORCES & TORQUES ################################################

    F_all, T_all = _compute_force_torque(sources, target_meshes, pivot, eps, n_path)

    # REDUCE COLLECTIONS ######################################################

    # Sum up targets that are in Collections
    F_all = np.add.reduceat(F_all, coll_idx, axis=2)
    T_all = np.add.reduceat(T_all, coll_idx, axis=2)

    # SQUEEZE OUTPUT ##########################################################

    if squeeze:
        return np.squeeze(F_all), np.squeeze(T_all)

    return F_all, T_all
