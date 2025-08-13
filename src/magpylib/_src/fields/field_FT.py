import numpy as np
from magpylib._src.fields.field_wrap_BH import getB
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_dimensions
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_Sensor import Sensor
from magpylib._src.utility import format_src_inputs

# def check_format_input_sources(sources):
#     """
#     Check and format sources input
#     """
#     if not isinstance(sources, list):
#         sources = [sources]
    
#     from magpylib._src.obj_classes.class_Collection import Collection
#     for s in sources:
#         if not hasattr(s, '_field_func') and not isinstance(s, Collection):
#             msg = (
#                 "getFT bad source input. Sources can only be Magpylib source objects."
#                 f" Instead received type {type(s)} source."
#             )
#             raise ValueError(msg)
#     return sources


def check_format_input_targets(targets):
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
        from magpylib._src.obj_classes.class_Collection import Collection
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
        from magpylib._src.obj_classes.class_misc_Dipole import Dipole
        if not isinstance(t, (Dipole,)):

            if not hasattr(t, 'meshing'):
                msg = (
                    "getFT bad target input. Targets can only be Magpylib objects Cuboid,..."
                    f" Instead received type {type(t)} target."
                )
                raise ValueError(msg)

            # check if meshing parameter is explicitly set
            if t.meshing is None:
                msg = (
                    f"getFT missing meshing input for target {t}."
                    " All targets must have the meshing parameter explicitly set."
                )
                raise ValueError(msg)

    return flat_targets, coll_idx


# def check_format_input_pivot(pivot, targets):
#     """
#     Check and format pivot input

#     Returns
#     -------
#     list of pivots with shape (n_target, ...) can be ragged if some tgts have path and others do not
#     """

#     msg = (
#         "Bad getFT pivot input. Input pivot must be str 'centroid', None or array_like of shape (3,)."
#         " It can also be (n,3) when there are n targets providing a different pivot for every target."
#     )

#     if isinstance(pivot, str) and pivot == "centroid":
#         return np.array([t.centroid for t in targets])

#     if pivot is None:
#         return None

#     if isinstance(pivot, (list, tuple, np.ndarray)):
#         try:
#             pivot = np.array(pivot, dtype=float)
#         except (ValueError, TypeError) as e:
#             raise ValueError(msg) from e
        
#         if pivot.shape == (3,):
#             return np.tile(pivot, (len(targets), 1))
#         if pivot.shape == (len(targets), 3):
#             return pivot

#     raise ValueError(msg)


def create_eps_vector(eps):
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


# def getFT_NOPATH(sources, targets, pivot="centroid", eps=1e-5, squeeze=True, meshreport=False):
#     """
#     Compute magnetic force and torque acting on the targets that are exposed
#     to the magnetic field of the sources.

#     SI units are assumed for all inputs and outputs.

#     Parameters
#     ----------
#     sources: source and collection objects or 1D list thereof
#         Sources that generate the magnetic field. Can be a single source (or collection)
#         or a 1D list of l sources and/or collection objects.

#     targets: single object or 1D list of t objects that are Sphere, Cuboid, Polyline,
#         Cylinder, CylinderSegment, or Dipole. Force and Torque acting on targets in the magnetic
#         field generated by the sources will be computed. A target (except of Dipole) must have a valid
#         `meshing` parameter.

#     pivot: str, None, or array_like, default="centroid"
#         The Force adds to the Torque via the pivot point. For a freely floating magnet
#         this would be the barycenter (= centroid when the density is homogeneous).
#         If pivot="centroid" the centroid is selected as the pivot point for all targets.
#         If pivot=None no pivot is used. This will give unphysical results.
#         If pivot=array_like of shape (3,) the same pivot is used for all targets.
#         Alternatively one can provide an individual pivot point for each target.

#     eps: float, default=1e-5
#         This is only used for magnet targets for computing the magnetic field gradient
#         using finite differences (FD). `eps` is the FD step size. A good value
#         is 1e-5 * characteristic_system_size (magnet size, distance between sources
#         and targets, ...).

#     squeeze: bool, default=True
#         The output of the computation has the shape (n,3) where n corresponds to the number
#         of targets. By default this is reduced to (3,) when there is only one target.
    
#     meshreport: bool, default=False
#         If True, a report of the mesh used for each target will be printed.

#     Returns
#     -------
#     Force-Torque as ndarray of shape (2,3), or (t,2,3) when t targets are given
#     """
#     # Input checks
#     targets, coll_idx = check_format_input_targets(targets)
#     pivot = check_format_input_pivot(pivot, targets)
#     sources = check_format_input_sources(sources)

#     # Get force types and create masks efficiently
#     mask_magnet = np.array([tgt._force_type == "magnet" for tgt in targets])
#     mask_current = np.array([tgt._force_type == "current" for tgt in targets])

#     # Important numbers
#     n_targets = len(targets)
#     n_sources = len(sources)
#     n_magnets = sum(mask_magnet)  # number of magnet targets
#     n_currents = sum(mask_current)  # number of current targets

#     # Allocate output arrays
#     FTOUT = np.zeros((2, n_sources, n_targets, 3))

#     # RUN MESHING FUNCTIONS ##############################################################
#     # Collect meshing function results - cannot separate mesh generation from generation 
#     #    of moments, lvecs, etc.
#     # Meshing functions are run now to collect all observers because the B-field is
#     #    computed for all targets at once, for efficiency reasons.
#     observer = []
#     mesh_sizes = []
#     mag_moments = []
#     cur_currents = []
#     cur_tvecs = []
#     eps_vec = create_eps_vector(eps)
#     for tgt in targets:

#         if tgt._force_type == "magnet":
#             mesh, mom = tgt._generate_mesh()
#             mag_moments.append(mom)
#             # if target is a magnet add 6 finite difference steps for gradient computation
#             mesh = (mesh[:, np.newaxis, :] + eps_vec[np.newaxis, :, :]).reshape(-1, 3)

#         if tgt._force_type == "current":
#             mesh, curr, tvec = tgt._generate_mesh()
#             cur_currents.append(curr)
#             cur_tvecs.append(tvec)

#         observer.append(mesh)
#         mesh_sizes.append(len(mesh))


#     # COMPUTE B FIELD ###################################################################
#     observer = np.concatenate(observer, axis=0)
#     B_all = getB(sources, observer, squeeze=False)
#     # shape (n_src, 1, 1, n_obs, 3)

#     # Indexing of observer and B_all arrays
#     mesh_sizes = np.array(mesh_sizes)
#     obs_ends = np.cumsum(mesh_sizes)
#     obs_starts = np.r_[0, obs_ends[:-1]]

#     # COMPUTATION IDEA:
#     #   The equations are very simple like F = DB.MOM, T = B x MOM.
#     #   All we need to do is to broadcast into large computation arrays that
#     #   contain DB, MOM, B, ... and apply the equations, sum over mesh cells,
#     #   and reshape the output to the desired shape.

#     # MAGNETS ########################################################################
#     if n_magnets > 0:
        
#         # Prepare index ranges for broadcasting
#         mesh_sizes_mag = mesh_sizes[mask_magnet] // 7
#         n_mesh_mag = sum(mesh_sizes_mag)
#         idx_ends = np.cumsum(mesh_sizes_mag)
#         idx_starts = np.r_[0, idx_ends[:-1]]

#         # Computation array allocations
#         POS = np.zeros((n_mesh_mag*n_sources, 3))   # central location of each cell
#         B = np.zeros((n_mesh_mag*n_sources, 3))     # B-field at each cell
#         DB = np.zeros((n_mesh_mag*n_sources, 3, 3)) # B-field gradient at each cell
#         MOM = np.zeros((n_mesh_mag*n_sources, 3))   # magnetic moment of each cell

#         # BROADCASTING into computation arrays:
#         #   rule: (src1 mesh1, src1 mesh2, src1 mesh3, ... src2 mesh1, src2 mesh2, ... )
#         for i, mom in enumerate(mag_moments):
#             # range in observer and B arrays
#             start = obs_starts[mask_magnet][i]
#             end = obs_ends[mask_magnet][i]
            
#             for j in range(n_sources):
#                 # range in computation arrays
#                 ids = idx_starts[i] + n_mesh_mag*j
#                 ide = idx_ends[i] + n_mesh_mag*j

#                 POS[ids : ide] = observer[start : end : 7]
#                 B[ids : ide] = B_all[j, 0, 0, start : end : 7]

#                 # ∂B/∂x
#                 DB[ids : ide, :, 0] = B_all[j, 0, 0, start+1 : end : 7]
#                 DB[ids : ide, :, 0] -= B_all[j, 0, 0, start+2 : end : 7]
#                 DB[ids : ide, :, 0] /= (2*eps)

#                 # ∂B/∂y
#                 DB[ids : ide, :, 1] = B_all[j, 0, 0, start+3 : end : 7]
#                 DB[ids : ide, :, 1] -= B_all[j, 0, 0, start+4 : end : 7]
#                 DB[ids : ide, :, 1] /= (2*eps)
                
#                 # ∂B/∂z
#                 DB[ids : ide, :, 2] = B_all[j, 0, 0, start+5 : end : 7]
#                 DB[ids : ide, :, 2] -= B_all[j, 0, 0, start+6 : end : 7]
#                 DB[ids : ide, :, 2] /= (2*eps)

#                 MOM[ids : ide] = mom

#         # ACTUAL FORCE AND TORQUE COMPUTATION
#         #   !!Performance!!: cross few time faster than einsum few times faster than array API.
#         #   Torque computation relies on force computation.
#         #   Therefore it makes no sense to separate the force and torque computation.

#         force = np.einsum('ijk,ik->ij', DB, MOM) # numpy only
#         #force = np.sum(DB * MOM[:, np.newaxis, :], axis=2) # array API
#         torque = np.cross(MOM, B)

#         # Add pivot point contribution to torque
#         if pivot is not None:
#             PIV = np.tile(np.repeat(pivot[mask_magnet], mesh_sizes_mag, axis=0), (n_sources, 1))
#             torque += np.cross(POS - PIV, force)

#         # Sum over mesh cells
#         idx_starts_all = np.concatenate([idx_starts + n_mesh_mag*j for j in range(n_sources)])
#         F = np.add.reduceat(force, idx_starts_all, axis=0)
#         T = np.add.reduceat(torque, idx_starts_all, axis=0)

#         # Broadcast into output arrays
#         FTOUT[0, :, mask_magnet] = F.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)
#         FTOUT[1, :, mask_magnet] = T.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)


#     # CURRENTS ########################################################################
#     if n_currents > 0:
#         # Prepare index ranges for broadcasting
#         mesh_sizes_cur = mesh_sizes[mask_current]
#         n_mesh_cur = sum(mesh_sizes_cur)
#         idx_ends = np.cumsum(mesh_sizes_cur)
#         idx_starts = np.r_[0, idx_ends[:-1]]

#         # Computation array allocations
#         POS = np.zeros((n_mesh_cur*n_sources, 3))  # central location of each cell
#         B = np.zeros((n_mesh_cur*n_sources, 3))    # B-field at POS
#         TVEC = np.zeros((n_mesh_cur*n_sources, 3)) # current path tangential vectors
#         CURR = np.zeros((n_mesh_cur*n_sources,))   # current

#         # BROADCASTING into computation arrays:
#         #   rule: (src1 mesh1, src1 mesh2, src1 mesh3, ... src2 mesh1, src2 mesh2, ... )
#         for i, (curr, tvec) in enumerate(zip(cur_currents, cur_tvecs)):
#             # range in observer and B arrays
#             start = obs_starts[mask_current][i]
#             end = obs_ends[mask_current][i]

#             for j in range(n_sources):
#                 # range in computation arrays
#                 ids = idx_starts[i] + n_mesh_cur*j
#                 ide = idx_ends[i] + n_mesh_cur*j

#                 POS[ids : ide] = observer[start : end]
#                 B[ids : ide] = B_all[j, 0, 0, start : end]
#                 TVEC[ids : ide] = tvec
#                 CURR[ids : ide] = curr

#         # ACTUAL FORCE AND TORQUE COMPUTATION
#         force = (CURR * np.cross(TVEC, B).T).T
#         torque = np.zeros_like(force)

#         # Add pivot point contribution to torque
#         if pivot is not None:
#             PIV = np.tile(np.repeat(pivot[mask_current], mesh_sizes_cur, axis=0), (n_sources, 1))
#             torque += np.cross(POS - PIV, force)

#         # Sum over mesh cells
#         idx_starts_all = np.concatenate([idx_starts + n_mesh_cur*j for j in range(n_sources)])
#         F = np.add.reduceat(force, idx_starts_all, axis=0)
#         T = np.add.reduceat(torque, idx_starts_all, axis=0)

#         # Broadcast into output arrays
#         FTOUT[0, :, mask_current] = F.reshape(n_sources, n_currents, 3).transpose(1, 0, 2)
#         FTOUT[1, :, mask_current] = T.reshape(n_sources, n_currents, 3).transpose(1, 0, 2)

#     # FINALIZE OUTPUT ##############################################################

#     # Sum up Collections
#     FTOUT = np.add.reduceat(FTOUT, coll_idx, axis=2)

#     if squeeze:
#         return np.squeeze(FTOUT)

#     return FTOUT

################################################################################
################################################################################
################################################################################
################################################################################

def check_format_input_pivot(pivot, targets, n_path):
    """
    Check and format pivot input

    Returns
    -------
    array of pivots with shape (n_tgt, n_path, 3) or None
    """
    msg = (
        "Bad getFT pivot input. Input pivot must be str 'centroid', `None`, or array_like of shape (3,)."
        " It can also be (n,3) when there are n targets providing a different pivot for every target."
        " It can also be (n,m,3) when there are n targets and pathlength is m."
    )

    if pivot is None:
        return None

    if isinstance(pivot, str) and pivot == "centroid":
        return np.array([np.pad(t._centroid, ((0,n_path-len(t._centroid)),(0,0)), "edge") for t in targets])


    if isinstance(pivot, (list, tuple, np.ndarray)):
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


def getFT(sources, targets, pivot="centroid", eps=1e-5, squeeze=True, meshreport=False):
    """
    Compute magnetic force and torque acting on the targets that are exposed
    to the magnetic field of the sources.

    SI units are assumed for all inputs and outputs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l sources and/or collection objects.

    targets: single target object or 1D list thereof.
        Objects on which the magnetic field acts, generating force and torque. All targets
        (except Dipoles) must have valid `meshing` parameters set.

    pivot: str, None, or array_like, default="centroid"
        The Force adds to the Torque via the pivot point. For a freely floating magnet
        this would be the barycenter (= centroid when the density is homogeneous).
        If pivot="centroid" the centroid is selected as the pivot point for all targets.
        If pivot=None no pivot is used. This will give unphysical results.
        If pivot=array_like of shape (3,) the same pivot is used for all targets.
        Alternatively one can provide an individual pivot point for each target.

    eps: float, default=1e-5
        Finite difference step size for gradient field computation required for magnet targets.
        A good value is 1e-5 * characteristic_system_size (magnet size, distance between sources
        and targets, ...).

    squeeze: bool, default=True
        The output of the computation has the shape (2,n,m,o,3) where n corresponds to the number
        of sources, m to the path length, and o to the number of targets. By default all dimensions
        of size 1 are eliminated.
    
    meshreport: bool, default=False
        If True, a report of the mesh used for each target will be printed.

    Returns
    -------
    Force-Torque as ndarray of shape (2,n,m,o,3), when n sources, m path length, and o targets
        are given.
    
    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy

    >>> cube = magpy.magnet.Cuboid(
    ...     dimension=(1, 1, 1),
    ...     polarization=(.1, .2, .3)
    ... )
    >>> circ = magpy.current.Circle(
    ...     diameter=2,
    ...     current=1e3,
    ...     position=(0, 0, 1),
    ...     meshing=50,
    ... )
    >>> F, T = magpy.getFT(cube, circ)

    >>> print(f'force: {np.round(F, decimals=2)} N')
    force: [ 13.65  27.31 -81.93] N

    >>> print(f'torque: {np.round(T, decimals=2)} N')
    torque: [-8.55  4.27 -0.  ] N
    
    Notes:
    ------
    The force equations are F = DB.MOM, T = B x MOM + F x R. The computation relies
    on a numerical approach, where the targets are split up into a lot of small
    parts and the contributions from all parts are summed up.

    For the gradient field, a finite difference approach is used.
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
    targets, coll_idx = check_format_input_targets(targets)
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
    pivot = check_format_input_pivot(pivot, targets, n_path)

    # RUN MESHING FUNCTIONS ###################################################
    #  - collect all meshing infos and prepare for later broadcasting

    meshes = [tgt._generate_mesh() for tgt in targets]

    mesh_sizes_all = np.array([len(m["pts"]) for m in meshes])
    if meshreport:
        print("Mesh report:")
        for t, m in zip(targets, mesh_sizes_all):
            print(f"  Target {t}: {m} points")
        print()
    
    mask_current = np.array(["currents" in m for m in meshes])
    mask_magnet = np.array(["moments" in m for m in meshes])

    n_magnet = sum(mask_magnet)
    n_current = sum(mask_current)
    
    # Add 6 FD steps for magnet gradient field computation
    eps_vec = create_eps_vector(eps)
    for i in np.where(mask_magnet)[0]:
        meshes[i]["pts"] = (meshes[i]["pts"][:, np.newaxis, :] + eps_vec[np.newaxis, :, :]).reshape(-1, 3)

    # Replace pts by sensor, transfer tgt path, apply padding
    for tgt,m in zip(targets, meshes):
        padlength = n_path - len(tgt._position)
        padded_path = np.pad(tgt._position, ((0,padlength), (0,0)), 'edge')
        m["pts"] = Sensor(
                    pixel=m["pts"],
                    position=padded_path,
                    orientation=tgt._orientation,
        )

    # Allocate output array
    FTOUT = np.zeros((2, n_src, n_path, n_tgt, 3))

    # COMPUTE B FIELD ###################################################################
    #  - B_all shape  (n_tgt, n_src, n_path, n_mesh, 3)
    #  - if ragged inputs (targets with different mesh sizes), compute B field for each
    #    sensor separately. in this case the first dim is a list
    #  - Improve this computation when getB allows ragged inputs

    sensors = [m["pts"] for m in meshes]

    sensors[0].pixel = sensors[0].orientation.inv().apply(sensors[0].pixel)

    if len(set(mesh_sizes_all)) == 1:
        B_all = getB(sources, sensors, squeeze=False)
        B_all = np.moveaxis(B_all, 2, 0)
    else: # ragged inputs - eval mesh by mesh
        B_all = [np.squeeze(getB(sources, s, squeeze=False), axis=2) for s in sensors]

    print(sensors[0].orientation.apply(np.squeeze(B_all)))

    # Transform B from local sensor coords into global coords ! :(((
    #  - this is another major problem of this sensor-based approach
    for i,(b, s) in enumerate(zip(B_all, sensors)):
        for j in range(n_path):
            shp = b[:,j].shape
            bb = np.reshape(b[:,j], (-1,3))
            bb = s._orientation[j].apply(bb)
            bb = np.reshape(bb, shp)
            B_all[i][:,j] = bb


    # MAGNETS ########################################################################
    if n_magnet > 0:

        # Computation array allocation
        #  - use broadcasting instead of concatenate (better memory and speed)
        mesh_sizes = mesh_sizes_all[mask_magnet]
        n_mesh = sum(mesh_sizes)

        B = np.empty((n_src, n_path, n_mesh, 3))     # B-field at cell location
        DB = np.empty((n_src, n_path, n_mesh, 3, 3)) # B-field gradient at cell location
        MOM = np.empty((n_src, n_path, n_mesh, 3))   # magnetic moment of cells
        if pivot is not None:
            POS_PIV = np.empty((n_src, n_path, n_mesh, 3))  # relative cell pos to pivot

        # Broadcasting and application of pos & ori
        idx_all = np.where(mask_magnet)[0]
        idx_ends = np.cumsum(mesh_sizes)
        idx_starts = np.r_[0, idx_ends[:-1]]
        for start, end, i in zip(idx_starts, idx_ends, idx_all):
            sens = meshes[i]["pts"]
            mom = meshes[i]["moments"]

            # eliminate FD steps of pixel
            sens.pixel = sens.pixel[::7]

            B[:, :, start:end] = B_all[i][:, :, ::7]

            # ∂B/∂x
            DB[:, :, start:end, 0] = B_all[i][:, :, 1 :: 7]
            DB[:, :, start:end, 0] -= B_all[i][:, :, 2 :: 7]
            DB[:, :, start:end, 0] /= (2*eps)

            # ∂B/∂y
            DB[:, :, start:end, 1] = B_all[i][:, :, 3 :: 7]
            DB[:, :, start:end, 1] -= B_all[i][:, :, 4 :: 7]
            DB[:, :, start:end, 1] /= (2*eps)
            
            # ∂B/∂z
            DB[:, :, start:end, 2] = B_all[i][:, :, 5 :: 7]
            DB[:, :, start:end, 2] -= B_all[i][:, :, 6 :: 7]
            DB[:, :, start:end, 2] /= (2*eps)

            for j in range(n_path):
                mom_rot = sens._orientation[j].apply(mom)
                MOM[:, j, start:end] = np.broadcast_to(mom_rot, (n_src, end-start,3))

                if pivot is not None:
                    pts = sens._orientation[j].apply(sens.pixel) + sens._position[j]
                    pos_piv = pts - pivot[i,j]
                    POS_PIV[:, j, start:end] = np.broadcast_to(pos_piv, (n_src, end-start, 3))

        # Force and Torque computation
        #  - !!Performance!!: cross (torque) few time faster than einsum (force) few
        #    times faster than array API force. Also, the torque computation relies on
        #    the force computation. Therefore it makes no sense to separate the force
        #    and torque computation.

        print(DB)

        F = np.einsum('abijk,abik->abij', DB, MOM)            # numpy only
        #F = np.sum(DB * MOM[:, :, :, np.newaxis, :], axis=4) # array API
        T = np.cross(MOM, B)

        if pivot is not None:
            T += np.cross(POS_PIV, F)

        # Reduce mesh to targets (sumover) -> (n_src, n_path, n_tgt, 3)
        F = np.add.reduceat(F, idx_starts, axis=2)
        T = np.add.reduceat(T, idx_starts, axis=2)

        # Broadcast into output array
        FTOUT[0][:,:,mask_magnet,:] = F
        FTOUT[1][:,:,mask_magnet,:] = T


    # CURRENT COMPUTATION #########################################################
    if n_current > 0:

        # Computation array allocation
        #  - use broadcasting instead of concatenate (better memory and speed)
        mesh_sizes = mesh_sizes_all[mask_current]
        n_mesh = sum(mesh_sizes)

        B = np.empty((n_src, n_path, n_mesh, 3))     # B-field at cell location
        TVEC = np.empty((n_src, n_path, n_mesh, 3))  # current tangential vectors
        CURR = np.empty((n_src, n_path, n_mesh))     # current
        if pivot is not None:
            POS_PIV = np.empty((n_src, n_path, n_mesh, 3))  # relative cell pos to pivot
        
        # Broadcasting and application of pos & ori
        idx_all = np.where(mask_current)[0]
        idx_ends = np.cumsum(mesh_sizes)
        idx_starts = np.r_[0, idx_ends[:-1]]
        for start, end, i in zip(idx_starts, idx_ends, idx_all):
            sens = meshes[i]["pts"]
            tvec = meshes[i]["tvecs"]
            curr = meshes[i]["currents"]

            B[:, :, start:end] = B_all[i]
            CURR[:,:,start:end] = np.broadcast_to(curr, (n_src, n_path, end-start))

            for j in range(n_path):
                tvec_rot = sens._orientation[j].apply(tvec)
                TVEC[:, j, start:end] = np.broadcast_to(tvec_rot, (n_src, end-start, 3))
            
                if pivot is not None:
                    pts = sens._orientation[j].apply(sens.pixel) + sens._position[j]
                    pos_piv = pts - pivot[i,j] #pivot shape (n_tgt, n_path, 3)
                    POS_PIV[:, j, start:end] = np.broadcast_to(pos_piv, (n_src, end-start, 3))

        # Force and Torque computation
        F = CURR[...,np.newaxis] * np.cross(TVEC, B)

        if pivot is None:
            T = np.zeros_like(F)
        else:
            T = np.cross(POS_PIV, F)

        # Reduce mesh to targets (sumover) -> (n_src, n_path, n_tgt, 3)
        F = np.add.reduceat(F, idx_starts, axis=2)
        T = np.add.reduceat(T, idx_starts, axis=2)

        # Broadcast into output array
        FTOUT[0][:,:,mask_current,:] = F
        FTOUT[1][:,:,mask_current,:] = T

    # FINALIZE OUTPUT ##############################################################
    
    # Do pivot calc here for all targets ?!?
    
    # Sum up Target Collections
    FTOUT = np.add.reduceat(FTOUT, coll_idx, axis=3)

    if squeeze:
        return np.squeeze(FTOUT)

    return FTOUT



if __name__ == "__main__":
    import numpy as np
    import magpylib as magpy
    from scipy.spatial.transform import Rotation as R

    def FTana(p1, p2, m1, m2, src, piv):
        # analytical force solution
        r = p2 - p1
        r_abs = np.linalg.norm(r)
        r_unit = r / r_abs
        F_ana = (
            3
            * magpy.mu_0
            / (4 * np.pi * r_abs**4)
            * (
                np.cross(np.cross(r_unit, m1), m2)
                + np.cross(np.cross(r_unit, m2), m1)
                - 2 * r_unit * np.dot(m1, m2)
                + 5 * r_unit * (np.dot(np.cross(r_unit, m1), np.cross(r_unit, m2)))
            )
        )
        B = src.getB(p2)
        T_ana = np.cross(m2, B) + np.cross(p2-piv, F_ana)
        return F_ana, T_ana


    m1, m2 = np.array((0.976, 4.304, 2.055))*1e3, np.array((0.878, -1.527, 2.918))*1e3
    p1, p2 = np.array((-1.248, 7.835, 9.273)), np.array((-2.331, 5.835, 0.578))
    piv = np.array((0.727, 5.152, 5.363))  # pivot point for torque calculation

    m1, m2 = np.array([(1e3,0,0), (0,1e3,0)])
    p1, p2 = np.array([(0,0,0), (1,0,0)])
    piv = np.array((0.1, 0, 0))

    rot = R.from_rotvec((0, 0, 10), degrees=True)
    m2_rot = rot.apply(m2)

    # magpylib
    src = magpy.misc.Dipole(position=p1, moment=m1)
    tgt1 = magpy.misc.Dipole(position=p2, moment=m2).rotate(rot)
    tgt2 = magpy.misc.Dipole(position=p2, moment=m2_rot)

    F1, T1 = magpy.getFT(src, tgt1, pivot=piv)
    F2, T2 = magpy.getFT(src, tgt2, pivot=piv)
    F3, T3 = FTana(p1, p2, m1, m2_rot, src, piv)

    print(F1)
    print(F2)
    print(F3)
    
    
    
    # import magpylib as magpy

    
    # loop = magpy.current.Circle(
    #     diameter=5,
    #     current=1e6,
    #     position=(0,0,0),
    #     meshing=2400*4,
    # )
    # dip = magpy.misc.Dipole(
    #     moment=(1e3,0,0),
    #     position=(0,0,0)
    # )
    # F1,T1 = getFT(dip, loop, pivot=(0,0,0))
    # print(T1)
    # F2,T2 = getFT(loop, dip, pivot=(0,0,0))
    # print(T2)

    # loop.rotate_from_angax(45, 'x')

    # F1,T1 = getFT(dip, loop, pivot=(0,0,0))
    # print(T1)
    # F2,T2 = getFT(loop, dip, pivot=(0,0,0))
    # print(T2)

    # loop.orientation=None
    # loop.rotate_from_angax(45, 'y')

    # F1,T1 = getFT(dip, loop, pivot=(0,0,0))
    # print(T1)
    # F2,T2 = getFT(loop, dip, pivot=(0,0,0))
    # print(T2)

    # loop.orientation=None
    # dip.rotate_from_angax(-45, 'y')

    # F1,T1 = getFT(dip, loop, pivot=(0,0,0))
    # print(T1)
    # F2,T2 = getFT(loop, dip, pivot=(0,0,0))
    # print(T2)

    # from scipy.spatial.transform import Rotation as R
    # loop.orientation=R.from_rotvec([22,-55,111], degrees=True)
    # dip.orientation=R.from_rotvec([123,-321,-11], degrees=True)
    # loop.position = (1, 2, 3)
    # dip.position = (-1, -.4, 2)

    # F1,T1 = getFT(dip, loop, pivot=(0,0,0))
    # print(T1)
    # F2,T2 = getFT(loop, dip, pivot=(0,0,0), eps=1e-7)
    # print(T2)
