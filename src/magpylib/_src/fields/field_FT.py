from itertools import compress

import numpy as np
import warnings
from magpylib._src.fields.field_wrap_BH import getB
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_dimensions
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_Sensor import Sensor

def check_format_input_sources(sources):
    """
    Check and format sources input
    """
    if not isinstance(sources, list):
        sources = [sources]
    
    from magpylib._src.obj_classes.class_Collection import Collection
    for s in sources:
        if not hasattr(s, '_field_func') and not isinstance(s, Collection):
            msg = (
                "getFT bad source input. Sources can only be Magpylib source objects."
                f" Instead received type {type(s)} source."
            )
            raise ValueError(msg)
    return sources


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


def check_format_input_pivot(pivot, targets):
    """
    Check and format pivot input

    Returns
    -------
    list of pivots with shape (n_target, ...) can be ragged if some tgts have path and others do not
    """

    msg = (
        "Bad getFT pivot input. Input pivot must be str 'centroid', None or array_like of shape (3,)."
        " It can also be (n,3) when there are n targets providing a different pivot for every target."
    )

    if isinstance(pivot, str) and pivot == "centroid":
        return np.array([t.centroid for t in targets])

    if pivot is None:
        return None

    if isinstance(pivot, (list, tuple, np.ndarray)):
        try:
            pivot = np.array(pivot, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(msg) from e
        
        if pivot.shape == (3,):
            return np.tile(pivot, (len(targets), 1))
        if pivot.shape == (len(targets), 3):
            return pivot

    raise ValueError(msg)


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


def getFT_NOPATH(sources, targets, pivot="centroid", eps=1e-5, squeeze=True, meshreport=False):
    """
    Compute magnetic force and torque acting on the targets that are exposed
    to the magnetic field of the sources.

    SI units are assumed for all inputs and outputs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l sources and/or collection objects.

    targets: single object or 1D list of t objects that are Sphere, Cuboid, Polyline,
        Cylinder, CylinderSegment, or Dipole. Force and Torque acting on targets in the magnetic
        field generated by the sources will be computed. A target (except of Dipole) must have a valid
        `meshing` parameter.

    pivot: str, None, or array_like, default="centroid"
        The Force adds to the Torque via the pivot point. For a freely floating magnet
        this would be the barycenter (= centroid when the density is homogeneous).
        If pivot="centroid" the centroid is selected as the pivot point for all targets.
        If pivot=None no pivot is used. This will give unphysical results.
        If pivot=array_like of shape (3,) the same pivot is used for all targets.
        Alternatively one can provide an individual pivot point for each target.

    eps: float, default=1e-5
        This is only used for magnet targets for computing the magnetic field gradient
        using finite differences (FD). `eps` is the FD step size. A good value
        is 1e-5 * characteristic_system_size (magnet size, distance between sources
        and targets, ...).

    squeeze: bool, default=True
        The output of the computation has the shape (n,3) where n corresponds to the number
        of targets. By default this is reduced to (3,) when there is only one target.
    
    meshreport: bool, default=False
        If True, a report of the mesh used for each target will be printed.

    Returns
    -------
    Force-Torque as ndarray of shape (2,3), or (t,2,3) when t targets are given
    """
    # Input checks
    targets, coll_idx = check_format_input_targets(targets)
    pivot = check_format_input_pivot(pivot, targets)
    sources = check_format_input_sources(sources)

    # Get force types and create masks efficiently
    mask_magnet = np.array([tgt._force_type == "magnet" for tgt in targets])
    mask_current = np.array([tgt._force_type == "current" for tgt in targets])

    # Important numbers
    n_targets = len(targets)
    n_sources = len(sources)
    n_magnets = sum(mask_magnet)  # number of magnet targets
    n_currents = sum(mask_current)  # number of current targets

    # Allocate output arrays
    FTOUT = np.zeros((2, n_sources, n_targets, 3))

    # RUN MESHING FUNCTIONS ##############################################################
    # Collect meshing function results - cannot separate mesh generation from generation 
    #    of moments, lvecs, etc.
    # Meshing functions are run now to collect all observers because the B-field is
    #    computed for all targets at once, for efficiency reasons.
    observer = []
    mesh_sizes = []
    mag_moments = []
    cur_currents = []
    cur_tvecs = []
    eps_vec = create_eps_vector(eps)
    for tgt in targets:

        if tgt._force_type == "magnet":
            mesh, mom = tgt._generate_mesh()
            mag_moments.append(mom)
            # if target is a magnet add 6 finite difference steps for gradient computation
            mesh = (mesh[:, np.newaxis, :] + eps_vec[np.newaxis, :, :]).reshape(-1, 3)

        if tgt._force_type == "current":
            mesh, curr, tvec = tgt._generate_mesh()
            cur_currents.append(curr)
            cur_tvecs.append(tvec)

        observer.append(mesh)
        mesh_sizes.append(len(mesh))


    # COMPUTE B FIELD ###################################################################
    observer = np.concatenate(observer, axis=0)
    B_all = getB(sources, observer, squeeze=False)
    # shape (n_src, 1, 1, n_obs, 3)

    # Indexing of observer and B_all arrays
    mesh_sizes = np.array(mesh_sizes)
    obs_ends = np.cumsum(mesh_sizes)
    obs_starts = np.r_[0, obs_ends[:-1]]

    # COMPUTATION IDEA:
    #   The equations are very simple like F = DB.MOM, T = B x MOM.
    #   All we need to do is to broadcast into large computation arrays that
    #   contain DB, MOM, B, ... and apply the equations, sum over mesh cells,
    #   and reshape the output to the desired shape.

    # MAGNETS ########################################################################
    if n_magnets > 0:
        
        # Prepare index ranges for broadcasting
        mesh_sizes_mag = mesh_sizes[mask_magnet] // 7
        n_mesh_mag = sum(mesh_sizes_mag)
        idx_ends = np.cumsum(mesh_sizes_mag)
        idx_starts = np.r_[0, idx_ends[:-1]]

        # Computation array allocations
        POS = np.zeros((n_mesh_mag*n_sources, 3))   # central location of each cell
        B = np.zeros((n_mesh_mag*n_sources, 3))     # B-field at each cell
        DB = np.zeros((n_mesh_mag*n_sources, 3, 3)) # B-field gradient at each cell
        MOM = np.zeros((n_mesh_mag*n_sources, 3))   # magnetic moment of each cell

        # BROADCASTING into computation arrays:
        #   rule: (src1 mesh1, src1 mesh2, src1 mesh3, ... src2 mesh1, src2 mesh2, ... )
        for i, mom in enumerate(mag_moments):
            # range in observer and B arrays
            start = obs_starts[mask_magnet][i]
            end = obs_ends[mask_magnet][i]
            
            for j in range(n_sources):
                # range in computation arrays
                ids = idx_starts[i] + n_mesh_mag*j
                ide = idx_ends[i] + n_mesh_mag*j

                POS[ids : ide] = observer[start : end : 7]
                B[ids : ide] = B_all[j, 0, 0, start : end : 7]

                # ∂B/∂x
                DB[ids : ide, :, 0] = B_all[j, 0, 0, start+1 : end : 7]
                DB[ids : ide, :, 0] -= B_all[j, 0, 0, start+2 : end : 7]
                DB[ids : ide, :, 0] /= (2*eps)

                # ∂B/∂y
                DB[ids : ide, :, 1] = B_all[j, 0, 0, start+3 : end : 7]
                DB[ids : ide, :, 1] -= B_all[j, 0, 0, start+4 : end : 7]
                DB[ids : ide, :, 1] /= (2*eps)
                
                # ∂B/∂z
                DB[ids : ide, :, 2] = B_all[j, 0, 0, start+5 : end : 7]
                DB[ids : ide, :, 2] -= B_all[j, 0, 0, start+6 : end : 7]
                DB[ids : ide, :, 2] /= (2*eps)

                MOM[ids : ide] = mom

        # ACTUAL FORCE AND TORQUE COMPUTATION
        #   !!Performance!!: cross few time faster than einsum few times faster than array API.
        #   Torque computation relies on force computation.
        #   Therefore it makes no sense to separate the force and torque computation.

        force = np.einsum('ijk,ik->ij', DB, MOM) # numpy only
        #force = np.sum(DB * MOM[:, np.newaxis, :], axis=2) # array API
        torque = np.cross(MOM, B)

        # Add pivot point contribution to torque
        if pivot is not None:
            PIV = np.tile(np.repeat(pivot[mask_magnet], mesh_sizes_mag, axis=0), (n_sources, 1))
            torque += np.cross(POS - PIV, force)

        # Sum over mesh cells
        idx_starts_all = np.concatenate([idx_starts + n_mesh_mag*j for j in range(n_sources)])
        F = np.add.reduceat(force, idx_starts_all, axis=0)
        T = np.add.reduceat(torque, idx_starts_all, axis=0)

        # Broadcast into output arrays
        FTOUT[0, :, mask_magnet] = F.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)
        FTOUT[1, :, mask_magnet] = T.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)


    # CURRENTS ########################################################################
    if n_currents > 0:
        # Prepare index ranges for broadcasting
        mesh_sizes_cur = mesh_sizes[mask_current]
        n_mesh_cur = sum(mesh_sizes_cur)
        idx_ends = np.cumsum(mesh_sizes_cur)
        idx_starts = np.r_[0, idx_ends[:-1]]

        # Computation array allocations
        POS = np.zeros((n_mesh_cur*n_sources, 3))  # central location of each cell
        B = np.zeros((n_mesh_cur*n_sources, 3))    # B-field at POS
        TVEC = np.zeros((n_mesh_cur*n_sources, 3)) # current path tangential vectors
        CURR = np.zeros((n_mesh_cur*n_sources,))   # current

        # BROADCASTING into computation arrays:
        #   rule: (src1 mesh1, src1 mesh2, src1 mesh3, ... src2 mesh1, src2 mesh2, ... )
        for i, (curr, tvec) in enumerate(zip(cur_currents, cur_tvecs)):
            # range in observer and B arrays
            start = obs_starts[mask_current][i]
            end = obs_ends[mask_current][i]

            for j in range(n_sources):
                # range in computation arrays
                ids = idx_starts[i] + n_mesh_cur*j
                ide = idx_ends[i] + n_mesh_cur*j

                POS[ids : ide] = observer[start : end]
                B[ids : ide] = B_all[j, 0, 0, start : end]
                TVEC[ids : ide] = tvec
                CURR[ids : ide] = curr

        # ACTUAL FORCE AND TORQUE COMPUTATION
        force = (CURR * np.cross(TVEC, B).T).T
        torque = np.zeros_like(force)

        # Add pivot point contribution to torque
        if pivot is not None:
            PIV = np.tile(np.repeat(pivot[mask_current], mesh_sizes_cur, axis=0), (n_sources, 1))
            torque += np.cross(POS - PIV, force)

        # Sum over mesh cells
        idx_starts_all = np.concatenate([idx_starts + n_mesh_cur*j for j in range(n_sources)])
        F = np.add.reduceat(force, idx_starts_all, axis=0)
        T = np.add.reduceat(torque, idx_starts_all, axis=0)

        # Broadcast into output arrays
        FTOUT[0, :, mask_current] = F.reshape(n_sources, n_currents, 3).transpose(1, 0, 2)
        FTOUT[1, :, mask_current] = T.reshape(n_sources, n_currents, 3).transpose(1, 0, 2)

    # FINALIZE OUTPUT ##############################################################

    # Sum up Collections
    FTOUT = np.add.reduceat(FTOUT, coll_idx, axis=2)

    if squeeze:
        return np.squeeze(FTOUT)

    return FTOUT

################################################################################
################################################################################
################################################################################
################################################################################

def check_format_input_pivot2(pivot, targets, n_path):
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

    targets: single object or 1D list of t objects that are Sphere, Cuboid, Polyline,
        Cylinder, CylinderSegment, or Dipole. Force and Torque acting on targets in the magnetic
        field generated by the sources will be computed. A target (except of Dipole) must have a valid
        `meshing` parameter.

    pivot: str, None, or array_like, default="centroid"
        The Force adds to the Torque via the pivot point. For a freely floating magnet
        this would be the barycenter (= centroid when the density is homogeneous).
        If pivot="centroid" the centroid is selected as the pivot point for all targets.
        If pivot=None no pivot is used. This will give unphysical results.
        If pivot=array_like of shape (3,) the same pivot is used for all targets.
        Alternatively one can provide an individual pivot point for each target.

    eps: float, default=1e-5
        This is only used for magnet targets for computing the magnetic field gradient
        using finite differences (FD). `eps` is the FD step size. A good value
        is 1e-5 * characteristic_system_size (magnet size, distance between sources
        and targets, ...).

    squeeze: bool, default=True
        The output of the computation has the shape (n,3) where n corresponds to the number
        of targets. By default this is reduced to (3,) when there is only one target.
    
    meshreport: bool, default=False
        If True, a report of the mesh used for each target will be printed.

    Returns
    -------
    Force-Torque as ndarray of shape (2,3), or (t,2,3) when t targets are given
    """
    # Input check targets
    targets, coll_idx = check_format_input_targets(targets)
    n_tgt = len(targets)
    
    # Input checks sources
    sources = check_format_input_sources(sources)
    n_src = len(sources)

    # Collect and tile up TARGET PATHS
    tgt_path_lengths = [len(tgt._position) for tgt in targets]
    src_path_lengths = [len(src._position) for src in sources]
    n_path = max(tgt_path_lengths + src_path_lengths)

    #tgt_poss = np.zeros((n_tgt, n_path, 3)) # shape (n_tgt, path_length, 3)
    tgt_oris = np.zeros((n_tgt, n_path, 4)) # shape (n_tgt, path_length, 4)
    

    for i,tgt in enumerate(targets):
        padlength = n_path - len(tgt._position)
        
        #pos = tgt._position
        #tgt_poss[i] = np.pad(pos, ((0,padlength), (0,0)), 'edge')

        ori = np.atleast_2d(tgt.orientation.as_quat())
        tgt_oris[i] = np.pad(ori, ((0,padlength), (0,0)), 'edge')    

    # Collect and tile up PIVOTS
    pivot = check_format_input_pivot2(pivot, targets, n_path) # shape (n_targets, path_length, 3)

    # Force type masks - 2 cases: magnets and currents
    mask_magnet = np.array([tgt._force_type == "magnet" for tgt in targets])
    mask_current = np.array([tgt._force_type == "current" for tgt in targets])

    n_magnets = sum(mask_magnet)  # number of magnet targets
    n_currents = sum(mask_current)  # number of current targets

    # Allocate output arrays - use n_tgt as first dim for masking
    FTOUT = np.zeros((n_tgt, 2, n_src, n_path, 3))


    # RUN MESHING FUNCTIONS ##############################################################
    # Collect all outputs - cannot meaningfuly separate mesh generation from
    #    generation of moments, lvecs, etc.
    # Meshing functions are run at this point to collect all observers for getB
    #    which can be computed for all targets at once.
    sensors = []
    mesh_sizes_all = np.zeros(n_tgt, dtype=int)
    mag_moments = []
    cur_currents = []
    cur_tvecs = []
    eps_vec = create_eps_vector(eps)
    for i,tgt in enumerate(targets):

        if tgt._force_type == "magnet":
            mesh, mom = tgt._generate_mesh()
            # if target is a magnet add 6 finite difference steps for gradient computation
            mesh = (mesh[:, np.newaxis, :] + eps_vec[np.newaxis, :, :]).reshape(-1, 3)
            mag_moments.append(mom)

        if tgt._force_type == "current":
            mesh, curr, tvec = tgt._generate_mesh()

            n_mesh = len(mesh)
            
            # Tile with n_path
            curr = np.tile(curr, n_path)
            tvec = np.tile(tvec, (n_path, 1))
                           
            # Apply path-tiled rotations
            rot_ext = R.from_quat(np.repeat(tgt_oris[i], n_mesh, axis=0))
            tvec = rot_ext.apply(tvec)

            cur_currents.append(curr.reshape(n_path, n_mesh))
            cur_tvecs.append(tvec.reshape(n_path, n_mesh, 3))

        sensors.append(Sensor(
                    pixel=mesh,
                    position=tgt._position,
                    orientation=R.from_quat(tgt_oris[i]),
                ))
        mesh_sizes_all[i] = n_mesh

    # Meshreport
    if meshreport:
        print("Mesh report:")
        for t, m in zip(targets, mesh_sizes_all):
            print(f"  Target {t}: {m} points")
        print()


    # COMPUTE B FIELD ###################################################################

    # Problem: meshes of multiple targets can have different sizes, but ragged
    #     inputs are not supported by getB.
    # Solution1: allow ragged inputs in getB (oo)
    # Solution2: compute instance by instance (functional)

    if len(set(mesh_sizes_all)) == 1:
        B_all = getB(sources, sensors, squeeze=False)
        B_all = np.moveaxis(B_all, 2, 0)
    else: # ragged inputs - eval mesh by mesh
        B_all = [np.squeeze(getB(sources, s, squeeze=False), axis=2) for s in sensors]

    # shapes: [n_tgt (list dim in case 2), n_src, n_path, n_mesh, 3]

    #mesh_ends = np.cumsum(mesh_sizes)
    #mesh_starts = np.r_[0, mesh_ends[:-1]]

    # COMPUTATION IDEA:
    #   The equations are very simple like F = DB.MOM, T = B x MOM.
    #   All we need to do is to broadcast into large computation arrays that
    #   contain DB, MOM, B, ... and apply the equations, sum over mesh cells,
    #   and reshape the output to the desired shape.

    # MAGNETS ########################################################################
    if n_magnets > 0:
        
        # Prepare index ranges for broadcasting
        mesh_sizes_mag = mesh_sizes[mask_magnet] // 7
        n_mesh_mag = sum(mesh_sizes_mag)
        idx_ends = np.cumsum(mesh_sizes_mag)
        idx_starts = np.r_[0, idx_ends[:-1]]

        # Computation array allocations
        POS = np.zeros((n_mesh_mag*n_src, 3))   # central location of each cell
        B = np.zeros((n_mesh_mag*n_src, 3))     # B-field at each cell
        DB = np.zeros((n_mesh_mag*n_src, 3, 3)) # B-field gradient at each cell
        MOM = np.zeros((n_mesh_mag*n_src, 3))   # magnetic moment of each cell

        # BROADCASTING into computation arrays:
        #   rule: (src1 mesh1, src1 mesh2, src1 mesh3, ... src2 mesh1, src2 mesh2, ... )
        for i, mom in enumerate(mag_moments):
            # range in observer and B arrays
            start = obs_starts[mask_magnet][i]
            end = obs_ends[mask_magnet][i]
            
            for j in range(n_sources):
                # range in computation arrays
                ids = idx_starts[i] + n_mesh_mag*j
                ide = idx_ends[i] + n_mesh_mag*j

                POS[ids : ide] = observer[start : end : 7]
                B[ids : ide] = B_all[j, 0, 0, start : end : 7]

                # ∂B/∂x
                DB[ids : ide, :, 0] = B_all[j, 0, 0, start+1 : end : 7]
                DB[ids : ide, :, 0] -= B_all[j, 0, 0, start+2 : end : 7]
                DB[ids : ide, :, 0] /= (2*eps)

                # ∂B/∂y
                DB[ids : ide, :, 1] = B_all[j, 0, 0, start+3 : end : 7]
                DB[ids : ide, :, 1] -= B_all[j, 0, 0, start+4 : end : 7]
                DB[ids : ide, :, 1] /= (2*eps)
                
                # ∂B/∂z
                DB[ids : ide, :, 2] = B_all[j, 0, 0, start+5 : end : 7]
                DB[ids : ide, :, 2] -= B_all[j, 0, 0, start+6 : end : 7]
                DB[ids : ide, :, 2] /= (2*eps)

                MOM[ids : ide] = mom

        # ACTUAL FORCE AND TORQUE COMPUTATION
        #   !!Performance!!: cross few time faster than einsum few times faster than array API.
        #   Torque computation relies on force computation.
        #   Therefore it makes no sense to separate the force and torque computation.

        force = np.einsum('ijk,ik->ij', DB, MOM) # numpy only
        #force = np.sum(DB * MOM[:, np.newaxis, :], axis=2) # array API
        torque = np.cross(MOM, B)

        # Add pivot point contribution to torque
        if pivot is not None:
            PIV = np.tile(np.repeat(pivot[mask_magnet], mesh_sizes_mag, axis=0), (n_sources, 1))
            torque += np.cross(POS - PIV, force)

        # Sum over mesh cells
        idx_starts_all = np.concatenate([idx_starts + n_mesh_mag*j for j in range(n_sources)])
        F = np.add.reduceat(force, idx_starts_all, axis=0)
        T = np.add.reduceat(torque, idx_starts_all, axis=0)

        # Broadcast into output arrays
        FTOUT[0, :, mask_magnet] = F.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)
        FTOUT[1, :, mask_magnet] = T.reshape(n_sources, n_magnets, 3).transpose(1, 0, 2)


    # CURRENTS ########################################################################
    if n_currents > 0:
        
        # Prepare index ranges for broadcasting
        mesh_sizes = mesh_sizes_all[mask_current]
        n_mesh = sum(mesh_sizes)
        idx_ends = np.cumsum(mesh_sizes)
        idx_starts = np.r_[0, idx_ends[:-1]]
        idx_bs = np.cumsum(mask_current, dtype=int)-1 # index of current tgt in B_all
        
        # Use broadcasting instead of concatenate (better memory and speed)
        B = np.empty((n_src, n_path, n_mesh, 3))  # B-field at cell location
        TVEC = np.empty((n_src, n_path, n_mesh, 3))  # current path tangential vectors       
        CURR = np.empty((n_src, n_path, n_mesh))     # current
        if pivot is not None:
            POS = np.empty((n_src, n_path, n_mesh, 3))  # central location of each cell
            PIV = np.empty((n_src, n_mesh, n_path, 3))  # pivot points

        # loop over targets that are currents
        for i in range(n_currents):
            start = idx_starts[i]
            end = idx_ends[i]
            ib = idx_bs[i]  
            tvec = cur_tvecs[i]    # tangential vectors of this tgt
            curr = cur_currents[i] # current of this tgt

            B[:,:,start:end] = B_all[ib]
            TVEC[:,:,start:end] = np.broadcast_to(tvec, (n_src, n_path, end-start, 3))
            CURR[:,:,start:end] = np.broadcast_to(curr, (n_src, n_path, end-start))

            if pivot is not None:
                sens = sensors[ib]
                for j in range(n_path):
                    mesh = sens._orientation[j].apply(sens.pixel) + sens._position[j]
                    POS[:, j, start:end] = np.broadcast_to(mesh, (n_src, end-start, 3))
                piv = pivot[mask_current][i]
                PIV[:, start:end] = np.broadcast_to(piv, (n_src, end-start, n_path, 3))
                # subtract piv directly from POS to avoid creating another array

        # Reshape arrays for 2D computation
        B = B.reshape(-1,3)
        TVEC = TVEC.reshape(-1,3)
        CURR = CURR.reshape(-1)

        # ACTUAL FORCE AND TORQUE COMPUTATION
        F = (CURR * np.cross(TVEC, B).T).T
        T = np.zeros_like(F)

        # Add pivot point contribution to torque
        if pivot is not None:
            PIV = PIV.swapaxes(1, 2) # shape (n_src, n_path, n_mesh, 3)
            POS = POS.reshape(-1, 3)
            PIV = PIV.reshape(-1, 3)
            T += np.cross(POS - PIV, F)

        F = F.reshape(n_src, n_path, n_mesh, 3)
        T = T.reshape(n_src, n_path, n_mesh, 3)

        # Sum over mesh cells of each target
        F = np.add.reduceat(F, idx_starts, axis=2)
        T = np.add.reduceat(T, idx_starts, axis=2)

        FTOUT[mask_current, 0] = np.moveaxis(F, 2, 0)
        FTOUT[mask_current, 1] = np.moveaxis(T, 2, 0)

    # FINALIZE OUTPUT ##############################################################

    FTOUT = np.moveaxis(FTOUT, 0, 3)

    # Sum up Collections
    FTOUT = np.add.reduceat(FTOUT, coll_idx, axis=3)

    if squeeze:
        return np.squeeze(FTOUT)

    return FTOUT

