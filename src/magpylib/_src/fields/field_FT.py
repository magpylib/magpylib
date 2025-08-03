from itertools import compress

import numpy as np
import warnings
from magpylib._src.fields.field_wrap_BH import getB
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_dimensions

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


def check_format_input_pivot(pivot, targets, path_length):
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
        return [np.pad(t._centroid, ) for t in targets]

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
    # Input checks
    targets, coll_idx = check_format_input_targets(targets)
    #pivot = check_format_input_pivot(pivot, targets) # PIVOT MUST ALSO BE TILE UP !!!!!!! <<<------
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

    # PATH TILING ########################################################################
    tgt_path_lengths = [len(tgt._position) for tgt in targets]
    src_path_lengths = [len(src._position) for src in sources]
    max_path_length = max(tgt_path_lengths + src_path_lengths)

    # Allocate and broadcast into
    tgt_paths = np.zeros((n_targets, max_path_length, 3))

    for i,tgt in enumerate(targets):
        padlength = max_path_length - len(tgt._position)
        tgt_paths[i] = np.pad(tgt._position, ((0,padlength), (0,0)), 'edge')

    print(tgt_paths)
    import sys
    sys.exit()

    # RUN MESHING FUNCTIONS ##############################################################
    # Collect meshing function results - cannot separate mesh generation from generation 
    #    of moments, lvecs, etc.
    # Meshing functions are run now to collect all observers because the B-field is
    #    computed for all targets at once, for efficiency reasons.
    meshes = []
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

        meshes.append(mesh)
        mesh_sizes.append(len(mesh))

    if meshreport:
        print("Mesh report:")
        for t, m in zip(targets, mesh_sizes):
            print(f"  Target {t}: {m} points")
        print()


    # apply orientations and positions of paths to mesh
    # tile up everything ?
    # ...
    # ...

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



if __name__ == "__main__":
    # Test the getFT function with some dummy data
    from magpylib._src.obj_classes.class_magnet_Tetrahedron import Tetrahedron
    from magpylib._src.obj_classes.class_current_Polyline import Polyline

    src = Tetrahedron(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], magnetization=(1e6, 0, 0))
    tgt1 = Polyline(vertices=[(2, 2, 2), (3, 3, 3)], current=1.0, meshing=10)
    tgt1.position = [(1,1,1)] * 4
    
    tgt2 = Polyline(vertices=[(2, 2, 2), (3, 3, 3)], current=1.0, meshing=10)

    targets = [tgt1, tgt2]
    #print([t.centroid for t in targets])

    getFT(src, targets)