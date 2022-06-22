"""
Implementation for the magnetic field of homogeneously
magnetized tetrahedra. Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input
from magpylib._src.fields.field_BH_facet import facet_field

#############
#help function
#############
def check_chirality(points):
    """
    Checks if quartupel of points (p0,p1,p2,p3) that forms tetrahedron is arranged in a way
    that the vectors p0p1, p0p2, p0p3 forms a right-handed system

    Parameters
    -----------
    points: 3d-array of shape (m x 4 x 3)
            m...number of tetrahedrons

    Returns
    ----------
    new list of points, where p2 and p3 are possibly exchanged so that all
    tetrahedron is given in a right-handed system
    """

    vecs = np.zeros((points.shape[0],3,3))
    vecs[:,:,0] = points[:,1,:] - points[:,0,:]
    vecs[:,:,1] = points[:,2,:] - points[:,0,:]
    vecs[:,:,2] = points[:,3,:] - points[:,0,:]

    dets = np.linalg.det(vecs)
    dets_neg = dets < 0

    if np.any(dets_neg):

        points[dets_neg,2:,:] = points[dets_neg,3:1:-1,:]

    return points


def point_inside(points, vertices):
    """
    Takes points, as well as the vertices of a tetrahedra.
    Returns boolean array indicating whether the points are inside the tetrahedra.
    """
    mat = np.zeros((vertices.shape[0],3,3))
    mat[:,0,:] = vertices[:,1,:] - vertices[:,0,:]
    mat[:,1,:] = vertices[:,2,:] - vertices[:,0,:]
    mat[:,2,:] = vertices[:,3,:] - vertices[:,0,:]
    tetra = np.linalg.inv(np.transpose(mat, (0,2,1)))

    newp = np.matmul(tetra, np.reshape(points-vertices[:,0,:], (*points.shape,1)))
    return (np.all(newp >= 0, axis=1) &
            np.all(newp <= 1, axis=1) &
            (np.sum(newp, axis=1) <= 1)).flatten()


  #############


def magnet_tetrahedron_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Code for the field calculation of a uniformly magnetized tetrahedron

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of [mT], if `field='H'` return H-field
        in units of [kA/m].

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of [mm].

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of [mT].

    vertices: ndarray, shape (n,4,3)
        Vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4) of tetrahedron

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

    Examples
    --------
    Compute the field of three different instances.
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> mag = np.array([(222,333,555), (33,44,55), (0,0,100)])
    >>> vertices = np.array([[(1,1,1), (2,-3,4), (1,2,3), (-1,-2,-3)],
        [(-4,5,6), (7,8,9), (1,2,3), (-10,-5,0)],
        [(0,1,2), (3,-2,-5), (-5,2,3), (3,6,9)]])
    >>> obs = np.array([(0,0,0), (2,3,4), (-10,-20,-10)])
    >>> B = magpy.core.magnet_tetrahedron_field('B', obs, mag, vertices)
    >>> print(B)
    [[ 2.09797687e+02  3.22359234e+02  5.54097897e+02]
    [ 3.35521579e+01  4.26144490e+01  5.27267114e+01]
    [ 3.03382140e-03  6.47059217e-03 -2.50174038e-03]]

    Notes
    -----
    The tetrahedron is built up via 4 facets.
    """

    bh = check_field_input(field, "magnet_tetrahedron_field()")

    n = observers.shape[0]

    vertices = check_chirality(vertices)
    facets_vertices = np.concatenate(
                        (vertices[:,[0,2,1],:], vertices[:,[0,1,3],:],
                        vertices[:,[1,2,3],:], vertices[:,[0,3,2],:]),
                        axis=0
                    )
    facets_fields = facet_field(
                        field,
                        np.tile(observers, (4,1)),
                        np.tile(magnetization, (4,1)),
                        facets_vertices
                    )
    field = facets_fields[0:n,:] + facets_fields[n:2*n,:] + \
            facets_fields[2*n:3*n,:] + facets_fields[3*n:4*n,:]

    if bh:
        # if inside magnet add magnetization vector
        mask_inside = point_inside(observers, vertices)
        field[mask_inside] += magnetization[mask_inside]
        return field

    return field
