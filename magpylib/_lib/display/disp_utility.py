""" Display function codes"""

import numpy as np
from numpy import sin,cos

def faces_box(src, show_path):
    """
    compute vertices and faces of Box input for plotting
    takes Box source
    returns vert, faces
    returns all faces when show_path=all
    """
    # pylint: disable=protected-access
    a,b,c = src.dim
    vert0 = np.array(((0,0,0),(a,0,0),(0,b,0),(0,0,c),
                     (a,b,0),(a,0,c),(0,b,c),(a,b,c)))
    vert0 = vert0 - src.dim/2

    if not isinstance(show_path, bool) and src._pos.ndim>1:
        rots = src._rot[::-show_path]
        poss = src._pos[::-show_path]
    else:
        rots = [src._rot[-1]]
        poss = [src._pos[-1]]

    faces = []
    for rot,pos in zip(rots,poss):
        vert = rot.apply(vert0) + pos
        faces += [
            [vert[0],vert[1],vert[4],vert[2]],
            [vert[0],vert[1],vert[5],vert[3]],
            [vert[0],vert[2],vert[6],vert[3]],
            [vert[7],vert[6],vert[2],vert[4]],
            [vert[7],vert[6],vert[3],vert[5]],
            [vert[7],vert[5],vert[1],vert[4]],
            ]
    return faces


def faces_cylinder(src, show_path):
    """
    Compute vertices and faces of Sphere input for plotting.

    Parameters
    ----------
    - src (source object)
    - show_path (bool or int)

    Returns
    -------
    vert, faces (returns all faces when show_path=int)
    """
    # pylint: disable=protected-access
    res = 15 #surface discretization

    # generate cylinder faces
    r,h2 = src.dim/2
    hs = np.array([-h2,h2])
    phis = np.linspace(0,2*np.pi,res)
    phis2 = np.roll(np.linspace(0,2*np.pi,res),1)
    faces = [np.array([
        (r*np.cos(p1), r*np.sin(p1),  h2),
        (r*np.cos(p1), r*np.sin(p1), -h2),
        (r*np.cos(p2), r*np.sin(p2), -h2),
        (r*np.cos(p2), r*np.sin(p2),  h2)])
        for p1,p2 in zip(phis,phis2)]
    faces += [np.array([
        (r*np.cos(phi), r*np.sin(phi), h)
         for phi in phis]) for h in hs]

    # add src attributes position and orientation depending on show_path
    if not isinstance(show_path, bool) and src._pos.ndim>1:
        rots = src._rot[::-show_path]
        poss = src._pos[::-show_path]
    else:
        rots = [src._rot[-1]]
        poss = [src._pos[-1]]

    # all faces (incl. along path) adding pos and rot
    all_faces = []
    for rot,pos in zip(rots,poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def faces_sphere(src, show_path):
    """
    Compute vertices and faces of Sphere input for plotting.

    Parameters
    ----------
    - src (source object)
    - show_path (bool or int)

    Returns
    -------
    vert, faces (returns all faces when show_path=int)
    """
    # pylint: disable=protected-access
    res = 15 #surface discretization

    # generate sphere faces
    r = src.dim/2
    phis = np.linspace(0,2*np.pi,res)
    phis2 = np.roll(np.linspace(0,2*np.pi,res),1)
    ths = np.linspace(0,np.pi,res)
    faces = [r*np.array([
            (cos(p)*sin(t1), sin(p)*sin(t1), cos(t1)),
            (cos(p)*sin(t2), sin(p)*sin(t2), cos(t2)),
            (cos(p2)*sin(t2), sin(p2)*sin(t2), cos(t2)),
            (cos(p2)*sin(t1), sin(p2)*sin(t1), cos(t1))])
            for p,p2 in zip(phis,phis2) for t1,t2 in zip(ths[1:-2],ths[2:-1])]
    faces += [r*np.array([
            (cos(p)*sin(th), sin(p)*sin(th), cos(th))
            for p in phis]) for th in [ths[1],ths[-2]]]

    # add src attributes position and orientation depending on show_path
    if not isinstance(show_path, bool) and src._pos.ndim>1:
        rots = src._rot[::-show_path]
        poss = src._pos[::-show_path]
    else:
        rots = [src._rot[-1]]
        poss = [src._pos[-1]]

    # all faces (incl. along path) adding pos and rot
    all_faces = []
    for rot,pos in zip(rots,poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def system_size(face_points, pix_points, dipole_points, markers, path_points):
    """compute system size for display
    """
    # limits of current axis with drawn sensors and paths

    # collect all vertices (collection faces do not reset ax limits)
    pts = []

    for face in face_points:
        pts += list(face)

    pts += dipole_points

    if len(markers)>0:
        pts += list(markers)

    if len(pix_points)>0:
        pts += pix_points

    if len(path_points)>0:
        pts += path_points

    # determine min/max from all to generate aspect=1 plot
    pts = np.array(pts)
    xs = [np.amin(pts[:,0]),np.amax(pts[:,0])]
    ys = [np.amin(pts[:,1]),np.amax(pts[:,1])]
    zs = [np.amin(pts[:,2]),np.amax(pts[:,2])]

    xsize = xs[1]-xs[0]
    ysize = ys[1]-ys[0]
    zsize = zs[1]-zs[0]

    xcenter = (xs[1]+xs[0])/2
    ycenter = (ys[1]+ys[0])/2
    zcenter = (zs[1]+zs[0])/2

    size = max([xsize,ysize,zsize])

    limx0 = xcenter + size/2
    limx1 = xcenter - size/2
    limy0 = ycenter + size/2
    limy1 = ycenter - size/2
    limz0 = zcenter + size/2
    limz1 = zcenter - size/2

    return limx0, limx1, limy0, limy1, limz0, limz1
