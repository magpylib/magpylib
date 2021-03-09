""" Display function codes"""

import numpy as np


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

    rots = [src._rot[-1]]
    poss = [src._pos[-1]]
    if show_path == 'all':
        rots = src._rot
        poss = src._pos

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
    compute vertices and faces of cylinder input for plotting
    takes Cylinder source,
    returns vert, faces
    returns all faces when show_path=all
    """
    # pylint: disable=protected-access
    res = 20 # resolution
    d,h = src.dim
    phis = np.linspace(0,2*np.pi,res)
    vert_circ = np.array([np.cos(phis),np.sin(phis),np.zeros(res)]).T*d/2
    v_t0 = vert_circ + np.array([0,0,h/2]) # top vertices
    v_b0 = vert_circ - np.array([0,0,h/2]) # bott vertices

    rots = [src._rot[-1]]
    poss = [src._pos[-1]]
    if show_path == 'all':
        rots = src._rot
        poss = src._pos

    faces = []
    for rot,pos in zip(rots,poss):
        v_t = rot.apply(v_t0) + pos
        v_b = rot.apply(v_b0) + pos
        faces += [[v_t[i], v_b[i], v_b[i+1], v_t[i+1]] for i in range(res-1)]
        faces += [v_t, v_b]
    return faces


def system_size(face_points, pix_points, markers, path_points):
    """compute system size for display
    """
    # limits of current axis with drawn sensors and paths

    # collect all vertices (collection faces do not reset ax limits)
    pts = []
    for face in face_points:
        pts += list(face)

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
