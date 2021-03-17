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

    if not isinstance(show_path, bool) and src._pos.ndim>1:
        rots = src._rot[::-show_path]
        poss = src._pos[::-show_path]
    else:
        rots = [src._rot[-1]]
        poss = [src._pos[-1]]

    faces = []
    for rot,pos in zip(rots,poss):
        v_t = rot.apply(v_t0) + pos
        v_b = rot.apply(v_b0) + pos
        faces += [[v_t[i], v_b[i], v_b[i+1], v_t[i+1]] for i in range(res-1)]
        faces += [v_t, v_b]
    return faces


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

    # generate faces
    r = src.dim/2
    Ns = [4,8,16,16,16,16,16,16,16,8,4]
    us = np.linspace(np.pi/2,-np.pi/2,len(Ns)+2)[1:-1]

    t4 = np.linspace(0,2*np.pi,5)
    u0,u1 = us[0], us[-1]
    f_top = np.array([(r*cos(t)*cos(u0), r*sin(t)*cos(u0), r*sin(u0)) for t in t4])
    f_bot = np.array([(r*cos(t)*cos(u1), r*sin(t)*cos(u1), r*sin(u1)) for t in t4])
    faces = [f_top,f_bot]

    shift = 1
    for i,N in enumerate(Ns[:-1]):
        if Ns[i+1]>N:
            faces += faces_sphere_add_incr(r,us[i],us[i+1],N)
        elif Ns[i+1]==N:
            faces += faces_sphere_add_keep(r,us[i],us[i+1],N,shift)
            shift *= -1
        else:
            faces += faces_sphere_add_incr(r,us[i+1],us[i],Ns[i+1])

    # add src attributes position and orientation depending on show_path
    if not isinstance(show_path, bool) and src._pos.ndim>1:
        rots = src._rot[::-show_path]
        poss = src._pos[::-show_path]
    else:
        rots = [src._rot[-1]]
        poss = [src._pos[-1]]

    all_faces = []
    for rot,pos in zip(rots,poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def faces_sphere_add_incr(r,u1,u2,N):
    """
    add a row of faces to sphere surface with increasing discretization

    Parameters
    ---------
    - r (float): radius
    - u1 (float): first polar angle
    - u2 (float): second polar angle
    - N (int): discretization

    Returns
    -------
    list of faces that are ndarray (N,3)
    """
    t1 = np.linspace(0,4*np.pi,2*N+1)
    t2 = np.linspace(0,4*np.pi,4*N+1)+np.pi/N
    v1 = r*np.array([(np.cos(t)*np.cos(u1), np.sin(t)*np.cos(u1), np.sin(u1)) for t in t1])
    v2 = r*np.array([(np.cos(t)*np.cos(u2), np.sin(t)*np.cos(u2), np.sin(u2)) for t in t2])
    faces = []
    for i in range(N):
        faces += [np.array([v1[i], v1[i+1], v2[2*i]])]
        faces += [np.array([v1[i+1], v2[2*i], v2[2*i+1]])]
        faces += [np.array([v1[i+1], v2[2*i+1], v2[2*i+2]])]
    return faces


def faces_sphere_add_keep(r,u1,u2,N,shift):
    """
    add a row of faces to sphere surface while keeping discretization

    Parameters
    ---------
    - r (float): radius
    - u1 (float): first polar angle
    - u2 (float): second polar angle
    - N (int): discretization

    Returns
    -------
    list of faces that are ndarray (N,3)
    """
    t1 = np.linspace(0,4*np.pi,2*N+1) - (1-shift)*np.pi/N/2
    t2 = np.linspace(0,4*np.pi,2*N+1) + (1+shift)*np.pi/N/2
    v1 = r*np.array([(np.cos(t)*np.cos(u1), np.sin(t)*np.cos(u1), np.sin(u1)) for t in t1])
    v2 = r*np.array([(np.cos(t)*np.cos(u2), np.sin(t)*np.cos(u2), np.sin(u2)) for t in t2])
    faces = []
    for i in range(N):
        faces += [np.array([v1[i], v1[i+1], v2[i]])]
        faces += [np.array([v1[i+1], v2[i], v2[i+1]])]
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
