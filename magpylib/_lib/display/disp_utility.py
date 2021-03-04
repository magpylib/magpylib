""" Display function codes"""

import numpy as np


def faces_box(src):
    """
    compute vertices and faces of Box input for plotting

    takes Box source, returns vert, faces
    """
    # pylint: disable=protected-access
    a,b,c = src.dim
    vert = np.array(((0,0,0),(a,0,0),(0,b,0),(0,0,c),
                     (a,b,0),(a,0,c),(0,b,c),(a,b,c)))
    vert = vert - src.dim/2
    vert = src._rot[-1].apply(vert) + src._pos[-1]
    faces = [
        [vert[0],vert[1],vert[4],vert[2]],
        [vert[0],vert[1],vert[5],vert[3]],
        [vert[0],vert[2],vert[6],vert[3]],
        [vert[7],vert[6],vert[2],vert[4]],
        [vert[7],vert[6],vert[3],vert[5]],
        [vert[7],vert[5],vert[1],vert[4]],
        ]
    return faces


def faces_cylinder(src):
    """
    compute vertices and faces of cylinder input for plotting

    takes Cylinder source, returns vert, faces
    """
    # pylint: disable=protected-access
    res = 20
    d,h = src.dim
    phis = np.linspace(0,2*np.pi,res)
    vert_circ = np.array([np.cos(phis),np.sin(phis),np.zeros(res)]).T*d/2
    v_t = vert_circ + np.array([0,0,h/2]) # top vertices
    v_b = vert_circ - np.array([0,0,h/2]) # bott vertices
    v_t = src._rot[-1].apply(v_t) + src._pos[-1]
    v_b = src._rot[-1].apply(v_b) + src._pos[-1]
    faces = [[v_t[i], v_b[i], v_b[i+1], v_t[i+1]] for i in range(res-1)]
    faces += [v_t, v_b]
    return faces


def system_size(faced_objects, sensors, markers):
    """compute system size for display
    """
    sys_size = 0
    for obj in faced_objects:
        size = np.amax(abs(obj.pos))+np.amax(obj.dim)
        if size>sys_size:
            sys_size = size

    for sens in sensors:
        size = np.amax(abs(sens.pos)) + np.amax(abs(sens.pos_pix))
        if size>sys_size:
            sys_size = size

    size = np.amax(abs(markers))
    if size>sys_size:
        sys_size = size

    return sys_size
