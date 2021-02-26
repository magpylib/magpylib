""" Display function codes"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import magpylib3._lib as _lib
from magpylib3._lib.math_utility.utility import format_src_input, same_path_length


def vert_face_box(s,p,r):
    """
    compute vertices and faces of Box input for plotting

    takes Box source, returns vert, faces
    """
    a,b,c = s.dim
    vert = np.array(((0,0,0),(a,0,0),(0,b,0),(0,0,c),
                     (a,b,0),(a,0,c),(0,b,c),(a,b,c)))
    vert = vert - s.dim/2
    vert = r.apply(vert) + p
    faces = [
        [vert[0],vert[1],vert[4],vert[2]],
        [vert[0],vert[1],vert[5],vert[3]],
        [vert[0],vert[2],vert[6],vert[3]],
        [vert[7],vert[6],vert[2],vert[4]],
        [vert[7],vert[6],vert[3],vert[5]],
        [vert[7],vert[5],vert[1],vert[4]],
        ]
    return vert, faces


def vert_face_cylinder(s,p,r):
    """
    compute vertices and faces of cylinder input for plotting

    takes Cylinder source, returns vert, faces
    """
    res = 20
    d,h = s.dim
    phis = np.linspace(0,2*np.pi,res)
    vert_circ = np.array([np.cos(phis),np.sin(phis),np.zeros(res)]).T*d/2
    v_t = vert_circ + np.array([0,0,h/2]) # top vertices
    v_b = vert_circ - np.array([0,0,h/2]) # bott vertices
    v_t = r.apply(v_t) + p
    v_b = r.apply(v_b) + p
    vert = np.r_[v_t, v_b]
    faces = [[v_t[i], v_b[i], v_b[i+1], v_t[i+1]] for i in range(res-1)]
    faces += [v_t, v_b]
    return vert, faces


def display(
        *objects,
        markers=[(0,0,0)],
        axis=None,
        direc=False,
        show_path=False):
    """ Display sources/sensors graphically

    Parameters
    ----------
    objects: sources, collections or sensors
        Display a 3D reprensation of given objects using matplotlib

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Mark positions in graphic output. Puts a marker in the origin.
        by default.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D).

    direc: bool, default=False
        Set True to plot magnetization and current directions

    show_path: bool/string, default=False
        Set True to plot object paths. Set 'all' to plot an object
        represenation at each path position.

    Returns
    -------
    None
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=dangerous-default-value

    # if no subplot axis is given
    generate_output=True

    # flatten input, determine number of objects
    obj_list = format_src_input(objects)
    n = len(obj_list)

    # create or set plotting axis
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8,8))
        ax = fig.gca(projection='3d')
    else:
        ax = axis
        generate_output = False

    # load color map
    cmap = plt.cm.get_cmap('hsv')

    # init sys_size x/y/z max and min values
    sys_size = [[],[],[],[],[],[]]

    # init directs for directs plotting :)
    directs = []

    # test object paths ----------------------------------------
    for obj in obj_list:
        if not same_path_length([obj]):
            sys.exit('ERROR: display() - bad path format (different pos/rot length)')

    # draw objects --------------------------------------------------
    for i,s in enumerate(obj_list):
        if isinstance(s, _lib.obj_classes.Box):
            vert_face = vert_face_box
            lw = 0.5
        elif isinstance(s, _lib.obj_classes.Cylinder):
            vert_face = vert_face_cylinder
            lw = 0.25
        else:
            sys.exit('ERROR: display(), bad src input')

        if show_path=='all':
            poss = s._pos
            rott = s._rot
        else:
            poss = [s._pos[-1]]
            rott = [s._rot[-1]]

        vert, faces = [],[]
        for p,r in zip(poss,rott):
            v, f = vert_face(s,p,r)
            vert += [v]
            faces += f

        # add faces to plot
        boxf = Poly3DCollection(
            faces,
            facecolors=cmap(i/n),
            linewidths=lw,
            edgecolors='k',
            alpha=1)
        ax.add_collection3d(boxf)

        # determine outmost vertices to adjust sys_size
        for j in range(3):
            sys_size[2*j] += [np.amax(np.array(vert)[:,:,j])]
            sys_size[2*j+1] += [np.amin(np.array(vert)[:,:,j])]

        # add to directions
        pos = s._pos[-1]
        mag = s._rot[-1].apply(s.mag)
        directs += [np.r_[pos,mag]]


    # path ------------------------------------------------------
    if show_path is True:
        for i,s in enumerate(obj_list):
            path = s._pos
            ax.plot(path[:,0],path[:,1],path[:,2],
                    ls = '-',
                    lw = 1,
                    color = cmap(i/n),
                    marker = '.',
                    mfc = 'k',
                    mec='k',
                    ms = 2.5)
            ax.plot([path[0,0]],[path[0,1]],[path[0,2]],
                    marker='o',
                    ms=4,
                    mfc=cmap(i/n),
                    mec='k')

            # determine outmost vertices to adjust sys_size
            for j in range(3):
                sys_size[2*j] += [np.amax(path[:,j])]
                sys_size[2*j+1] += [np.amin(path[:,j])]

    # markers -------------------------------------------------------
    markers = np.array(markers)
    ax.plot(markers[:,0],markers[:,1],markers[:,2],ls='',marker='x',ms=5)

    # determine outlost markers to adjust sys_size
    for i in range(3):
        sys_size[2*i] += [np.amax(markers[:,i])]
        sys_size[2*i+1] += [np.amin(markers[:,i])]

    # final system size analysis ------------------------------------
    xmax, xmin = max(sys_size[0]), min(sys_size[1])
    ymax, ymin = max(sys_size[2]), min(sys_size[3])
    zmax, zmin = max(sys_size[4]), min(sys_size[5])

    # center
    centx = (xmax + xmin)/2
    centy = (ymax + ymin)/2
    centz = (zmax + zmin)/2
    # size
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    # cube plot
    dd = max([dx,dy,dz])*.7

    # draw directions -----------------------------------------------
    if direc:
        dira = np.array(directs).T
        ax.quiver(dira[0], dira[1], dira[2],
                  dira[3], dira[4], dira[5],
                  normalize=True,
                  length=dd/np.sqrt(n),
                  color='.3')

    # plot styling --------------------------------------------------
    ax.set(
        xlabel = 'x [mm]',
        ylabel = 'y [mm]',
        zlabel = 'z [mm]',
        xlim = (centx-dd, centx+dd),
        ylim = (centy-dd, centy+dd),
        zlim = (centz-dd, centz+dd)
        )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()
