""" Display function codes"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import magpylib3._lib as _lib
from magpylib3._lib.math.utility import format_src_input


def vert_face_box(s):
    """
    compute vertices and faces of Box input for plotting
    
    takes Box source, returns vert, faces
    """
    a,b,c = s.dim
    vert = np.array(((0,0,0),(a,0,0),(0,b,0),(0,0,c),
                    (a,b,0),(a,0,c),(0,b,c),(a,b,c)))
    vert = vert - s.dim/2
    vert = s.rot.apply(vert) + s.pos
    faces = [
        [vert[0],vert[1],vert[4],vert[2]],
        [vert[0],vert[1],vert[5],vert[3]],
        [vert[0],vert[2],vert[6],vert[3]],
        [vert[7],vert[6],vert[2],vert[4]],
        [vert[7],vert[6],vert[3],vert[5]],
        [vert[7],vert[5],vert[1],vert[4]],
        ]
    return vert, faces


def vert_face_cylinder(s):
    """
    compute vertices and faces of cylinder input for plotting
    
    takes Cylinder source, returns vert, faces
    """
    res = 20
    d,h = s.dim
    phis = np.linspace(0,2*np.pi,res)
    vert_circ = np.array([np.cos(phis),np.sin(phis),np.zeros(res)]).T*d/2
    vt = vert_circ + np.array([0,0,h/2]) # top vertices
    vb = vert_circ - np.array([0,0,h/2]) # bott vertices
    vt = s.rot.apply(vt) + s.pos
    vb = s.rot.apply(vb) + s.pos
    vert = np.r_[vt,vb]
    faces = [[vt[i], vb[i], vb[i+1], vt[i+1]] for i in range(res-1)]
    faces += [vt, vb]
    return vert, faces


def display(
    *sources,  
    markers=[(0,0,0)], 
    subplotAx=None,
    direc=False):
    """ Display sources/sensors graphically

    Args:
        *sources (objects): can be sources, collections or sensors
        markers (list of positions): Mark positions in graphic output.
            Defaults to [(0,0,0)] which marks the origin.
        subplotAx (pyplot.axis): display graphical output in 
            a given pyplot axis (must be 3D). Defaults to None.
        direc (bool): True to plot magnetization and current directions,
            Defaults to False.
    """

    # if no subplot axis is given
    suppress=False

    # flatten input, determine number of sources
    src_list = format_src_input(sources)
    n = len(src_list)

    # create or set plotting axis
    if subplotAx is None:
        fig = plt.figure(dpi=80, figsize=(8,8))
        ax = fig.gca(projection='3d')
    else:
        ax = subplotAx
        suppress = True
    
    # load color map
    cm = plt.cm.get_cmap('hsv')

    # init sys_size
    sys_size = [[],[],[],[],[],[]]

    # init directs for directs plotting :)
    directs = []

    # draw sources --------------------------------------------------
    for i,s in enumerate(src_list):
        if isinstance(s, _lib.obj_classes.Box):
            vert, faces = vert_face_box(s)
            lw = 0.5
        elif isinstance(s, _lib.obj_classes.Cylinder):
            vert, faces = vert_face_cylinder(s)
            lw = 0.25
        else:
            print('ERROR display: bad input source type')
            sys.exit()
        # add faces to plot
        boxf = Poly3DCollection(
            faces, 
            facecolors=cm(i/n),
            linewidths=lw,
            edgecolors='k',
            alpha=1)
        ax.add_collection3d(boxf)
        
        # determine outmost vertices to adjust sys_size
        for i in range(3):
            sys_size[2*i] += [np.amax(vert[:,i])]
            sys_size[2*i+1] += [np.amin(vert[:,i])]
        
        # add to directions
        pos = s.pos
        mag = s.rot.apply(s.mag)
        directs += [np.r_[pos,mag]]

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
    cx = (xmax + xmin)/2
    cy = (ymax + ymin)/2
    cz = (zmax + zmin)/2
    # size
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    # cube plot
    dd = max([dx,dy,dz])/2 * (5 + n)/n
    
    # draw directions -----------------------------------------------
    if direc:
        dira = np.array(directs).T
        ax.quiver(dira[0], dira[1], dira[2],   # pylint: disable=unsubscriptable-object
                    dira[3], dira[4], dira[5], # pylint: disable=unsubscriptable-object
                    normalize=True,
                    length=dd/np.sqrt(n),
                    color='.3'
                    )

    # plot styling --------------------------------------------------
    ax.set(
        xlabel = 'x [mm]',
        ylabel = 'y [mm]',
        zlabel = 'z [mm]',
        xlim = (cx-dd, cx+dd),
        ylim = (cy-dd, cy+dd),
        zlim = (cz-dd, cz+dd)
        )

    if not suppress:
        plt.show()
    
    return None