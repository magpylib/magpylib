""" matplotlib draw-functionalities"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_directs(faced_objects, cmap, ax):
    """draw direction of magetization and currents
    """
    # pylint: disable=protected-access
    for i,obj in enumerate(faced_objects):
        col = cmap(i/len(faced_objects))
        pos = obj._pos[-1]
        mag = obj.mag
        length = 2*np.amax(obj.dim)
        direc = mag / (np.linalg.norm(mag)+1e-6)
        ax.quiver(pos[0], pos[1], pos[2],
                  direc[0], direc[1], direc[2],
                  length=length,
                  color=col)


def draw_markers(markers, ax):
    """ name = programm
    """
    ax.plot(markers[:,0],markers[:,1],markers[:,2],
            ls='',
            marker='x',
            ms=5)


def draw_path(obj, col, ax):
    """ se name is se program :)
    """
    # pylint: disable=protected-access
    path = obj._pos
    if len(path)>1:
        ax.plot(path[:,0], path[:,1], path[:,2],
                ls='-',
                lw=1,
                color=col,
                marker='.',
                mfc='k',
                mec='k',
                ms=2.5)
        ax.plot([path[0,0]],[path[0,1]],[path[0,2]],
                marker='o',
                ms=4,
                mfc=col,
                mec='k')


def draw_faces(faces, col, lw, ax):
    """ the name is the progam :)
    """
    boxf = Poly3DCollection(
        faces,
        facecolors=col,
        linewidths=lw,
        edgecolors='k',
        alpha=1)
    ax.add_collection3d(boxf)


def draw_sensors(sensors,ax):
    """ the name is the program :)
    """
    # pylint: disable=protected-access
    poss, exs, eys, ezs = np.empty((4,len(sensors),3))
    pixel = np.empty((1,3))
    # collect data for plots
    for i,sens in enumerate(sensors):
        rot = sens._rot[-1]
        pos = sens._pos[-1]
        poss[i] = pos
        exs[i] = rot.apply((1,0,0))
        eys[i] = rot.apply((0,1,0))
        ezs[i] = rot.apply((0,0,1))
        pos_pix_flat = sens.pos_pix.reshape((-1,3))
        pixel = np.r_[pixel, pos + rot.apply(pos_pix_flat)]
    # quiver plot of basis vectors
    for col,es in zip(['r','g','b'],[exs,eys,ezs]):
        ax.quiver(poss[:,0], poss[:,1], poss[:,2], es[:,0], es[:,1], es[:,2],
                 color=col,
                 length=1)
    # plot of pixels
    ax.plot(pixel[1:,0], pixel[1:,1], pixel[1:,2],
            marker='.',
            ms=2,
            color='k',
            ls='')
