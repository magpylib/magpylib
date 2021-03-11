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
    """ draw path in given color and return list of path-points
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
    return list(path)


def draw_faces(faces, col, lw, ax):
    """ draw faces in respective color and return list of vertex-points
    """
    boxf = Poly3DCollection(
        faces,
        facecolors=col,
        linewidths=lw,
        edgecolors='k',
        alpha=1)
    ax.add_collection3d(boxf)
    return faces


def draw_sensors(sensors, ax, show_path):
    """ draw sensors and return a list of pixel-points in gloabl CS
    """
    if not sensors:
        return []

    # pylint: disable=protected-access
    possis, exs, eys, ezs, pixel = [], [], [], [], []
    # collect data for plots
    for sens in sensors:
        if not isinstance(show_path, bool) and sens._pos.ndim>1:
            rots = sens._rot[::-show_path]
            poss = sens._pos[::-show_path]
        else:
            rots = [sens._rot[-1]]
            poss = [sens._pos[-1]]
        pos_pixel_flat = np.reshape(sens.pos_pix, (-1,3))

        for rot,pos in zip(rots,poss):
            possis += [pos]
            exs += [rot.apply((1,0,0))]
            eys += [rot.apply((0,1,0))]
            ezs += [rot.apply((0,0,1))]

            for pix in pos_pixel_flat:
                pixel += [pos + rot.apply(pix)]

    possis = np.array(possis)
    pixel = np.array(pixel)
    exs = np.array(exs)
    eys = np.array(eys)
    ezs = np.array(ezs)

    # quiver plot of basis vectors
    for col,es in zip(['r','g','b'],[exs,eys,ezs]):
        ax.quiver(possis[:,0], possis[:,1], possis[:,2], es[:,0], es[:,1], es[:,2],
                 color=col,
                 length=1)

    # plot of pixels
    ax.plot(pixel[:,0], pixel[:,1], pixel[:,2],
            marker='o',
            mfc='w',
            mew=1,
            mec='k',
            ms=2,
            ls='')

    return list(pixel[1:])
