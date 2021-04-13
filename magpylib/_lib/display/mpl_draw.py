""" matplotlib draw-functionalities"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_directs_faced(faced_objects, cmap, ax, show_path, size_direc):
    """draw direction of magetization of faced magnets

    Parameters
    ----------
    - faced_objects(list of src objects): with magnetization vector to be drawn
    - cmap(Pylplot colormap): used in display() function
    - ax(Pyplot 3D axis): to draw in
    - show_path(bool or int): draw on every position where object is displayed
    """
    #pylint: disable=protected-access

    for i,obj in enumerate(faced_objects):

        # add src attributes position and orientation depending on show_path
        if not isinstance(show_path, bool) and obj._pos.ndim>1:
            rots = obj._rot[::-show_path]
            poss = obj._pos[::-show_path]
        else:
            rots = [obj._rot[-1]]
            poss = [obj._pos[-1]]

        # vector length, color and magnetization
        length = 1.8*np.amax(obj.dim)
        col = cmap(i/len(faced_objects))
        mag = obj.mag

        # collect all draw positions and directions
        draw_pos, draw_direc = [], []
        for rot,pos in zip(rots,poss):
            draw_pos += [pos]
            direc = mag / (np.linalg.norm(mag)+1e-6)
            draw_direc += [rot.apply(direc)]
        draw_pos = np.array(draw_pos)
        draw_direc = np.array(draw_direc)

        # use quiver() separately for each object to easier control
        # color and vector length
        ax.quiver(draw_pos[:,0], draw_pos[:,1], draw_pos[:,2],
            draw_direc[:,0], draw_direc[:,1], draw_direc[:,2],
            length=length*size_direc,
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


def draw_sensors(sensors, ax, show_path, size_sensors):
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
                 length=size_sensors)

    # plot of pixels
    ax.plot(pixel[:,0], pixel[:,1], pixel[:,2],
            marker='o',
            mfc='w',
            mew=1,
            mec='k',
            ms=2,
            ls='')

    return list(pixel[1:])
