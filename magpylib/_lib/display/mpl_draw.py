""" matplotlib draw-functionalities"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from magpylib import _lib


def draw_directs_faced(faced_objects, cmap, ax, show_path, size_direction):
    """draw direction of magetization of faced magnets

    Parameters
    ----------
    - faced_objects(list of src objects): with magnetization vector to be drawn
    - cmap(Pylplot colormap): used in display() function
    - ax(Pyplot 3D axis): to draw in
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # avoid circular imports
    Cuboid = _lib.obj_classes.Cuboid
    Cylinder = _lib.obj_classes.Cylinder

    for i,obj in enumerate(faced_objects):

        # add src attributes position and orientation depending on show_path
        if not isinstance(show_path, bool) and obj._position.ndim>1:
            rots = obj._orientation[::-show_path]
            poss = obj._position[::-show_path]
        else:
            rots = [obj._orientation[-1]]
            poss = [obj._position[-1]]

        # vector length, color and magnetization
        if isinstance(obj, Cuboid):
            length = 1.8*np.amax(obj.dimension)
        elif isinstance(obj, Cylinder):
            odim = obj.dimension
            if odim.shape == (2,):
                length = 1.8*np.amax(odim)
            else:
                length = 1.8*np.amax(odim[[1,4]]) #r2,h
        else:
            length = 1.8*obj.diameter # Sphere
        col = cmap(i/len(faced_objects))
        mag = obj.magnetization

        # collect all draw positions and directions
        draw_pos, draw_direc = [], []
        for rot,pos in zip(rots,poss):
            if isinstance(obj, Cylinder): # change cylinder_tile draw_pos to geometric center
                odim = obj.dimension
                if odim.shape==(5,):
                    r1,r2,phi1,phi2,_ = odim
                    phi_mid = (phi1+phi2)/2*np.pi/180
                    r_mid = (r2+r1)/2
                    shift = r_mid*np.array([np.cos(phi_mid),np.sin(phi_mid),0])
                    draw_pos += [pos + shift]
                else:
                    draw_pos += [pos]
            else:
                draw_pos += [pos]
            direc = mag / (np.linalg.norm(mag)+1e-6)
            draw_direc += [rot.apply(direc)]
        draw_pos = np.array(draw_pos)
        draw_direc = np.array(draw_direc)

        # use quiver() separately for each object to easier control
        # color and vector length
        ax.quiver(draw_pos[:,0], draw_pos[:,1], draw_pos[:,2],
            draw_direc[:,0], draw_direc[:,1], draw_direc[:,2],
            length=length*size_direction,
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
    path = obj._position
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
    cuboidf = Poly3DCollection(
        faces,
        facecolors=col,
        linewidths=lw,
        edgecolors='k',
        alpha=1)
    ax.add_collection3d(cuboidf)
    return faces


def draw_pixel(sensors, ax, show_path):
    """ draw pixels and return a list of pixel-points in gloabl CS
    """
    # pylint: disable=protected-access

    # no sensors given
    if not sensors:
        return []

    # collect sensor and pixel positions in global CS
    pos_sens, pos_pixel = [], []
    for sens in sensors:
        if not isinstance(show_path, bool) and sens._position.ndim>1:
            rots = sens._orientation[::-show_path]
            poss = sens._position[::-show_path]
        else:
            rots = [sens._orientation[-1]]
            poss = [sens._position[-1]]
        pos_pixel_flat = np.reshape(sens.pixel, (-1,3))

        for rot,pos in zip(rots,poss):
            pos_sens += [pos]

            for pix in pos_pixel_flat:
                pos_pixel += [pos + rot.apply(pix)]

    pos_all = pos_sens + pos_pixel
    pos_pixel = np.array(pos_pixel)

    # display pixel positions
    ax.plot(pos_pixel[:,0], pos_pixel[:,1], pos_pixel[:,2],
            marker='o',
            mfc='w',
            mew=1,
            mec='k',
            ms=2,
            ls='')

    # return all positions for system size evaluation
    return list(pos_all)


def draw_sensors(sensors, ax, sys_size, show_path, size_sensors):
    """ draw sensor cross
    """
    # pylint: disable=protected-access

    # no sensors in display
    if not sensors:
        return

    # collect plot data
    possis, exs, eys, ezs = [], [], [], []
    for sens in sensors:
        if not isinstance(show_path, bool) and sens._position.ndim>1:
            rots = sens._orientation[::-show_path]
            poss = sens._position[::-show_path]
        else:
            rots = [sens._orientation[-1]]
            poss = [sens._position[-1]]

        for rot,pos in zip(rots,poss):
            possis += [pos]
            exs += [rot.apply((1,0,0))]
            eys += [rot.apply((0,1,0))]
            ezs += [rot.apply((0,0,1))]

    possis = np.array(possis)
    exs = np.array(exs)
    eys = np.array(eys)
    ezs = np.array(ezs)

    # quiver plot of basis vectors
    arrowlength = sys_size*size_sensors/15
    for col,es in zip(['r','g','b'],[exs,eys,ezs]):
        ax.quiver(possis[:,0], possis[:,1], possis[:,2], es[:,0], es[:,1], es[:,2],
                 color=col,
                 length=arrowlength)

    return


def draw_dipoles(dipoles, ax, sys_size, show_path, size_dipoles):
    """ draw dipoles
    """
    # pylint: disable=protected-access

    # no sensors in display
    if not dipoles:
        return

    # collect plot data
    possis, moms = [], []
    for dip in dipoles:
        if not isinstance(show_path, bool) and dip._position.ndim>1:
            rots = dip._orientation[::-show_path]
            poss = dip._position[::-show_path]
        else:
            rots = [dip._orientation[-1]]
            poss = [dip._position[-1]]
        mom = dip.moment/np.linalg.norm(dip.moment)

        for rot,pos in zip(rots,poss):
            possis += [pos]
            moms += [rot.apply(mom)]

    possis = np.array(possis)
    moms = np.array(moms)

    # quiver plot of basis vectors
    arrowlength = sys_size*size_dipoles/15
    ax.quiver(possis[:,0], possis[:,1], possis[:,2], moms[:,0], moms[:,1], moms[:,2],
        color='k',
        length=arrowlength)

    return


def draw_circular(circulars, show_path, ax):
    """ draw circulars and return a list of positions
    """
    # pylint: disable=protected-access

    # graphical settings
    discret = 72+1
    col = 'k'
    lw = 1

    draw_pos = [] # line positions
    for circ in circulars:

        # add src attributes position and orientation depending on show_path
        if not isinstance(show_path, bool) and circ._position.ndim>1:
            rots = circ._orientation[::-show_path]
            poss = circ._position[::-show_path]
        else:
            rots = [circ._orientation[-1]]
            poss = [circ._position[-1]]

        # init orientation line positions
        radius = circ.diameter/2
        angs = np.linspace(0, 2*np.pi, discret)
        possis0 = np.array([radius*np.cos(angs), radius*np.sin(angs), np.zeros((discret))]).T

        # apply pos and rot, draw, store line positions
        for rot,pos in zip(rots,poss):
            possis1 = rot.apply(possis0) + pos
            ax.plot(possis1[:,0], possis1[:,1], possis1[:,2], color=col, lw=lw)
            draw_pos += list(possis1)

    return draw_pos


def draw_line(lines, show_path, ax) -> list:
    """ draw lines and return a list of positions
    """
    # pylint: disable=protected-access

    # graphical settings
    col = 'k'
    lw = 1

    draw_pos = [] # line positions
    for line in lines:

        # add src attributes position and orientation depending on show_path
        if not isinstance(show_path, bool) and line._position.ndim>1:
            rots = line._orientation[::-show_path]
            poss = line._position[::-show_path]
        else:
            rots = [line._orientation[-1]]
            poss = [line._position[-1]]

        # init orientation line positions
        possis0 = line.vertices

        # apply pos and rot, draw, store line positions
        for rot,pos in zip(rots,poss):
            possis1 = rot.apply(possis0) + pos
            ax.plot(possis1[:,0], possis1[:,1], possis1[:,2], color=col, lw=lw)
            draw_pos += list(possis1)

    return draw_pos
