"""collection class code"""

import copy
from magpylib._lib.math_utility import format_src_input, check_duplicates
from magpylib._lib.fields import getB, getH
from magpylib._lib.display import display


class Collection:
    """ Group multiple sources in one collection for common manipulation.

    Operations applied to a collection are sequentially applied to all
        sources in the collection.

    Collections do not allow duplicate sources.

    Parameters
    ----------
    sources: src objects, collections or arbitrary lists thereof
        Ordered list of sources in the collection.

    Dunders
    -------
    __add__:
        Add collections like "col1 + col2"

    __sub__:
        Subtract sources like "col - src"

    __iter__:
        sources list is returned as collection iterable "[src for src in col]"

    __getitem__:
        sources list is returned for getitem "src = col[1]"

    __repr__:
        returns string "Collection (id(self))"

    Methods
    -------
    add: src objects, collections or arbitrary lists thereof
        Add arbitrary sources to collection.

    remove: source object
        Remove source from collection

    getB: array_like or sens_obj or list of sens_obj
        Compute B-field of collection at observer positions.

    getH: array_like or sens_obj or list of sens_obj
        Compute H-field of collection at observer positions.

    display: **kwargs of top level function display()
        Display Collection graphically using matplotlib.

    move_by: displacement
        Linear displacement of Collection by argument vector.

    rotate: scipy Rotation object
        Rotate all sources in collection.

    rotate_from_angax: angle(float), axis(array_like, shape(3,))
        Rotate all sources in collection with angle-axis input.

    copy:
        Returns a copy of the collection.
    
    reset_path:
        Set all src.pos to (0,0,0) and src.rot to unit rotation.

    Returns
    -------
    Collection object
    """

    def __init__(self, *sources):

        self.sources = sources


    # sources properties --------------------------------------------
    @property
    def sources(self):
        """ list of sources in Collection
        """
        return self._sources


    @sources.setter
    def sources(self, sources):
        """ set sources in the Collection, arbitrary input
        """
        # format input
        src_list = format_src_input(sources)
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # set attributes
        self._sources = src_list


    # dunders -------------------------------------------------------
    def __add__(self, source):
        """ add sources or collections
        """
        self.add(source)
        return self


    def __sub__(self, source):
        """ remove a source from the Collection
        """
        self.remove(source)
        return self


    def __iter__(self):
        """ make Collection iterable
        """
        yield from self._sources


    def __getitem__(self,i):
        """ provide getitem property
        """
        return self._sources[i]

    def __repr__(self) -> str:
        return f'Collection ({str(id(self))})'


    # methods -------------------------------------------------------
    def add(self,*sources):
        """ Add arbitrary sources to collection.

        Parameters
        ----------
        sources: src objects, collections or arbitrary lists thereof
            Add arbitrary sequences of sources and collections to the collection.
            The new sources will be added at the end of self.sources. Duplicates
            will be eliminated.

        Returns:
        --------
        self: Collection
        """
        # format input
        src_list = format_src_input(sources)
        # combine with original src_list
        src_list = self._sources + src_list
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # set attributes
        self._sources = src_list
        return self


    def remove(self,source):
        """ Remove source from collection

        Parameters
        ----------
        source: source object
            Remove source from collection

        Returns
        -------
        self: collection
        """
        self._sources.remove(source)
        return self


    def getB(self, observers, **specs):
        """ Compute B-field of collection at observer positions.

        Parameters
        ----------
        observers: array_like or sens_obj or list of sens_obj
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
            a 1D list of K sensors with pos_pix shape of (N1, N2, ..., 3)
            in units of millimeters.

        specs:
            Specific keyword arguments of different source types

        Returns
        -------
        B-field: ndarray, shape (M, K, N1, N2, ..., 3), unit [mT]
            B-field of collection at each path position M for each sensor K and each sensor pixel
            position N in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor)
            is removed.
        """
        B = getB(self, observers, **specs)
        return B


    def getH(self, observers, **specs):
        """ Compute H-field of collection at observer positions.

        Parameters
        ----------
        observers: array_like or sens_obj or list of sens_obj
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
            a 1D list of K sensors with pos_pix shape of (N1, N2, ..., 3)
            in units of millimeters.

        specs:
            Specific keyword arguments of different source types

        Returns
        -------
        H-field: ndarray, shape (M, K, N1, N2, ..., 3), unit [kA/m]
            H-field of collection at each path position M for each sensor K and each sensor pixel
            position N in units of [kA/m].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor)
            is removed.
        """
        H = getH(self, observers, **specs)
        return H


    def display(self, **kwargs):
        """ Display Collection graphically using matplotlib.

        Parameters
        ----------
        markers: array_like, shape (N,3), default=[(0,0,0)]
            Mark positions in graphic output. Default value puts a marker
            in the origin.

        axis: pyplot.axis, default=None
            Display graphical output in a given pyplot axis (must be 3D).

        direc: bool, default=False
            Set True to plot magnetization and current directions

        show_path: bool/string, default=True
            Set True to plot object paths. Set to 'all' to plot an object
            represenation at each path position.

        Returns
        -------
        no return
        """
        display(self, **kwargs)


    def move_by(self, displacement, steps=None):
        """ Linear displacement of source objects in Collection.

        Parameters
        ----------
        displacement: array_like, shape (3,)
            Displacement vector in units of mm.

        steps: int or None, default=None
            If steps=None: Object will simply be moved without generating a
                path. Specifically, path[-1] of object is set anew. This is
                similar to having steps=-1.
            If steps < 0: apply a linear motion from 0 to displ on top
                of existing path[steps:]. Specifically, steps=-1 will just
                displace path[-1].
            If steps > 0: add linear displacement to existing path starting
                at path[-1].

        Returns:
        --------
        self: Collection.
        """
        for s in self:
            s.move_by(displacement, steps)
        return self


    def rotate(self, rot, anchor=None, steps=-1):
        """
        Rotate all sources in collection.

        Parameters
        ----------
        rot: scipy Rotation object

        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. When anchor=None
            the object will rotate about its own center.

        steps: int, optional, default=-1
            If steps < 0: apply linear rotation steps from 0 to rot on top
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : Collection
        """
        for s in self:
            s.rotate(rot, anchor, steps)
        return self


    def rotate_from_angax(self, angle, axis, anchor=None, steps=-1, degree=True):
        """ Rotate all sources in collection.

        Parameters
        ----------
        angle: float
            Angle of rotation (in [deg] by default).

        axis: array_like, shape (3,)
            The axis of rotation [dimensionless]

        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. By default
            anchor=None the object will rotate about its own center.

        degree: bool, default=True
            If True, Angle is given in [deg]. If False, angle is given in [rad].

        steps: int, optional, default=-1
            If steps < 0: apply linear rotation steps from 0 to rot on top
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : Collection
        """
        for s in self:
            s.rotate_from_angax(angle, axis, anchor, steps, degree)
        return self


    def copy(self):
        """ Returns a copy of the collection"""
        return copy.copy(self)


    def reset_path(self):
        """Set all src.pos to (0,0,0) and src.rot to unit rotation.
        """
        for obj in self:
            obj.reset_path()
