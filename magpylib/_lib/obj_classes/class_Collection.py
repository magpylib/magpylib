"""collection class code"""

import copy
from magpylib._lib.utility import (format_obj_input, check_duplicates,
    only_allowed_src_types, format_getBH_class_inputs)
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr


# ON INTERFACE
class Collection(BaseDisplayRepr):
    """
    Group multiple sources in one Collection for common manipulation.

    Operations applied to a Collection are sequentially applied to all sources in the Collection.

    Collections do not allow duplicate sources. They will be eliminated automatically.

    Parameters
    ----------
    sources: src_obj, Collections or arbitrary lists thereof
        Ordered list of sources in the Collection.

    Dunders
    -------
    __add__:
        Add collections like "col1 + col2".

    __sub__:
        Subtract sources like "col - src".

    __iter__:
        Sources list is returned as Collection iterable "[src for src in col]".

    __getitem__:
        Sources list is returned for getitem "src = col[1]".

    __repr__:
        Returns "Collection(id)".

    Methods
    -------
    add(*sources):
        Add arbitrary sources to Collection.

    remove(source):
        Remove source from Collection.

    getB(observers):
        Compute B-field of Collection at observers.

    getH(observers):
        Compute H-field of collection at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Collection graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Collection by argument vector.

    rotate(rot, anchor=None, steps=None):
        Rotate all sources in Collection.

    rotate_from_angax(angle, axis, anchor=None, steps=None):
        Rotate all sources in Collection using angle-axis-anchor input.

    copy():
        Returns a copy of the Collection.

    reset_path():
        Set all src.pos to (0,0,0) and src.rot to unit rotation.

    Returns
    -------
    Collection object
    """

    def __init__(self, *sources):

        # inherit
        BaseDisplayRepr.__init__(self)

        self.sources = sources
        self.obj_type = 'Collection'


    # sources properties --------------------------------------------
    @property
    def sources(self):
        """ List of sources in Collection.
        """
        return self._sources


    @sources.setter
    def sources(self, sources):
        """ Set Collection sources.
        """
        # format input
        src_list = format_obj_input(sources)
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # allow only designated source types in Collection
        src_list = only_allowed_src_types(src_list)
        # set attributes
        self._sources = src_list


    # dunders -------------------------------------------------------
    def __add__(self, source):
        self.add(source)
        return self


    def __sub__(self, source):
        self.remove(source)
        return self


    def __iter__(self):
        yield from self._sources


    def __getitem__(self,i):
        return self._sources[i]


    # methods -------------------------------------------------------
    def add(self,*sources):
        """
        Add arbitrary sources to Collection.

        Parameters
        ----------
        sources: src objects, Collections or arbitrary lists thereof
            Add arbitrary sequences of sources and Collections to the Collection.
            The new sources will be added at the end of self.sources. Duplicates
            will be eliminated.

        Returns:
        --------
        self: Collection
        """
        # format input
        src_list = format_obj_input(sources)
        # combine with original src_list
        src_list = self._sources + src_list
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # set attributes
        self._sources = src_list
        return self


    def remove(self,source):
        """
        Remove source from Collection.

        Parameters
        ----------
        source: source object
            Remove the given source from the Collection.

        Returns
        -------
        self: Collection
        """
        self._sources.remove(source)
        return self


    def getB(self, *observers, squeeze=True, **specs):
        """
        Compute B-field of Collection at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        specs:
            Specific keyword arguments of different source types.

        Returns
        -------
        B-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [mT]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor
            or single pixel) is removed.
        """
        observers = format_getBH_class_inputs(observers)
        B = getB(self, observers, squeeze=squeeze, **specs)
        return B


    def getH(self, *observers, squeeze=True, **specs):
        """
        Compute H-field of Collection at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        specs:
            Specific keyword arguments of different source types.

        Returns
        -------
        H-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [kA/m]
            H-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [kA/m].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor
            or single pixel) is removed.
        """
        observers = format_getBH_class_inputs(observers)
        H = getH(self, observers, squeeze=squeeze, **specs)
        return H


    def move_by(self, displacement, steps=None):
        """
        Linear displacement of Collection by argument vector.

        Parameters
        ----------
        displacement: array_like, shape (3,), units [mm]
            Displacement vector in units of [mm].

        steps: int or None, default=None
            steps=None: Object is moved without generating a path. Specifically,
                path[-1] of object is set to the new position. This is similar
                to having steps=-1.
            steps < 0: Merge last |steps| path positions with a linear motion
                from 0 to displacement. Specifically, steps=-1 will just
                add displacement to path[-1].
            steps > 0: Append a linear motion from path[-1] to path[-1] + displacement
                to the exising path.

        Returns:
        --------
        self: Collection
        """
        for s in self:
            s.move_by(displacement, steps)
        return self


    def rotate(self, rot, anchor=None, steps=None):
        """
        Rotate all sources in Collection.

        Parameters
        ----------
        rot: scipy Rotation object
            Rotation input.

        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. By default the objects will
            rotate about their own center.

        steps: int, optional, default=None
            steps=None: Objects are rotated without generating a path. Specifically, path[-1]
                of objects are set to new position and orientation. This is similar to having
                steps=-1.
            steps < 0: Merge last |steps| path entries with stepwise rotation from 0 to rot.
                Specifically, steps=-1 will just add the rotation to path[-1].
            steps > 0: Append stepwise rotation from 0 to rot to existing path starting
                at path[-1].

        Returns:
        --------
        self : Collection
        """
        for s in self:
            s.rotate(rot, anchor, steps)
        return self


    def rotate_from_angax(self, angle, axis, anchor=None, steps=None, degree=True):
        """
        Rotate all sources in Collection using angle-axis-anchor input.

        Parameters
        ----------
        angle: float, unit [deg] or [rad]
            Angle of rotation in [deg] or [rad].

        axis: array_like, shape (3,)
            The axis of rotation.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default every object will rotate about its own center.

        degree: bool, default=True
            By default angle is given in units of [deg]. If degree=False, angle is given
            in units of [rad].

        steps: int, optional, default=None
            steps=None: Objects are rotated without generating a path. Specifically, path[-1]
                of objects are set to new position and orientation. This is similar to having
                steps=-1.
            steps < 0: Merge last |steps| path entries with stepwise rotation from 0 to rot.
                Specifically, steps=-1 will just add the rotation to path[-1].
            steps > 0: Append stepwise rotation from 0 to rot to existing path starting
                at path[-1].

        Returns:
        --------
        self : object with position and orientation properties
        """
        for s in self:
            s.rotate_from_angax(angle, axis, anchor, steps, degree)
        return self


    def copy(self):
        """
        Returns a copy of the Collection.
        """
        return copy.copy(self)


    def reset_path(self):
        """
        Set all src.pos to (0,0,0) and src.rot to unit rotation.
        """
        for obj in self:
            obj.reset_path()
