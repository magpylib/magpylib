# -------------------------------------------------------------------------------
# magpylib -- A Python 3 toolbox for working with magnetic fields.
# Copyright (C) Silicon Austria Labs, https://silicon-austria-labs.com/,
#               Michael Ortner <magpylib@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
# The acceptance of the conditions of the GNU Affero General Public License are
# compulsory for the usage of the software.
#
# For contact information, reach out over at <magpylib@gmail.com> or our issues
# page at https://www.github.com/magpylib/magpylib/issues.
# -------------------------------------------------------------------------------

from magpylib._lib.classes.base import FieldSampler
from magpylib._lib.utility import addListToCollection, isSource,  addUniqueSource

class Collection(FieldSampler):
    """
    Create a collection of :mod:`magpylib.source` objects for common manipulation.

    Parameters
    ----------
    sources : source objects
        python magic variable passes source objects to the collection at initialization.

    Attributes
    ----------
    sources : list of source objects
        List of all sources that have been added to the collection.

    Example
    -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> col = Collection(pm1,pm2,pm3)
        >>> B = col.getB([1,0,1])
        >>> print(B)
        [9.93360625e+01 1.76697482e-14 3.12727683e+01]
    """

    def __init__(self, *sources, dupWarning=True):

        self.sources = []

        # The following will add Sources to the Collection sources list,
        # The code is the same as the addsource method.
        # addSource() is not cast here because it will
        # put a tuple inside a tuple.
        # Iterating for this would compromise performance.
        for s in sources:
            if type(s) == Collection:
                addListToCollection(self.sources, s.sources, dupWarning)
            elif isinstance(s, list) or isinstance(s, tuple):
                addListToCollection(self.sources, s, dupWarning)
            else:
                assert isSource(s), "Argument " + str(s) + \
                    " in addSource is not a valid source for Collection"
                if dupWarning is True:
                    addUniqueSource(s, self.sources)
                else:
                    self.sources += [s]

    def removeSource(self, source_ref=-1):
        """
        Remove a source from the sources list. 

        Parameters
        ----------

        source_ref : source object or int
            [Optional] Remove the inputted source from the list
            [Optional] If given an int, remove a source at the given index position. Default: Last position.

        Return
        ------

        Popped source object.

        Raises
        ------

        ValueError
            Will be thrown if you attempt to remove a source that is not in the Collection.

        AssertionError
            Will be thrown if inputted index kwarg type is not type int

        Example
        -------

            >>> from magpylib import Collection, source
            >>> s = source.magnet.Sphere(mag=[1,2,3],dim=1,pos=[3,3,3])
            >>> s2 = source.magnet.Sphere(mag=[1,2,3],dim=2,pos=[-3,-3,-3])
            >>> m = source.moment.Dipole(moment=[1,2,3],pos=(0,0,0))
            >>> c = Collection(s,s2,m)
            >>> print(c.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa31eafcc>, 
            <magpylib._lib.classes.magnets.Sphere object at 0xa31ea1cc>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.removeSource(s)
            >>> print(c.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa31ea1cc>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.removeSource(s2)
            >>> print(c.sources)
            [<magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.removeSource()
            >>> print(c.sources)
            []



        """
        assert type(source_ref) == int or isSource(
            source_ref), "Reference in removeSource is not an int nor a source"
        if type(source_ref) == int:
            try:
                return self.sources.pop(source_ref)
            except IndexError as e:  # Give a more helpful error message.
                raise type(e)(str(e) + ' - Index ' + str(source_ref) +
                              ' in collection source is not accessible for removeSource')
        else:
            try:
                self.sources.remove(source_ref)
            except ValueError as e:  # Give a more helpful error message.
                raise type(e)(str(e) + ' - ' + str(type(source_ref)
                                                   ) + ' not in list for removeSource')
            return source_ref

    def addSources(self, *sources, dupWarning=True):
        """
        This method adds the argument source objects to the collection.
        May also include other collections.

        Parameters
        ----------
        source : source object
            adds the source object `source` to the collection.

        dupWarning : bool
            Warn and prevent if there is an attempt to add a 
            duplicate source into the collection. Set to false to disable
            check and increase performance.

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> col = Collection(pm1)
        >>> print(col.getB([1,0,1]))
          [4.29223532e+01 1.76697482e-14 1.37461635e+01]
        >>> col.addSource(pm2)
        >>> print(col.getB([1,0,1]))
          [7.72389756e+01 1.76697482e-14 2.39070726e+01]
        >>> col.addSource(pm3)
        >>> print(
          [9.93360625e+01 1.76697482e-14 3.12727683e+01]
        """
        for s in sources:
            if type(s) == Collection:
                addListToCollection(self.sources, s.sources, dupWarning)
            elif isinstance(s, list) or isinstance(s, tuple):
                addListToCollection(self.sources, s, dupWarning)
            else:
                assert isSource(s), "Argument " + str(s) + \
                    " in addSource is not a valid source for Collection"
                if dupWarning is True:
                    addUniqueSource(s, self.sources)
                else:
                    self.sources += [s]

    def getB(self, pos):
        """
        This method returns the magnetic field vector generated by the whole
        collection at the argument position `pos` in units of [mT]

        Parameters
        ----------
        pos : vec3 [mm]
            Position where magnetic field should be determined.

        Returns
        -------
        magnetic field vector : arr3 [mT]
            Magnetic field at the argument position `pos` generated by the
            collection in units of [mT].
        """
        Btotal = sum([s.getB(pos) for s in self.sources])
        return Btotal

    def move(self, displacement):
        """
        This method moves each source in the collection by the argument vector `displacement`. 
        Vector input format can be either list, tuple or array of any data
        type (float, int).

        Parameters
        ----------
        displacement : vec3 - [mm]
            Displacement vector

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> print(pm1.position,pm2.position)
          [0. 0. 0.] [0. 0. 0.]
        >>> col = Collection(pm1,pm2)
        >>> col.move([1,1,1])
        >>> print(pm1.position,pm2.position)
          [1. 1. 1.] [1. 1. 1.]
        """
        for s in self.sources:
            s.move(displacement)

    def rotate(self, angle, axis, anchor='self.position'):
        """
        This method rotates each source in the collection about `axis` by `angle`. The axis passes
        through the center of rotation anchor. Scalar input is either integer or
        float. Vector input format can be either list, tuple or array of any
        data type (float, int).

        Parameters
        ----------
        angle  : scalar [deg]
            Angle of rotation in units of [deg]
        axis : vec3
            Axis of rotation
        anchor : vec3
            The Center of rotation which defines the position of the axis of rotation.
            If not specified all sources will rotate about their respective center.

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> print(pm1.position, pm1.angle, pm1.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> print(pm2.position, pm2.angle, pm2.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> col = Collection(pm1,pm2)
        >>> col.rotate(90, [0,1,0], anchor=[1,0,0])
        >>> print(pm1.position, pm1.angle, pm1.axis)
          [1. 0. 1.] 90.0 [0. 1. 0.]
        >>> print(pm2.position, pm2.angle, pm2.axis)
          [1. 0. 1.] 90.0 [0. 1. 0.]
        """
        for s in self.sources:
            s.rotate(angle, axis, anchor=anchor)

    def __repr__(self):
        name = getattr(self,'name',None)
        str_repr = [f"type: {type(self).__name__}"]
        if name is not None:
            str_repr.append(f"name: {name}")
        str_repr.extend([
            "sources: {}".format([f'{type(s).__name__}' for s in self.sources])
        ])
        return '\n '.join(str_repr)