"""Collection class code"""
import copy
from magpylib._src.utility import (
    format_obj_input,
    check_duplicates,
    LIBRARY_SENSORS,
    LIBRARY_SOURCES)
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseTransform import BaseTransform
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.default_utils import validate_style_keys
from magpylib._src.exceptions import MagpylibBadUserInput


class BaseCollection(BaseDisplayRepr):
    """
    Group multiple objects in one Collection for common manipulation.

    Operations applied to a Collection are sequentially applied to all objects in the Collection.
    Collections do not allow duplicate objects (will be eliminated automatically).

    Collections have the following dunders defined: __add__, __sub__, __iter__, __getitem__,
    __repr__.

    Depending on the input, Collection objects can be sources, observers or both. A Collection
    with only source objects will become either a SourceCollection, one with only Sensors a
    SensorCollection, and one with sources and sensors a MixedCollection
    A SourceCollection functions like any SINGLE source. A SensorCollection functions like a
    list of observer inputs. A MixedCollection will function as source or as observer.

    Parameters
    ----------
    objects: sources, sensors, collections or arbitrary lists thereof
        Ordered list of objects in the Collection.

    Returns
    -------
    Collection object: Collection

    Examples
    --------

    Create Collections for common manipulation. All objects added to a Collection
    are stored in the ``objects`` attribute, and additionally in the ``sensors``
    and ``sources`` attributes. These three return ordered sets (lists with
    unique elements only)

    >>> import magpylib as magpy
    >>> sphere = magpy.magnet.Sphere((1,2,3),1)
    >>> loop = magpy.current.Loop(1,1)
    >>> dipole = magpy.misc.Dipole((1,2,3))
    >>> col = magpy.Collection(sphere, loop, dipole)
    >>> print(col.objects)
    [Sphere(id=1879891544384), Loop(id=1879891543040), Dipole(id=1879892157152)]

    Cycle directly through the Collection ``objects`` attribute

    >>> for src in col:
    >>>    print(src)
    Sphere(id=1879891544384)
    Loop(id=1879891543040)
    Dipole(id=1879892157152)

    and directly access objects from the Collection

    >>> print(col[1])
    Loop(id=1879891543040)

    Add and subtract objects to form a Collection and to remove objects from a Collection.

    >>> col = sphere + loop
    >>> print(col.objects)
    [Sphere(id=1879891544384), Loop(id=1879891543040)]
    >>> col - sphere
    >>> print(col.objects)
    [Loop(id=1879891543040)]

    Consider three collections, a SourceCollection sCol a SensorCollection xCol and a
    MixedCollection mCol, all made up from the same objects.

    >>> import numpy as np
    >>> import magpylib as magpy

    >>> s1=magpy.magnet.Sphere((1,2,3), 1)
    >>> s2=magpy.magnet.Cylinder((1,2,3), (1,1), (3,0,0))
    >>> s3=magpy.magnet.Cuboid((1,2,3), (1,1,1), (6,0,0))

    >>> x1=magpy.Sensor((1,0,3))
    >>> x2=magpy.Sensor((4,0,3))
    >>> x3=magpy.Sensor((7,0,3))

    >>> sCol = magpy.Collection(s1, s2, s3)
    >>> xCol = magpy.Collection(x1, x2, x3)
    >>> mCol = magpy.Collection(sCol, xCol)

    All the following lines will all give the same output

    >>> magpy.getB([s1,s2,s3], [x1,x2,x3], sumup=True)
    >>> magpy.getB(sCol, xCol)
    >>> magpy.getB(mCol, mCol)
    >>> sCol.getB(xCol)
    >>> xCol.getB(sCol)
    >>> sCol.getB(mCol)
    >>> xCol.getB(mCol)
    >>> mCol.getB()
    """

    def __init__(self, *objects):

        self._object_type = "Collection"

        # init inheritance
        BaseDisplayRepr.__init__(self)

        # instance attributes
        self._objects = []
        self._sources = []
        self._sensors = []
        self.objects = objects

    # property getters and setters
    @property
    def objects(self):
        """Collection objects attribute getter and setter."""
        return self._objects

    @objects.setter
    def objects(self, objects):
        """Set Collection objects."""
        obj_list = format_obj_input(objects, allow="sources+sensors")
        self._objects = []
        self.add(obj_list)

    @property
    def sources(self):
        """Collection sources attribute getter and setter."""
        return self._sources

    @sources.setter
    def sources(self, sources):
        """Set Collection sources."""
        src_list = format_obj_input(sources, allow="sources")
        self._objects = [o for o in self._objects if o not in self._sources]
        self.add(src_list)

    @property
    def sensors(self):
        """Collection sensors attribute getter and setter."""
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        """Set Collection sensors."""
        sens_list = format_obj_input(sensors, allow="sensors")
        self._objects = [o for o in self._objects if o not in self._sensors]
        self.add(sens_list)

    # dunders
    def __add__(self, obj):
        if obj._object_type == "Collection":
            new_obj = Collection(self, obj)
        else:
            new_obj = self.add(obj)
        return new_obj

    def __sub__(self, obj):
        return self.remove(obj)

    def __iter__(self):
        yield from self._objects

    def __getitem__(self, i):
        return self._objects[i]

    def __len__(self):
        return len(self._objects)

    def __repr__(self) -> str:
        # pylint: disable=protected-access
        if not self._sources:
            pref = "Sensor"
        elif not self._sensors:
            pref = "Source"
        else:
            pref = "Mixed"
        s = super().__repr__()
        return f"{pref}{s}"

    # methods -------------------------------------------------------
    def add(self, *objects):
        """
        Add arbitrary Magpylib objects or Collections.

        Parameters
        ----------
        objects: sources, Sensors, Collections or arbitrary lists thereof
            Add arbitrary sequences of objects and Collections to the Collection.
            The new objects will be added at the end of self.objects. Duplicates
            will be eliminated.

        Returns
        -------
        self: Collection

        Examples
        --------

        Add objects to a Collection:

        >>> import magpylib as magpy
        >>> src = magpy.current.Loop(1,1)
        >>> col = magpy.Collection()
        >>> col.add(src)
        >>> print(col.objects)
        [Loop(id=2519738714432)]

        """
        # format input
        obj_list = format_obj_input(objects)
        # combine with original obj_list
        obj_list = self._objects + obj_list
        # check and eliminate duplicates
        obj_list = check_duplicates(obj_list)
        # set attributes
        self._objects = obj_list
        self._update_src_and_sens()
        return self

    def _update_src_and_sens(self):
        # pylint: disable=protected-access
        """updates source and sensor list when an object is added or removed"""
        self._sources = [
            obj for obj in self._objects if obj._object_type in LIBRARY_SOURCES
        ]
        self._sensors = [
            obj for obj in self._objects if obj._object_type in LIBRARY_SENSORS
        ]

    def remove(self, obj):
        """
        Remove a specific object from the Collection.

        Parameters
        ----------
        object: object object
            Remove the given object from the Collection.

        Returns
        -------
        self: Collection

        Examples
        --------
        Remove a specific object from a Collection:

        >>> import magpylib as magpy
        >>> src1 = magpy.current.Loop(1,1)
        >>> src2 = magpy.current.Loop(1,1)
        >>> col = src1 + src2
        >>> print(col.objects)
        [Loop(id=2405009623360), Loop(id=2405010235504)]
        >>> col.remove(src1)
        >>> print(col.objects)
        [Loop(id=2405010235504)]

        """
        self._objects.remove(obj)
        self._update_src_and_sens()
        return self

    def copy(self):
        """
        Returns a copy of the Collection.

        Returns
        -------
        self: Collection

        Examples
        --------
        Create a copy of a Collection object:

        >>> import magpylib as magpy
        >>> col = magpy.Collection()
        >>> print(id(col))
        2221754911040
        >>> col2 = col.copy()
        >>> print(id(col2))
        2221760504160

        """
        return copy.copy(self)

    def reset_path(self):
        """
        Reset all object paths to position = (0,0,0) and orientation = unit rotation.

        Returns
        -------
        self: Collection

        Examples
        --------
        Create a collection with non-zero paths

        >>> import magpylib as magpy
        >>> dipole = magpy.misc.Dipole((1,2,3), position=(1,2,3))
        >>> loop = magpy.current.Loop(1,1, position=[(1,1,1)]*2)
        >>> col = loop + dipole
        >>> for src in col:
        >>>     print(src.position)
        [[1. 1. 1.]  [1. 1. 1.]]
        [1. 2. 3.]
        >>> col.reset_path()
        >>> for src in col:
        >>>     print(src.position)
        [0. 0. 0.]
        [0. 0. 0.]
        """
        for obj in self:
            obj.reset_path()
        return self

    def set_styles(self, arg=None, **kwargs):
        """
        Set display style of all objects in the Collection. Only matching properties
        will be applied. Input can be a style-dict or style-underscore_magic.

        Returns
        -------
        self

        Examples
        --------
        Apply a style to all objects inside a Collection using a style-dict or
        style-underscore_magic.

        >>> import magpylib as magpy
        >>>
        >>> # create collection
        >>> col = magpy.Collection()
        >>> for i in range(3):
        >>>     col + magpy.magnet.Sphere((1,1,1), 1, (i,0,0))
        >>>
        >>> # separate object
        >>> src = magpy.magnet.Sphere((1,1,1), 1, (3,0,0))
        >>>
        >>> # set collection style
        >>> col.set_styles(color='g')
        >>>
        >>> # set collection style with style-dict
        >>> style_dict = {'magnetization_size':0.5}
        >>> col.set_styles(style_dict)
        >>>
        >>> magpy.display(col, src)
        ---> graphic output
        """

        if arg is None:
            arg = {}
        if kwargs:
            arg.update(kwargs)
        style_kwargs = validate_style_keys(arg)
        for src in self._objects:
            # match properties false will try to apply properties from kwargs only if it finds it
            # withoug throwing an error
            style_kwargs_specific = {
                k: v
                for k, v in style_kwargs.items()
                if k.split("_")[0] in src.style.as_dict()
            }
            src.style.update(**style_kwargs_specific, _match_properties=True)
        return self

    def _validate_getBH_inputs(self, *objects):
        # pylint: disable=too-many-branches
        """validate Collection.getBH inputs"""
        # pylint: disable=protected-access
        sources, sensors = list(self._sources), list(self._sensors)
        if self._sensors and self._sources:
            sources, sensors = self, self
            if objects:
                raise MagpylibBadUserInput(
                    "No inputs allowed for a Mixed Collection, "
                    "since it already has Sensors and Sources"
                )
        elif not sources:
            sources, sensors = objects, self
        elif not sensors:
            sources, sensors = self, objects
        return sources, sensors

    def getB(self, *objects, sumup=False, squeeze=True):
        """
        Compute B-field in [mT] for given sources and observers.

        Parameters
        ----------
        objects: source or observer objects
            If parent is a SourceCollection, input can only be M observers.
            If parent is a SensorCollection, input can only be L sources.

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            B-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [mT]. Paths of objects that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------
        """

        sources, sensors = self._validate_getBH_inputs(*objects)

        return getBH_level2(True, sources, sensors, sumup, squeeze)

    def getH(self, *objects, sumup=False, squeeze=True):
        """
        Compute H-field in [kA/m] for given sources and observers.

        Parameters
        ----------
        objects: source or observer objects
            If parent is a SourceCollection, input can only be M observers.
            If parent is a SensorCollection, input can only be L sources.

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            H-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [kA/m]. Paths of objects that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------
        """

        sources, sensors = self._validate_getBH_inputs(*objects)

        return getBH_level2(False, sources, sensors, sumup, squeeze)


class Collection(BaseCollection, BaseTransform):
    """
    Group multiple objects in one Collection for common manipulation.

    Operations applied to a Collection are sequentially applied to all objects in the Collection.
    Collections do not allow duplicate objects (will be eliminated automatically).

    Collections have the following dunders defined: __add__, __sub__, __iter__, __getitem__,
    __repr__.

    Depending on the input, Collection objects can be sources, observers or both. A Collection
    with only source objects will become either a SourceCollection, one with only Sensors a
    SensorCollection, and one with sources and sensors a MixedCollection
    A SourceCollection functions like any SINGLE source. A SensorCollection functions like a
    list of observer inputs. A MixedCollection will function as source or as observer.

    Parameters
    ----------
    objects: sources, sensors, collections or arbitrary lists thereof
        Ordered list of objects in the Collection.

    Returns
    -------
    Collection object: Collection

    Examples
    --------

    Create Collections for common manipulation. All objects added to a Collection
    are stored in the ``objects`` attribute, and additionally in the ``sensors``
    and ``sources`` attributes. These three return ordered sets (lists with
    unique elements only)

    >>> import magpylib as magpy
    >>> sphere = magpy.magnet.Sphere((1,2,3),1)
    >>> loop = magpy.current.Loop(1,1)
    >>> dipole = magpy.misc.Dipole((1,2,3))
    >>> col = magpy.Collection(sphere, loop, dipole)
    >>> print(col.objects)
    [Sphere(id=1879891544384), Loop(id=1879891543040), Dipole(id=1879892157152)]

    Cycle directly through the Collection ``objects`` attribute

    >>> for src in col:
    >>>    print(src)
    Sphere(id=1879891544384)
    Loop(id=1879891543040)
    Dipole(id=1879892157152)

    and directly access objects from the Collection

    >>> print(col[1])
    Loop(id=1879891543040)

    Add and subtract objects to form a Collection and to remove objects from a Collection.

    >>> col = sphere + loop
    >>> print(col.objects)
    [Sphere(id=1879891544384), Loop(id=1879891543040)]
    >>> col - sphere
    >>> print(col.objects)
    [Loop(id=1879891543040)]

    Manipulate all objects in a Collection directly using ``move`` and ``rotate`` methods

    >>> import magpylib as magpy
    >>> sphere = magpy.magnet.Sphere((1,2,3),1)
    >>> loop = magpy.current.Loop(1,1)
    >>> col = sphere + loop
    >>> col.move((1,1,1))
    >>> print(sphere.position)
    [1. 1. 1.]

    and compute the total magnetic field generated by the Collection.

    >>> B = col.getB((1,2,3))
    >>> print(B)
    [-0.00372678  0.01820438  0.03423079]

    Consider three collections, a SourceCollection sCol a SensorCollection xCol and a
    MixedCollection mCol, all made up from the same objects.

    >>> import numpy as np
    >>> import magpylib as magpy

    >>> s1=magpy.magnet.Sphere((1,2,3), 1)
    >>> s2=magpy.magnet.Cylinder((1,2,3), (1,1), (3,0,0))
    >>> s3=magpy.magnet.Cuboid((1,2,3), (1,1,1), (6,0,0))

    >>> x1=magpy.Sensor((1,0,3))
    >>> x2=magpy.Sensor((4,0,3))
    >>> x3=magpy.Sensor((7,0,3))

    >>> sCol = magpy.Collection(s1, s2, s3)
    >>> xCol = magpy.Collection(x1, x2, x3)
    >>> mCol = magpy.Collection(sCol, xCol)

    All the following lines will all give the same output

    >>> magpy.getB([s1,s2,s3], [x1,x2,x3], sumup=True)
    >>> magpy.getB(sCol, xCol)
    >>> magpy.getB(mCol, mCol)
    >>> sCol.getB(xCol)
    >>> xCol.getB(sCol)
    >>> sCol.getB(mCol)
    >>> xCol.getB(mCol)
    >>> mCol.getB()
    """

    def __init__(self, *objects):
        super().__init__(*objects)