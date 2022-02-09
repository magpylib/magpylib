"""Collection class code"""
from magpylib._src.utility import (
    format_obj_input,
    check_duplicates,
    LIBRARY_SENSORS,
    LIBRARY_SOURCES,
)

from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.defaults.defaults_utility import validate_style_keys
from magpylib._src.exceptions import MagpylibBadUserInput


class BaseCollection(BaseDisplayRepr):
    """
    Group multiple children in one Collection for common manipulation.

    Operations applied to a Collection are sequentially applied to all children in the Collection.
    Collections do not allow duplicate children (will be eliminated automatically).

    Collections have the following dunders defined: __add__, __sub__, __iter__, __getitem__,
    __repr__.

    Depending on the input, Collection children can be sources, observers or both. A Collection
    with only source children will become either a SourceCollection, one with only Sensors a
    SensorCollection, and one with sources and sensors a MixedCollection
    A SourceCollection functions like any SINGLE source. A SensorCollection functions like a
    list of observer inputs. A MixedCollection will function as source or as observer.

    Parameters
    ----------
    children: sources, sensors, collections or arbitrary lists thereof
        Ordered list of children in the Collection.

    Returns
    -------
    Collection object: Collection

    Examples
    --------

    Create Collections for common manipulation. All children added to a Collection
    are stored in the ``children`` attribute, and additionally in the ``sensors``
    and ``sources`` attributes. These three return ordered sets (lists with
    unique elements only)

    >>> import magpylib as magpy
    >>> sphere = magpy.magnet.Sphere((1,2,3),1)
    >>> loop = magpy.current.Loop(1,1)
    >>> dipole = magpy.misc.Dipole((1,2,3))
    >>> col = magpy.Collection(sphere, loop, dipole)
    >>> print(col.children)
    [Sphere(id=1879891544384), Loop(id=1879891543040), Dipole(id=1879892157152)]

    Cycle directly through the Collection ``children`` attribute

    >>> for src in col:
    >>>    print(src)
    Sphere(id=1879891544384)
    Loop(id=1879891543040)
    Dipole(id=1879892157152)

    and directly access children from the Collection

    >>> print(col[1])
    Loop(id=1879891543040)

    Add and subtract children to form a Collection and to remove children from a Collection.

    >>> col = sphere + loop
    >>> print(col.children)
    [Sphere(id=1879891544384), Loop(id=1879891543040)]
    >>> col - sphere
    >>> print(col.children)
    [Loop(id=1879891543040)]

    Consider three collections, a SourceCollection sCol a SensorCollection xCol and a
    MixedCollection mCol, all made up from the same children.

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

    def __init__(self, *children):

        self._object_type = "Collection"

        # init inheritance
        BaseDisplayRepr.__init__(self)

        # instance attributes
        self._children = []
        self._sources = []
        self._sensors = []
        self.children = children

    # property getters and setters
    @property
    def children(self):
        """Collection children attribute getter and setter."""
        return self._children

    @children.setter
    def children(self, children):
        """Set Collection children."""
        obj_list = format_obj_input(children, allow="sources+sensors")
        self._children = []
        self.add(obj_list)

    @property
    def sources(self):
        """Collection sources attribute getter and setter."""
        return self._sources

    @sources.setter
    def sources(self, sources):
        """Set Collection sources."""
        src_list = format_obj_input(sources, allow="sources")
        self._children = [o for o in self._children if o not in self._sources]
        self.add(src_list)

    @property
    def sensors(self):
        """Collection sensors attribute getter and setter."""
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        """Set Collection sensors."""
        sens_list = format_obj_input(sensors, allow="sensors")
        self._children = [o for o in self._children if o not in self._sensors]
        self.add(sens_list)

    # dunders
    def __sub__(self, obj):
        return self.remove(obj)

    def __iter__(self):
        yield from self._children

    def __getitem__(self, i):
        return self._children[i]

    def __len__(self):
        return len(self._children)

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
    def add(self, *children):
        """
        Add arbitrary Magpylib children or Collections.

        Parameters
        ----------
        children: sources, Sensors, Collections or arbitrary lists thereof
            Add arbitrary sequences of children and Collections to the Collection.
            The new children will be added at the end of self.children. Duplicates
            will be eliminated.

        Returns
        -------
        self: Collection

        Examples
        --------

        Add children to a Collection:

        >>> import magpylib as magpy
        >>> src = magpy.current.Loop(1,1)
        >>> col = magpy.Collection()
        >>> col.add(src)
        >>> print(col.children)
        [Loop(id=2519738714432)]

        """
        # format input
        obj_list = format_obj_input(children)
        # combine with original obj_list
        obj_list = self._children + obj_list
        # check and eliminate duplicates
        obj_list = check_duplicates(obj_list)
        # set attributes
        self._children = obj_list
        self._update_src_and_sens()
        return self

    def _update_src_and_sens(self):
        # pylint: disable=protected-access
        """updates source and sensor list when a child is added or removed"""
        self._sources = [
            obj for obj in self._children if obj._object_type in LIBRARY_SOURCES
        ]
        self._sensors = [
            obj for obj in self._children if obj._object_type in LIBRARY_SENSORS
        ]

    def remove(self, child):
        """
        Remove a specific child from the Collection.

        Parameters
        ----------
        child: child object
            Remove the given child from the Collection.

        Returns
        -------
        self: Collection

        Examples
        --------
        Remove a specific child from a Collection:

        >>> import magpylib as magpy
        >>> src1 = magpy.current.Loop(1,1)
        >>> src2 = magpy.current.Loop(1,1)
        >>> col = src1 + src2
        >>> print(col.children)
        [Loop(id=2405009623360), Loop(id=2405010235504)]
        >>> col.remove(src1)
        >>> print(col.children)
        [Loop(id=2405010235504)]

        """
        self._children.remove(child)
        self._update_src_and_sens()
        return self


    def set_children_styles(self, arg=None, **kwargs):
        """
        Set display style of all children in the Collection. Only matching properties
        will be applied. Input can be a style-dict or style-underscore_magic.

        Returns
        -------
        self

        Examples
        --------
        Apply a style to all children inside a Collection using a style-dict or
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
        >>> col.set_children_styles(color='g')
        >>>
        >>> # set collection style with style-dict
        >>> style_dict = {'magnetization_size':0.5}
        >>> col.set_children_styles(style_dict)
        >>>
        >>> magpy.show(col, src)
        ---> graphic output
        """

        if arg is None:
            arg = {}
        if kwargs:
            arg.update(kwargs)
        style_kwargs = validate_style_keys(arg)
        for src in self._children:
            # match properties false will try to apply properties from kwargs only if it finds it
            # withoug throwing an error
            style_kwargs_specific = {
                k: v
                for k, v in style_kwargs.items()
                if k.split("_")[0] in src.style.as_dict()
            }
            src.style.update(**style_kwargs_specific, _match_properties=True)
        return self

    def _validate_getBH_inputs(self, *children):
        # pylint: disable=too-many-branches
        """validate Collection.getBH inputs"""
        # pylint: disable=protected-access
        sources, sensors = list(self._sources), list(self._sensors)
        if self._sensors and self._sources:
            sources, sensors = self, self
            if children:
                raise MagpylibBadUserInput(
                    "No inputs allowed for a Mixed Collection, "
                    "since it already has Sensors and Sources"
                )
        elif not sources:
            sources, sensors = children, self
        elif not sensors:
            sources, sensors = self, children
        return sources, sensors

    def getB(self, *children, squeeze=True):
        """
        Compute B-field in [mT] for given sources and observers.

        Parameters
        ----------
        children: source or observer children
            If parent is a SourceCollection, input can only be M observers.
            If parent is a SensorCollection, input can only be L sources.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            B-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [mT]. Paths of children that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------
        """

        sources, sensors = self._validate_getBH_inputs(*children)

        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field='B')

    def getH(self, *children, squeeze=True):
        """
        Compute H-field in [kA/m] for given sources and observers.

        Parameters
        ----------
        children: source or observer children
            If parent is a SourceCollection, input can only be M observers.
            If parent is a SensorCollection, input can only be L sources.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            H-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [kA/m]. Paths of children that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------
        """

        sources, sensors = self._validate_getBH_inputs(*children)

        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field='H')


class Collection(BaseGeo, BaseCollection):
    """
    Group multiple children in one Collection for common manipulation.

    Operations applied to a Collection are sequentially applied to all children in the Collection.
    Collections do not allow duplicate children (will be eliminated automatically).

    Collections have the following dunders defined: __add__, __sub__, __iter__, __getitem__,
    __repr__.

    Depending on the input, Collection children can be sources, observers or both. A Collection
    with only source children will become either a SourceCollection, one with only Sensors a
    SensorCollection, and one with sources and sensors a MixedCollection
    A SourceCollection functions like any SINGLE source. A SensorCollection functions like a
    list of observer inputs. A MixedCollection will function as source or as observer.

    Parameters
    ----------
    children: sources, sensors, collections or arbitrary lists thereof
        Ordered list of children in the Collection.

    Returns
    -------
    Collection object: Collection

    Examples
    --------

    Create Collections for common manipulation. All children added to a Collection
    are stored in the ``children`` attribute, and additionally in the ``sensors``
    and ``sources`` attributes. These three return ordered sets (lists with
    unique elements only)

    >>> import magpylib as magpy
    >>> sphere = magpy.magnet.Sphere((1,2,3),1)
    >>> loop = magpy.current.Loop(1,1)
    >>> dipole = magpy.misc.Dipole((1,2,3))
    >>> col = magpy.Collection(sphere, loop, dipole)
    >>> print(col.children)
    [Sphere(id=1879891544384), Loop(id=1879891543040), Dipole(id=1879892157152)]

    Cycle directly through the Collection ``children`` attribute

    >>> for src in col:
    >>>    print(src)
    Sphere(id=1879891544384)
    Loop(id=1879891543040)
    Dipole(id=1879892157152)

    and directly access children from the Collection

    >>> print(col[1])
    Loop(id=1879891543040)

    Add and subtract children to form a Collection and to remove children from a Collection.

    >>> col = sphere + loop
    >>> print(col.children)
    [Sphere(id=1879891544384), Loop(id=1879891543040)]
    >>> col - sphere
    >>> print(col.children)
    [Loop(id=1879891543040)]

    Manipulate all children in a Collection directly using ``move`` and ``rotate`` methods

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
    MixedCollection mCol, all made up from the same children.

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

    def __init__(
        self, *args, position=(0.0, 0.0, 0.0), orientation=None, style=None, **kwargs
    ):
        BaseGeo.__init__(
            self, position=position, orientation=orientation, style=style, **kwargs
        )
        BaseCollection.__init__(self, *args)
