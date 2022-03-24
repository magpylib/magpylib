"""Collection class code"""

from collections import Counter

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


def repr_obj(obj, labels=False):
    """Returns obj repr based on label"""
    if labels and obj.style.label:
        return f"{obj.style.label}"
    return f"{obj!r}"


def collection_tree_generator(
    dir_child,
    prefix="",
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
    labels=False,
    max_elems=20,
):
    """A recursive generator, given a collection child object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    # pylint: disable=protected-access
    contents = getattr(dir_child, "children", [])
    if len(contents) > max_elems:
        counts = Counter([c._object_type for c in contents])
        contents = [f"{v}x {k}s" for k, v in counts.items()]
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, child in zip(pointers, contents):
        child_repr = child if isinstance(child, str) else repr_obj(child, labels)
        yield prefix + pointer + child_repr
        if getattr(child, "children", False):  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from collection_tree_generator(
                child,
                prefix=prefix + extension,
                labels=labels,
                max_elems=max_elems,
            )


class BaseCollection(BaseDisplayRepr):
    """ Collection base class without BaseGeo properties
    """

    def __init__(self, *children, override_parent=False):

        self._object_type = "Collection"

        BaseDisplayRepr.__init__(self)

        self._children = []
        self._sources = []
        self._sensors = []
        self._collections = []
        self.add(*children, override_parent=override_parent)

    # property getters and setters
    @property
    def children(self):
        """Collection children attribute getter and setter."""
        return self._children

    @children.setter
    def children(self, children):
        """Set Collection children."""
        # pylint: disable=protected-access
        for child in self._children:
            child._parent = None
        self._children = []
        self.add(*children)

    @property
    def sources(self):
        """Collection sources attribute getter and setter."""
        return self._sources

    @sources.setter
    def sources(self, sources):
        """Set Collection sources."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._sources:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        src_list = format_obj_input(sources, allow="sources")
        self.add(src_list)

    @property
    def sensors(self):
        """Collection sensors attribute getter and setter."""
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        """Set Collection sensors."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._sensors:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        sens_list = format_obj_input(sensors, allow="sensors")
        self.add(sens_list)

    @property
    def collections(self):
        """Collection sub-collections attribute getter and setter."""
        return self._collections

    @collections.setter
    def collections(self, collections):
        """Set Collection sub-collections."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._collections:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        sens_list = format_obj_input(collections, allow="collections")
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
        s = super().__repr__()
        if self._children:
            if self._collections:
                pref = "Nested"
            elif not self._sources:
                pref = "Sensor"
            elif not self._sensors:
                pref = "Source"
            else:
                pref = "Mixed"
            return f"{pref}{s}"
        return s

    def describe(self, labels=False, max_elems=10):
        """Returns a tree view of the nested collection elements.

        Parameters
        ----------
        labels: bool, default=False
            If True, `object.style.label` is used if available, instead of `repr(object)`
        max_elems:
            If number of children at any level is higher than `max_elems`, elements are replaced by
            counters by object type.
        """
        print(repr_obj(self, labels))
        for line in collection_tree_generator(
            self, labels=labels, max_elems=max_elems
        ):
            print(line)

    # methods -------------------------------------------------------
    def add(self, *children, override_parent=False):
        """Add sources, sensors or collections.

        Parameters
        ----------
        children: source, `Sensor` or `Collection` objects or arbitrary lists thereof
            Add arbitrary sources, sensors or other collections to this collection.
            Duplicate children will automatically be eliminated.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we add a sensor object to a collection:

        >>> import magpylib as magpy
        >>> col = magpy.Collection()
        >>> sens = magpy.Sensor()
        >>> col.add(sens)
        >>> print(col.children)
        [Sensor(id=2236606343584)]
        """
        # pylint: disable=protected-access
        # format input
        obj_list = format_obj_input(children, allow="sensors+sources+collections")
        # check and eliminate duplicates
        obj_list = check_duplicates(obj_list)
        # assign parent
        for obj in obj_list:
            if obj._parent is None or override_parent:
                obj._parent = self
            else:
                raise ValueError(
                    f"`{self!r}` cannot receive `{obj!r}`, as the child already has a parent "
                    f"(`{obj._parent!r}`). "
                    "You can use the `.add(*children, override_parent=True)` method to ignore and "
                    "override the current object parent. Note that this will remove the object "
                    "from the previous parent collection."
                )
        # combine with original obj_list
        obj_list = self._children + obj_list
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
        self._collections = [
            obj for obj in self._children if obj._object_type == "Collection"
        ]

    @staticmethod
    def _remove(parent, child, recursive=True, issearching=False):
        """Remove a specific child from a parent collection.

        Parameters
        ----------
        parent: parent collection object

        child: child object

        recursive: bool, default=True
            If True, the method will also search in lower nested levels

        issearching: bool, default=False
            Tells the current searching status over the nested collection. This avoids the search,
            to be stopped to early if the child has not been found yet
        """
        # pylint: disable=protected-access
        isfound = False
        if child in parent._children or not recursive:
            parent._children.remove(child)
            child._parent = None
            isfound = True
        else:
            for child_col in parent._collections:
                isfound = parent.__class__._remove(
                    child_col, child, recursive=True, issearching=True
                )
                if isfound:
                    break
        if issearching:
            return isfound
        if isfound:
            parent._update_src_and_sens()
        elif not isfound:
            raise ValueError(f"""{parent}.remove({child}) : {child!r} not found.""")

    def remove(self, child, recursive=True):
        """Remove a specific child from the collection.

        Parameters
        ----------
        child: child object
            Remove the given child from the collection.

        recursive: bool, default=True
            If True, the method will also search in lower nested levels

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we remove a child from a Collection:

        >>> import magpylib as magpy
        >>> sens = magpy.Sensor()
        >>> col = magpy.Collection(sens)
        >>> print(col.children)
        [Sensor(id=2048351734560)]

        >>> col.remove(sens)
        >>> print(col.children)
        []
        """
        self._remove(self, child, recursive=recursive)
        return self

    def set_children_styles(self, arg=None, _validate=True, recursive=True, **kwargs):
        """Set display style of all children in the collection. Only matching properties
        will be applied. Input can be a style dict or style underscore magic.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we start by creating a collection from three sphere magnets:

        >>> import magpylib as magpy
        >>> col = magpy.Collection()
        >>> for i in range(3):
        >>>     col = col + magpy.magnet.Sphere((0,0,1), 1, position=(i,0,0))

        We apply styles using underscore magic for magnetization vector size and a style
        dictionary for the color.

        >>> col.set_children_styles(magnetization_size=0.5)
        >>> col.set_children_styles({'color':'g'})

        Finally we create a separate sphere magnet to demonstrate the default style and display
        the collection and the separate magnet with Matplotlib:

        >>> src = magpy.magnet.Sphere((0,0,1), 1, position=(3,0,0))
        >>> magpy.show(col, src)
        ---> graphic output
        """
        # pylint: disable=protected-access

        if arg is None:
            arg = {}
        if kwargs:
            arg.update(kwargs)
        style_kwargs = arg
        if _validate:
            style_kwargs = validate_style_keys(arg)

        for child in self._children:
            # match properties false will try to apply properties from kwargs only if it finds it
            # without throwing an error
            if child._object_type == "Collection" and recursive:
                self.__class__.set_children_styles(child, style_kwargs, _validate=False)
            style_kwargs_specific = {
                k: v
                for k, v in style_kwargs.items()
                if k.split("_")[0] in child.style.as_dict()
            }
            child.style.update(**style_kwargs_specific, _match_properties=True)
        return self

    def _validate_getBH_inputs(self, *inputs):
        """validate Collection.getBH inputs"""
        # pylint: disable=protected-access
        # pylint: disable=too-many-branches
        current_sources = format_obj_input(self, allow="sources")
        current_sensors = format_obj_input(self, allow="sensors")
        if current_sensors and current_sources:
            sources, sensors = self, self
            if inputs:
                raise MagpylibBadUserInput(
                    "Collections with sensors and sources do not allow `collection.getB()` inputs."
                    "Consider using `magpy.getB()` instead."
                )
        elif not current_sources:
            sources, sensors = inputs, self
        elif not current_sensors:
            sources, sensors = self, inputs
        return sources, sensors

    def getB(self, *inputs, squeeze=True):
        """Compute B-field in [mT] for given sources and observer inputs.

        Parameters
        ----------
        sources_observers: source or observer inputs
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains only sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
            B-field at each path position (m) for each sensor (k) and each sensor pixel
            position (n1,n2,...) in units of [mT]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than m will be
            considered as static beyond their end.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere((0,0,1000), 1)
        >>> src2 = src1.copy()
        >>> sens1 = magpy.Sensor(position=(0,0,1))
        >>> sens2 = sens1.copy()
        >>> col = src1 + src2 + sens1 + sens2

        The following computations all give the same result:

        >>> B = col.getB()
        >>> B = magpy.getB(col, col)
        >>> B = magpy.getB(col, [sens1, sens2])
        >>> B = magpy.getB([src1, src2], col)
        >>> B = magpy.getB([src1, src2], [sens1, sens2])
        >>> print(B)
        [[  0.           0.         166.66666667]
         [  0.           0.         166.66666667]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field="B")

    def getH(self, *inputs, squeeze=True):
        """Compute H-field in [kA/m] for given sources and observer inputs.

        Parameters
        ----------
        sources_observers: source or observer inputs
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
            H-field at each path position (m) for each sensor (k) and each sensor pixel
            position (n1,n2,...) in units of [kA/m]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than m will be
            considered as static beyond their end.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere((0,0,1000), 1)
        >>> src2 = src1.copy()
        >>> sens1 = magpy.Sensor(position=(0,0,1))
        >>> sens2 = sens1.copy()
        >>> col = src1 + src2 + sens1 + sens2

        The following computations all give the same result:

        >>> H = col.getH()
        >>> H = magpy.getH(col, col)
        >>> H = magpy.getH(col, [sens1, sens2])
        >>> H = magpy.getH([src1, src2], col)
        >>> H = magpy.getH([src1, src2], [sens1, sens2])
        >>> print(H)
        [[  0.           0.         66.31455962]
         [  0.           0.         66.31455962]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)

        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field="H")


class Collection(BaseGeo, BaseCollection):
    """ Group multiple children (sources and sensors) in one Collection for
    common manipulation.

    Collections can be used as `sources` and `observers` input for magnetic field
    computation. For magnetic field computation a collection that contains sources
    functions like a single (compound) source. When the collection contains sensors
    it functions like a list of all its sensors.

    Collections function like compound-objects. They have their own `position` and
    `orientation` attributes. Move, rotate and setter operations acting on a
    `Collection` object are individually applied to all child objects so that the
    geometric compound structure is maintained. For example, `rotate()` with
    `anchor=None` rotates all children about `collection.position`.

    Parameters
    ----------
    children: sources, sensors, collections or arbitrary lists thereof
        Ordered list of all children.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    collection: `Collection` object

    Examples
    --------
    Collections function as groups of multiple magpylib objects. In this example
    we create a collection with two sources and move the whole collection:

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,2,3), 1, position=(2,0,0))
    >>> src2 = magpy.current.Loop(1, 1, position=(-2,0,0))
    >>> col = magpy.Collection(src1, src2)
    >>> col.move(((0,0,2)))
    >>> print(src1.position)
    >>> print(src2.position)
    >>> print(col.position)
    [2. 0. 2.]
    [-2.  0.  2.]
    [0. 0. 2.]

    We can still directly access individual objects by name and by index:

    >>> src1.move((2,0,0))
    >>> col[1].move((-2,0,0))
    >>> print(src1.position)
    >>> print(src2.position)
    >>> print(col.position)
    [4. 0. 2.]
    [-4.  0.  2.]
    [0. 0. 2.]

    The field can be computed at position (0,0,0) as if the collection was a single source:

    B = col.getB((0,0,0))
    print(B)
    [ 0.00126232 -0.00093169 -0.00034448]

    We add a sensor at position (0,0,0) to the collection:

    >>> sens = magpy.Sensor()
    >>> col.add(sens)
    >>> print(col.children)
    [Sphere(id=2236606344304), Loop(id=2236606344256), Sensor(id=2236606343584)]

    and can compute the field of the sources in the collection seen by the sensor with
    a single command:

    B = col.getB()
    print(B)
    [ 0.00126232 -0.00093169 -0.00034448]
    """

    def __init__(
        self,
        *args,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        override_parent=False,
        **kwargs,
    ):
        BaseGeo.__init__(
            self, position=position, orientation=orientation, style=style, **kwargs,
        )
        BaseCollection.__init__(self, *args, override_parent=override_parent)
