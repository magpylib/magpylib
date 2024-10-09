"""Collection class code"""

# pylint: disable=redefined-builtin
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-positional-arguments

from collections import Counter

from magpylib._src.defaults.defaults_utility import validate_style_keys
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_wrap_BH import getBH_level2
from magpylib._src.input_checks import check_format_input_obj
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import rec_obj_remover


def repr_obj(obj, format="type+id+label"):
    """
    Returns a string that describes the object depending on the chosen tag format.
    """
    # pylint: disable=protected-access
    show_type = "type" in format
    show_label = "label" in format
    show_id = "id" in format

    tag = ""
    if show_type:
        tag += f"{type(obj).__name__}"

    if show_label:
        if show_type:
            tag += " "
        label = getattr(getattr(obj, "style", None), "label", None)
        if label is None:
            label = "nolabel" if show_type else f"{type(obj).__name__}"
        tag += label

    if show_id:
        if show_type or show_label:
            tag += " "
        tag += f"(id={id(obj)})"
    return tag


def collection_tree_generator(
    obj,
    format="type+id+label",
    max_elems=20,
    prefix="",
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
):
    """
    Recursively creates a generator that will yield a visual tree structure of
    a collection object and all its children.
    """
    # pylint: disable=protected-access

    # store children and properties of this branch
    contents = []

    children = getattr(obj, "children", [])
    if len(children) > max_elems:  # replace with counter if too many
        counts = Counter([type(c).__name__ for c in children])
        children = [f"{v}x {k}s" for k, v in counts.items()]

    props = []
    view_props = "properties" in format
    if view_props:
        desc = getattr(obj, "_get_description", False)
        if desc:
            desc_out = desc(
                exclude=(
                    "children",
                    "parent",
                    "style",
                    "field_func",
                    "sources",
                    "sensors",
                    "collections",
                    "children_all",
                    "sources_all",
                    "sensors_all",
                    "collections_all",
                )
            )
            props = [d.strip() for d in desc_out[1:]]

    contents.extend(props)
    contents.extend(children)

    # generate and store "pointer" structure for this branch
    pointers = [tee] * (len(contents) - 1) + [last]
    pointers[: len(props)] = [branch if children else space] * len(props)

    # create branch entries
    for pointer, child in zip(pointers, contents):
        child_repr = child if isinstance(child, str) else repr_obj(child, format)
        yield prefix + pointer + child_repr

        # recursion
        has_child = getattr(child, "children", False)
        if has_child or (view_props and desc):
            # space because last, └── , above so no more |
            extension = branch if pointer == tee else space

            yield from collection_tree_generator(
                child,
                format=format,
                max_elems=max_elems,
                prefix=prefix + extension,
                space=space,
                branch=branch,
                tee=tee,
                last=last,
            )


class BaseCollection(BaseDisplayRepr):
    """Collection base class without BaseGeo properties"""

    get_trace = None

    def __init__(self, *children, override_parent=False):
        BaseDisplayRepr.__init__(self)

        self._children = []
        self._sources = []
        self._sensors = []
        self._collections = []
        self.add(*children, override_parent=override_parent)

    # property getters and setters
    @property
    def children(self):
        """An ordered list of top level child objects."""
        return self._children

    @children.setter
    def children(self, children):
        """Set Collection children."""
        # pylint: disable=protected-access
        for child in self._children:
            child._parent = None
        self._children = []
        self.add(*children, override_parent=True)

    @property
    def children_all(self):
        """An ordered list of all child objects in the collection tree."""
        return check_format_input_obj(self, "collections+sensors+sources")

    @property
    def sources(self):
        """An ordered list of top level source objects."""
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
        self.add(*src_list, override_parent=True)

    @property
    def sources_all(self):
        """An ordered list of all source objects in the collection tree."""
        return check_format_input_obj(self, "sources")

    @property
    def sensors(self):
        """An ordered list of top level sensor objects."""
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
        self.add(*sens_list, override_parent=True)

    @property
    def sensors_all(self):
        """An ordered list of all sensor objects in the collection tree."""
        return check_format_input_obj(self, "sensors")

    @property
    def collections(self):
        """An ordered list of top level collection objects."""
        return self._collections

    @collections.setter
    def collections(self, collections):
        """Set Collection collections."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._collections:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        coll_list = format_obj_input(collections, allow="collections")
        self.add(*coll_list, override_parent=True)

    @property
    def collections_all(self):
        """An ordered list of all collection objects in the collection tree."""
        return check_format_input_obj(self, "collections")

    # dunders
    def __iter__(self):
        yield from self._children

    def __getitem__(self, i):
        return self._children[i]

    def __len__(self):
        return len(self._children)

    def _repr_html_(self):
        lines = []
        lines.append(repr_obj(self))
        for line in collection_tree_generator(
            self,
            format="type+label+id",
            max_elems=10,
        ):
            lines.append(line)
        return f"""<pre>{'<br>'.join(lines)}</pre>"""

    def describe(self, format="type+label+id", max_elems=10, return_string=False):
        # pylint: disable=arguments-differ
        """Returns or prints a tree view of the collection.

        Parameters
        ----------
        format: bool, default='type+label+id'
            Object description in tree view. Can be any combination of `'type'`, `'label'`
            and `'id'` and `'properties'`.
        max_elems: default=10
            If number of children at any level is higher than `max_elems`, elements are
            replaced by counters.
        return_string: bool, default=`False`
            If `False` print description with stdout, if `True` return as string.
        """
        tree = collection_tree_generator(
            self,
            format=format,
            max_elems=max_elems,
        )
        output = [repr_obj(self, format)]
        for t in tree:
            output.append(t)
        output = "\n".join(output)

        if return_string:
            return output
        print(output)
        return None

    # methods -------------------------------------------------------
    def add(self, *children, override_parent=False):
        """Add sources, sensors or collections.

        Parameters
        ----------
        children: sources, sensors or collections
            Add arbitrary sources, sensors or other collections to this collection.

        override_parent: bool, default=`True`
            Accept objects as children that already have parents. Automatically
            removes such objects from previous parent collection.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we add a sensor object to a collection:

        >>> import magpylib as magpy
        >>> x1 = magpy.Sensor(style_label='x1')
        >>> coll = magpy.Collection(x1, style_label='coll')
        >>> coll.describe(format='label')
        coll
        └── x1

        >>> x2 = magpy.Sensor(style_label='x2')
        >>> coll.add(x2)
        Collection(id=...)
        >>> coll.describe(format='label')
        coll
        ├── x1
        └── x2
        """
        # pylint: disable=protected-access

        # allow flat lists as input
        if len(children) == 1 and isinstance(children[0], (list, tuple)):
            children = children[0]

        # check and format input
        obj_list = check_format_input_obj(
            children,
            allow="sensors+sources+collections",
            recursive=False,
            typechecks=True,
        )

        # assign parent
        for obj in obj_list:
            if isinstance(obj, Collection):
                # no need to check recursively with `collections_all` if obj is already self
                if obj is self or self in obj.collections_all:
                    raise MagpylibBadUserInput(
                        f"Cannot add {obj!r} because a Collection must not reference itself."
                    )
            if obj._parent is None:
                obj._parent = self
            elif override_parent:
                obj._parent.remove(obj)
                obj._parent = self
            else:
                raise MagpylibBadUserInput(
                    f"Cannot add {obj!r} to {self!r} because it already has a parent.\n"
                    "Consider using `override_parent=True`."
                )

        # set attributes
        self._children += obj_list
        self._update_src_and_sens()

        return self

    def _update_src_and_sens(self):
        """updates sources, sensors and collections attributes from children"""
        # pylint: disable=protected-access
        from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
        from magpylib._src.obj_classes.class_Sensor import Sensor

        self._sources = [obj for obj in self._children if isinstance(obj, BaseSource)]
        self._sensors = [obj for obj in self._children if isinstance(obj, Sensor)]
        self._collections = [
            obj for obj in self._children if isinstance(obj, Collection)
        ]

    def remove(self, *children, recursive=True, errors="raise"):
        """Remove children from the collection tree.

        Parameters
        ----------
        children: child objects
            Remove the given children from the collection.

        recursive: bool, default=`True`
            Remove children also when they are in child collections.

        errors: str, default=`'raise'`
            Can be `'raise'` or `'ignore'` to toggle error output when child is
            not found for removal.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we remove a child from a Collection:

        >>> import magpylib as magpy
        >>> x1 = magpy.Sensor(style_label='x1')
        >>> x2 = magpy.Sensor(style_label='x2')
        >>> col = magpy.Collection(x1, x2, style_label='col')
        >>> col.describe(format='label')
        col
        ├── x1
        └── x2

        >>> col.remove(x1)
        Collection(id=...)
        >>> col.describe(format='label')
        col
        └── x2
        """
        # pylint: disable=protected-access

        # allow flat lists as input
        if len(children) == 1 and isinstance(children[0], (list, tuple)):
            children = children[0]

        # check and format input
        remove_objects = check_format_input_obj(
            children,
            allow="sensors+sources+collections",
            recursive=False,
            typechecks=True,
        )
        self_objects = check_format_input_obj(
            self,
            allow="sensors+sources+collections",
            recursive=recursive,
        )
        for child in remove_objects:
            if child in self_objects:
                rec_obj_remover(self, child)
                child._parent = None
            else:
                if errors == "raise":
                    raise MagpylibBadUserInput(
                        f"Cannot find and remove {child} from {self}."
                    )
                if errors != "ignore":
                    raise MagpylibBadUserInput(
                        "Input `errors` must be one of ('raise', 'ignore').\n"
                        f"Instead received {errors}."
                    )
        return self

    def set_children_styles(self, arg=None, recursive=True, _validate=True, **kwargs):
        """Set display style of all children in the collection. Only matching properties
        will be applied.

        Parameters
        ----------
        arg: style dictionary or style underscore magic input
            Style arguments to be applied.

        recursive: bool, default=`True`
            Apply styles also to children of child collections.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we start by creating a collection from three sphere magnets:

        >>> import magpylib as magpy
        >>>
        >>> col = magpy.Collection(
        ...     [
        ...         magpy.magnet.Sphere(position=(i, 0, 0), diameter=1, polarization=(0, 0, 0.1))
        ...         for i in range(3)
        ...     ]
        ... )
        >>> # We apply styles using underscore magic for magnetization vector size and a style
        >>> # dictionary for the color.
        >>>
        >>> col.set_children_styles(magnetization_size=0.5)
        Collection(id=...)
        >>> col.set_children_styles({"color": "g"})
        Collection(id=...)
        >>>
        >>> # Finally we create a separate sphere magnet to demonstrate the default style
        >>> # the collection and the separate magnet with Matplotlib:
        >>>
        >>> src = magpy.magnet.Sphere(position=(3, 0, 0), diameter=1, polarization=(0, 0, .1))
        >>> magpy.show(col, src) # doctest: +SKIP
        >>> # graphic output
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
            if isinstance(child, Collection) and recursive:
                self.__class__.set_children_styles(child, style_kwargs, _validate=False)
            style_kwargs_specific = {
                k: v
                for k, v in style_kwargs.items()
                if k.split("_")[0] in child.style.as_dict()
            }
            child.style.update(**style_kwargs_specific, _match_properties=True)
        return self

    def _validate_getBH_inputs(self, *inputs):
        """
        select correct sources and observers for getBHJM_level2
        """
        # pylint: disable=protected-access
        # pylint: disable=too-many-branches
        # pylint: disable=possibly-used-before-assignment
        current_sources = format_obj_input(self, allow="sources")
        current_sensors = format_obj_input(self, allow="sensors")

        # if collection includes source and observer objects, select itself as
        #   source and observer in gethBH
        if current_sensors and current_sources:
            sources, sensors = self, self
            if inputs:
                raise MagpylibBadUserInput(
                    "Collections with sensors and sources do not allow `collection.getB()` inputs."
                    "Consider using `magpy.getB()` instead."
                )
        # if collection has no sources, *inputs must be the sources
        elif not current_sources:
            sources, sensors = inputs, self

        # if collection has no sensors, *inputs must be the observers
        elif not current_sensors:
            if len(inputs) == 1:
                sources, sensors = self, inputs[0]
            else:
                sources, sensors = self, inputs

        return sources, sensors

    def getB(
        self, *inputs, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute B-field for given sources and observers.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        inputs: source or observer objects
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains only sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            B-field at each path position ( index m) for each sensor (index k) and each
            sensor pixel position (indices n1,n2,...) in units of T. Sensor pixel positions
            are equivalent to simple observer positions. Paths of objects that are shorter
            than index m are considered as static beyond their end.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere(polarization=(0,0,1.), diameter=1)
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
        [[[0.         0.         0.08333333]
          [0.         0.         0.08333333]]
        <BLANKLINE>
         [[0.         0.         0.08333333]
          [0.         0.         0.08333333]]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getBH_level2(
            sources,
            sensors,
            field="B",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getH(
        self, *inputs, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute H-field for given sources and observers.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        inputs: source or observer objects
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        Returns
        -------
        H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            H-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of A/m. Sensor pixel positions are
            equivalent to simple observer positions. Paths of objects that are shorter than
            index m are considered as static beyond their end.

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere(polarization=(0,0,1.), diameter=1)
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
        [[[    0.             0.         66314.55958552]
          [    0.             0.         66314.55958552]]
        <BLANKLINE>
         [[    0.             0.         66314.55958552]
          [    0.             0.         66314.55958552]]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)

        return getBH_level2(
            sources,
            sensors,
            field="H",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getM(
        self, *inputs, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute M-field for given sources and observers.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        inputs: source or observer objects
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        Returns
        -------
        M-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            M-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of A/m. Sensor pixel positions are
            equivalent to simple observer positions. Paths of objects that are shorter than
            index m are considered as static beyond their end.

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)

        return getBH_level2(
            sources,
            sensors,
            field="M",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getJ(
        self, *inputs, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute J-field for given sources and observers.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        inputs: source or observer objects
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains only sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a
            `numpy.ndarray` object is returned. If 'dataframe' is chosen, a `pandas.DataFrame`
            object is returned (the Pandas library must be installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        J-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            J-field at each path position ( index m) for each sensor (index k) and each
            sensor pixel position (indices n1,n2,...) in units of T. Sensor pixel positions
            are equivalent to simple observer positions. Paths of objects that are shorter
            than index m are considered as static beyond their end.
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)

        return getBH_level2(
            sources,
            sensors,
            field="J",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        items = []
        if self.children_all:
            nums = {
                "sensor": len(self.sensors_all),
                "source": len(self.sources_all),
            }
            for name, num in nums.items():
                if num > 0:
                    items.append(f"{num} {name}{'s'[:num^1]}")
        else:
            items.append("no children")
        return ", ".join(items)


class Collection(BaseGeo, BaseCollection):
    """Group multiple children (sources, sensors and collections) in a collection for
    common manipulation.

    Collections span a local reference frame. All objects in a collection are held to
    that reference frame when an operation (e.g. move, rotate, setter, ...) is applied
    to the collection.

    Collections can be used as `sources` and `observers` input for magnetic field
    computation. For magnetic field computation a collection that contains sources
    functions like a single source. When the collection contains sensors
    it functions like a list of all its sensors.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    children: sources, `Sensor` or `Collection` objects
        An ordered list of all children in the collection.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    override_parent: bool, default=False
        If False thrown an error when an attempt is made to add an object that
        has already a parent to a Collection. If True, allow adding the object
        and override the objects parent attribute thus removing it from its
        previous collection.

    sensors: `Sensor` objects
        An ordered list of all sensor objects in the collection.

    sources: `Source` objects
        An ordered list of all source objects (magnets, currents, misc) in the collection.

    collections: `Collection` objects
        An ordered list of all collection objects in the collection.


    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

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
    >>> src1 = magpy.magnet.Sphere(position=(2,0,0), diameter=1,polarization=(.1,.2,.3))
    >>> src2 = magpy.current.Circle(position=(-2,0,0), diameter=1, current=1)
    >>> col = magpy.Collection(src1, src2)
    >>> col.move(((0,0,2)))
    Collection(id=...)
    >>> print(src1.position)
    [2. 0. 2.]
    >>> print(src2.position)
    [-2.  0.  2.]
    >>> print(col.position)
    [0. 0. 2.]

    We can still directly access individual objects by name and by index:

    >>> src1.move((2,0,0))
    Sphere(id=...)
    >>> col[1].move((-2,0,0))
    Circle(id=...)
    >>> print(src1.position)
    [4. 0. 2.]
    >>> print(src2.position)
    [-4.  0.  2.]
    >>> print(col.position)
    [0. 0. 2.]

    The field can be computed at position (0,0,0) as if the collection was a single source:

    >>> B = col.getB((0,0,0))
    >>> print(B)
    [ 2.32922681e-04 -9.31694991e-05 -3.44484717e-10]

    We add a sensor at position (0,0,0) to the collection:

    >>> sens = magpy.Sensor()
    >>> col.add(sens)
    Collection(id=...)
    >>> print(col.children)
    [Sphere(id=...), Circle(id=...), Sensor(id=...)]

    and can compute the field of the sources in the collection seen by the sensor with
    a single command:

    >>> B = col.getB()
    >>> print(B)
    [ 2.32922681e-04 -9.31694991e-05 -3.44484717e-10]
    """

    def __init__(
        self,
        *args,
        position=(0, 0, 0),
        orientation=None,
        override_parent=False,
        style=None,
        **kwargs,
    ):
        BaseGeo.__init__(
            self,
            position=position,
            orientation=orientation,
            style=style,
            **kwargs,
        )
        BaseCollection.__init__(self, *args, override_parent=override_parent)
