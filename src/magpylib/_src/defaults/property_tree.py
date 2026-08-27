"""Declarative typed-property tree for style and settings classes.

Successor of ``MagicProperties``: a settings class declares each field once
as a typed descriptor,

    class Line(PropertyNode):
        style = Choice(*ALLOWED_LINESTYLES, doc="Line style.")
        color = Color(doc="Line color.")
        width = Number(minimum=0, doc="Line width.")

and validation, magic-underscore updates, dict round-tripping, schema
introspection, change observation and layered resolution all derive
generically from the declarations.

Value semantics: a field only ever stores explicitly set values; assigning
``None`` unsets it, after which reading returns the field default (usually
``None``, meaning "defer to the next defaults layer at display time").
Collections of child nodes (`NodeSequence`) are held as tuples, so they too
can only change by assignment, and every change stays validated and observed.

Public contract for GUIs and third-party tooling:

- ``schema()``: JSON-Schema description of the field tree
- ``get(path)`` / ``set(path, value)``: dotted-path access
- ``observe(callback)``: change events bubble up the tree and report the
  dotted path of the changed leaf
- ``is_set(name)`` / ``set_values()``: distinguish explicitly set values
  from deferred ones
- ``merged(*layers)``: first-set-wins resolution across defaults layers
"""

from copy import deepcopy
from typing import ClassVar

from magpylib._src.defaults.defaults_utility import (
    color_validator,
    linearize_dict,
    magic_to_dict,
)


class Property:
    """Base descriptor for a leaf field of a `PropertyNode`.

    Parameters
    ----------
    default: any
        Effective value reported while the field is not explicitly set.
    doc: str
        Single-line description, surfaced in `schema()`.
    """

    kind = "any"

    def __init__(self, default=None, doc=""):
        self.default = default
        self.doc = doc
        self.name = None

    def __set_name__(self, owner, name):
        # name validation happens in PropertyNode.__init_subclass__: exceptions
        # raised in __set_name__ get wrapped in RuntimeError before python 3.12
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)

    def __set__(self, obj, value):
        if value is None:
            obj.__dict__.pop(self.name, None)  # unset
            new = self.default
        else:
            new = obj.__dict__[self.name] = self.validate(obj, value)
        if obj._observed:
            obj._notify(self.name, new)

    def validate(self, obj, value):  # noqa: ARG002  (interface for subclasses)
        """Return the validated (possibly normalized) value or raise."""
        return value

    def _fail(self, obj, value, expected):
        msg = (
            f"Input {self.name} of {type(obj).__name__} must be {expected}; "
            f"instead received {value!r}."
        )
        raise ValueError(msg)

    def schema(self):
        """Return the JSON-Schema description of this field."""
        out = {"type": [self.json_type, "null"]} if self.json_type else {}
        if self.doc:
            out["description"] = self.doc
        if self.default is not None and not callable(self.default):
            out["default"] = self.default  # a callable has no JSON form
        return out

    json_type = None


class Number(Property):
    """A float or int field with optional bounds."""

    kind = "number"
    json_type = "number"
    _types = (int, float)

    def __init__(
        self, default=None, doc="", minimum=None, maximum=None, exclusive_minimum=None
    ):
        super().__init__(default, doc)
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum

    def validate(self, obj, value):
        if (
            not isinstance(value, self._types)
            or isinstance(value, bool)
            or (self.minimum is not None and value < self.minimum)
            or (self.maximum is not None and value > self.maximum)
            or (self.exclusive_minimum is not None and value <= self.exclusive_minimum)
        ):
            bounds = [
                f">={self.minimum}" if self.minimum is not None else "",
                f">{self.exclusive_minimum}"
                if self.exclusive_minimum is not None
                else "",
                f"<={self.maximum}" if self.maximum is not None else "",
            ]
            bounds = " and ".join(b for b in bounds if b)
            expected = f"a {self.kind}" + (f" {bounds}" if bounds else "")
            self._fail(obj, value, expected)
        return value

    def schema(self):
        out = super().schema()
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.exclusive_minimum is not None:
            out["exclusiveMinimum"] = self.exclusive_minimum
        return out


class Integer(Number):
    """An int field with optional bounds."""

    kind = "integer"
    json_type = "integer"
    _types = (int,)


class Boolean(Property):
    """A strict True/False field."""

    kind = "boolean"
    json_type = "boolean"

    def validate(self, obj, value):
        if value is not True and value is not False:
            self._fail(obj, value, "either True or False")
        return value


class String(Property):
    """A string field, optionally coercing any input via str()."""

    kind = "string"
    json_type = "string"

    def __init__(self, default=None, doc="", coerce=False):
        super().__init__(default, doc)
        self.coerce = coerce

    def validate(self, obj, value):
        if self.coerce:
            return str(value)
        if not isinstance(value, str):
            self._fail(obj, value, "a string")
        return value


class Choice(Property):
    """A field restricted to a set of allowed values.

    The values are given either directly, or as a single zero-argument
    callable returning them, for sets that are only known at call time (e.g.
    the display backends registered so far). A callable is re-evaluated on
    every `validate()` and `schema()`, so a generated schema describes the
    choices as they stand at generation time.
    """

    kind = "choice"

    def __init__(self, *choices, default=None, doc=""):
        super().__init__(default, doc)
        if len(choices) == 1 and callable(choices[0]):
            self._choices = choices[0]
        else:
            self._choices = lambda: choices

    @property
    def choices(self):
        """The currently allowed values."""
        return tuple(self._choices())

    def validate(self, obj, value):
        choices = self.choices
        if value not in choices:
            self._fail(obj, value, f"one of {choices}")
        return value

    def schema(self):
        out = super().schema()
        # tuples serialize as JSON arrays
        out["enum"] = [*self.choices, None]
        return out


class Color(Property):
    """A CSS color field, normalized by `color_validator`."""

    kind = "color"
    json_type = "string"

    def validate(self, obj, value):
        return color_validator(value, parent_name=type(obj).__name__)

    def schema(self):
        return {**super().schema(), "format": "color"}


class ColorSequence(Property):
    """A tuple of CSS colors, each normalized by `color_validator`."""

    kind = "colorsequence"

    def validate(self, obj, value):
        name = type(obj).__name__
        try:
            return tuple(
                color_validator(c, allow_None=False, parent_name=name) for c in value
            )
        except TypeError as err:
            msg = (
                f"Input {self.name} of {name} must be an "
                f"iterable of colors; instead received {value!r}."
            )
            raise ValueError(msg) from err

    def schema(self):
        out = super().schema()
        out["type"] = ["array", "null"]
        out["items"] = {"type": "string", "format": "color"}
        return out


def _adopt_child(parent, child, name):
    """Link a child node into its parent's tree under ``name`` and return it.

    Adoption is what makes a change inside the child bubble up as
    ``name.<leaf>``; a child adopted into an observed tree starts observing,
    down to its own children.
    """
    object.__setattr__(child, "_parent", parent)
    object.__setattr__(child, "_name", name)
    if parent._observed:
        child._mark_observed()
    return child


class Nested(Property):
    """A child `PropertyNode`, instantiated lazily on first access.

    Assigning a dict builds a fresh node from it (replace, not merge — use
    ``update()`` to merge); assigning ``None`` resets to a pristine node.
    With ``from_str``, assigning a string builds a node with that field set,
    e.g. ``style.description = "hello"`` for ``Nested(Description,
    from_str="text")``.
    """

    kind = "node"

    def __init__(self, node_class, doc="", from_str=None):
        super().__init__(None, doc)
        self.node_class = node_class
        self.from_str = from_str

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        child = obj.__dict__.get(self.name)
        if child is None:
            child = self._adopt(obj, self.node_class())
        return child

    def __set__(self, obj, value):
        if value is None:
            obj.__dict__.pop(self.name, None)  # pristine node on next access
        elif isinstance(value, dict):
            self._adopt(obj, self.node_class(**value))
        elif isinstance(value, str) and self.from_str:
            self._adopt(obj, self.node_class(**{self.from_str: value}))
        elif isinstance(value, self.node_class):
            self._adopt(obj, value)
        else:
            msg = (
                f"The {self.name} property of {type(obj).__name__} must be an "
                f"instance of {self.node_class.__name__} or a dictionary with "
                f"equivalent key/value pairs; instead received {value!r}."
            )
            raise TypeError(msg)
        if obj._observed:
            obj._notify(self.name, getattr(obj, self.name))

    def _adopt(self, obj, child):
        obj.__dict__[self.name] = child
        return _adopt_child(obj, child, self.name)

    def schema(self):
        out = self.node_class.schema()
        if self.doc:
            out["description"] = self.doc
        return out


class NodeSequence(Property):
    """A tuple of child `PropertyNode` objects, e.g. the 3d traces of a style.

    The stored value is always a tuple, so the collection can only change by
    assignment,

        node.field = [*node.field, extra]   # not node.field.append(extra)

    which keeps every change validated and reported to observers — both of
    which in-place mutation of a list would bypass. Elements are adopted like
    `Nested` children, so a change inside one bubbles up as
    ``<field>.<index>.<leaf>`` and `PropertyNode.get`/`set` reach it through
    that same path. Adoption transfers ownership: assign copies to hold the
    same node in two trees.
    """

    kind = "nodes"

    def __init__(self, node_class, doc=""):
        super().__init__((), doc)
        self.node_class = node_class

    def validate(self, obj, value):
        nodes, seen = [], set()
        for item in self.coerce(obj, value):
            # a node has one parent link, so it can only hold one slot
            node = item.copy() if id(item) in seen else item
            seen.add(id(node))
            nodes.append(node)
        return tuple(nodes)

    def coerce(self, obj, value):
        """Return the given input as an iterable of `node_class` instances."""
        items = value if isinstance(value, list | tuple) else [value]
        nodes = []
        for item in items:
            node = self.node_class(**item) if isinstance(item, dict) else item
            if not isinstance(node, self.node_class):
                msg = (
                    f"The {self.name} property of {type(obj).__name__} must be a "
                    f"sequence of {self.node_class.__name__} instances or of "
                    f"dictionaries with equivalent key/value pairs; instead "
                    f"received {item!r}."
                )
                raise TypeError(msg)
            nodes.append(node)
        return nodes

    def __set__(self, obj, value):
        # `Property.__set__` inlined: elements have to be adopted before
        # observers are notified, so a callback sees a fully linked tree.
        previous = obj.__dict__.get(self.name, ())
        if value is None:
            obj.__dict__.pop(self.name, None)  # unset
            new = self.default
        else:  # validated first: a rejected value changes nothing
            new = obj.__dict__[self.name] = self.validate(obj, value)
        for child in previous:
            if child._parent is obj:  # dropped nodes leave the tree
                object.__setattr__(child, "_parent", None)
                object.__setattr__(child, "_name", None)
        for index, child in enumerate(new):
            _adopt_child(obj, child, f"{self.name}.{index}")
        if obj._observed:
            obj._notify(self.name, new)

    def schema(self):
        out = super().schema()
        out["type"] = ["array", "null"]
        out["items"] = self.node_class.schema()
        return out


def _detached(value):
    """Return ``value`` with every property node inside it replaced by a copy.

    Assignment adopts nodes, transferring ownership (see `NodeSequence`), so
    values handed to a node that must not take them over are detached first:
    a defaults layer feeding `PropertyNode.merged`, or caller style kwargs
    feeding the throwaway style that `get_style` resolves against.
    """
    if isinstance(value, PropertyNode):
        return value.copy()
    if isinstance(value, dict):
        return {key: _detached(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        # the container type is part of the value: a tuple that came back as a
        # list would fail validation of e.g. the args of a Trace3d
        items = [_detached(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


class PropertyNode:
    """Base class for nodes of a typed-property tree.

    Subclasses declare fields as `Property` descriptors (multiple
    inheritance of field mixins is supported). Instances store only
    explicitly set values; everything else defers to field defaults and,
    at resolution time, to defaults layers via `merged()`.
    """

    _fields: ClassVar[dict] = {}
    _parent = None
    _name = None
    _observed = False
    _observers = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._fields = {
            key: val
            for klass in reversed(cls.__mro__)
            for key, val in vars(klass).items()
            if isinstance(val, Property)
        }
        for key in cls._fields:
            if "_" in key:
                msg = (
                    f"Property name {key!r} of {cls.__name__} must not contain "
                    "underscores; they are reserved for magic underscore notation."
                )
                raise ValueError(msg)

    def __init__(self, **kwargs):
        if kwargs:
            self.update(kwargs)

    def __setattr__(self, name, value):
        if (
            name in self._fields
            or name.startswith("_")
            # plain python properties are allowed, e.g. deprecated aliases
            or isinstance(getattr(type(self), name, None), property)
        ):
            object.__setattr__(self, name, value)
        else:
            msg = (
                f"{type(self).__name__} has no property {name!r}. "
                f"Available properties: {list(self._fields)}."
            )
            raise AttributeError(msg)

    def __repr__(self):
        args = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields)
        return f"{type(self).__name__}({args})"

    def __eq__(self, other):
        return type(other) is type(self) and self.as_dict() == other.as_dict()

    __hash__ = None  # mutable container

    # dict round-tripping ------------------------------------------------

    def as_dict(self, flatten=False, separator="."):
        """Return all effective values as a nested dict (`flatten=True` for
        a flat dict with `separator`-joined keys)."""
        dict_ = {
            name: getattr(self, name).as_dict()
            if isinstance(field, Nested)
            else getattr(self, name)
            for name, field in self._fields.items()
        }
        if flatten:
            dict_ = linearize_dict(dict_, separator=separator)
        return dict_

    def update(
        self, arg=None, _match_properties=True, _replace_None_only=False, **kwargs
    ):
        """Update properties from a dict and/or keyword arguments, supporting
        magic underscore notation. Nested dicts merge into child nodes.

        Parameters
        ----------
        _match_properties: bool
            If ``True`` (default), unknown property names raise an
            AttributeError; if ``False`` they are silently ignored.
        _replace_None_only: bool
            If ``True``, only properties that are not explicitly set get
            updated.
        """
        data = magic_to_dict({**(arg or {}), **kwargs})
        for key, value in data.items():
            field = self._fields.get(key)
            if field is None:
                if isinstance(getattr(type(self), key, None), property):
                    setattr(self, key, value)  # e.g. deprecated aliases
                elif _match_properties:
                    msg = (
                        f"{type(self).__name__} has no property {key!r}. "
                        f"Available properties: {list(self._fields)}."
                    )
                    raise AttributeError(msg)
                continue
            if isinstance(field, Nested) and isinstance(value, dict):
                getattr(self, key).update(
                    value,
                    _match_properties=_match_properties,
                    _replace_None_only=_replace_None_only,
                )
            elif not (_replace_None_only and self.is_set(key)):
                setattr(self, key, value)
        return self

    _IMMUTABLE_TYPES = (str, int, float, bool, type(None))

    def __deepcopy__(self, memo):
        """Deep copies are detached, like `copy()`: a node is copied with its
        own subtree, never with the tree it hangs in — whose observers would
        otherwise fire for edits made to the copy."""
        new = self.copy()
        memo[id(self)] = new
        return new

    def copy(self):
        """Return a detached, unobserved deep copy."""
        new = type(self).__new__(type(self))
        for key, value in self.__dict__.items():
            field = self._fields.get(key)
            if field is None:
                continue
            if isinstance(value, PropertyNode):
                new.__dict__[key] = _adopt_child(new, value.copy(), key)
            elif isinstance(value, self._IMMUTABLE_TYPES):
                new.__dict__[key] = value
            elif isinstance(field, NodeSequence):
                new.__dict__[key] = tuple(
                    _adopt_child(new, child.copy(), f"{key}.{index}")
                    for index, child in enumerate(value)
                )
            else:
                new.__dict__[key] = deepcopy(value)
        return new

    # public contract for GUIs / third-party tooling ---------------------

    @classmethod
    def schema(cls):
        """Return the JSON-Schema description of the field tree."""
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: field.schema() for name, field in cls._fields.items()},
        }

    def get(self, path):
        """Return the value at a dotted path, e.g. ``'arrow.width'``; a node
        inside a collection is addressed by index, e.g. ``'data.0.show'``."""
        obj = self
        for part in path.split("."):
            if isinstance(obj, list | tuple):
                if not part.lstrip("-").isdigit():
                    msg = (
                        f"Path {path!r} is invalid: {part!r} must be an index "
                        "into the collection it addresses."
                    )
                    raise ValueError(msg)
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def set(self, path, value):
        """Set the value at a dotted path and return self."""
        parent_path, _, leaf = path.rpartition(".")
        node = self.get(parent_path) if parent_path else self
        if not isinstance(node, PropertyNode):
            msg = (
                f"Cannot set {path!r}: {parent_path!r} is a "
                f"{type(node).__name__}, not a property node; assign the whole "
                f"value at {parent_path!r} instead."
            )
            raise TypeError(msg)
        setattr(node, leaf, value)
        return self

    def is_set(self, name):
        """Return whether a direct property holds an explicitly set value
        (for child nodes: whether anything in the subtree is set)."""
        value = self.__dict__.get(name)
        if isinstance(value, PropertyNode):
            return bool(value.set_values())
        return name in self.__dict__

    def set_values(self, separator="."):
        """Return a flat ``{dotted_path: value}`` dict of all explicitly set
        leaf values."""
        out = {}
        for name in self._fields:
            value = self.__dict__.get(name)
            if value is None:
                continue
            if isinstance(value, PropertyNode):
                for path, leaf in value.set_values(separator).items():
                    out[f"{name}{separator}{path}"] = leaf
            else:
                out[name] = value
        return out

    def merged(self, *layers):
        """Return a copy where unset properties are filled from the given
        layers, first-set-wins (self has the highest priority)."""
        new = self.copy()
        for layer in layers:
            values = {
                key: _detached(value)
                for key, value in layer.set_values(separator="_").items()
            }
            new.update(
                magic_to_dict(values),
                _match_properties=False,
                _replace_None_only=True,
            )
        return new

    # change observation --------------------------------------------------

    def observe(self, callback):
        """Register ``callback(path, value)`` to fire on any change in this
        subtree; ``path`` is the dotted path of the changed property relative
        to this node."""
        object.__setattr__(self, "_observers", (*self._observers, callback))
        self._mark_observed()

    def unobserve(self, callback):
        """Unregister a previously registered callback."""
        observers = list(self._observers)
        observers.remove(callback)
        object.__setattr__(self, "_observers", tuple(observers))

    def _mark_observed(self):
        object.__setattr__(self, "_observed", True)
        for key, value in self.__dict__.items():
            field = self._fields.get(key)
            if field is None:  # e.g. the _parent backlink
                continue
            if isinstance(value, PropertyNode):
                value._mark_observed()
            elif isinstance(field, NodeSequence):
                for child in value:
                    child._mark_observed()

    def _notify(self, path, value):
        for callback in self._observers:
            callback(path, value)
        if self._parent is not None:
            self._parent._notify(f"{self._name}.{path}", value)
