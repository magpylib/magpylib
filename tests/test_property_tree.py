"""Tests for the declarative typed-property tree (successor of MagicProperties)."""

import json
from copy import deepcopy

import pytest

from magpylib._src.defaults.property_tree import (
    Boolean,
    Choice,
    Color,
    Integer,
    Nested,
    NodeSequence,
    Number,
    Property,
    PropertyNode,
    String,
)


class Line(PropertyNode):
    style = Choice("solid", "dashed", "dotted", doc="Line style.")
    color = Color(doc="Line color.")
    width = Number(minimum=0, doc="Line width.")


class Marker(PropertyNode):
    size = Number(minimum=0)
    symbol = String()


class Path(PropertyNode):
    line = Nested(Line)
    marker = Nested(Marker)
    show = Boolean(default=True)
    frames = Integer(minimum=1)


class LabelMixin(PropertyNode):
    label = String(coerce=True, doc="Object label.")


class Style(LabelMixin):
    path = Nested(Path)
    opacity = Number(minimum=0, maximum=1)


class Layer(PropertyNode):
    name = String()
    weight = Number(minimum=0)


class Stack(LabelMixin):
    layers = NodeSequence(Layer, doc="Stacked layers.")
    meta = Property(doc="Free-form metadata.")


# field registration ---------------------------------------------------


def test_fields_registration_and_inheritance():
    """Fields are collected per class, through multiple inheritance,
    subclass declarations overriding base ones."""
    assert list(Line._fields) == ["style", "color", "width"]
    assert list(Style._fields) == ["label", "path", "opacity"]

    class Special(Style):
        opacity = Number()  # override without bounds

    assert Special._fields["opacity"].maximum is None
    assert Style._fields["opacity"].maximum == 1


def test_underscore_field_names_are_rejected():
    """Underscores in field names would break magic underscore notation."""
    with pytest.raises(ValueError, match="underscore"):

        class Bad(PropertyNode):
            show_default = Boolean()


# validation ------------------------------------------------------------


def test_validation():
    """Each descriptor type validates on assignment with a ValueError."""
    line = Line()
    with pytest.raises(ValueError, match="one of"):
        line.style = "wiggly"
    with pytest.raises(ValueError, match="number >=0"):
        line.width = -1
    with pytest.raises(ValueError, match="color"):
        line.color = "notacolor"
    line.color = [0.1, 0.2, 0.3]  # normalized like all validated colors
    assert line.color == "#19334c"

    path = Path()
    with pytest.raises(ValueError, match="True or False"):
        path.show = 1
    with pytest.raises(ValueError, match="integer"):
        path.frames = 2.5

    style = Style()
    with pytest.raises(ValueError, match="<=1"):
        style.opacity = 2
    style.label = 42  # coercing string
    assert style.label == "42"


def test_choice_from_callable_resolves_at_call_time():
    """A Choice built from a callable re-reads its choices on every use."""
    allowed = ["red"]

    class Dynamic(PropertyNode):
        color = Choice(lambda: tuple(allowed), doc="Dynamic color.")

    dyn = Dynamic()
    assert Dynamic.__dict__["color"].choices == ("red",)
    assert Dynamic.schema()["properties"]["color"]["enum"] == ["red", None]
    with pytest.raises(ValueError, match="one of"):
        dyn.color = "green"

    allowed.append("green")
    dyn.color = "green"  # now allowed, without redefining the class
    assert Dynamic.schema()["properties"]["color"]["enum"] == ["red", "green", None]


def test_unknown_property_raises():
    """The attribute set is frozen: typos raise with available properties."""
    with pytest.raises(AttributeError, match="Available properties"):
        Line().colour = "red"
    with pytest.raises(AttributeError, match="Available properties"):
        Line(colour="red")


# set/unset semantics ----------------------------------------------------


def test_unset_returns_default_and_none_unsets():
    """Unset fields report their default; assigning None resets to it."""
    path = Path()
    assert path.show is True  # non-None default
    assert path.frames is None
    assert not path.is_set("show")

    path.show = False
    assert path.is_set("show")
    path.show = None
    assert path.show is True
    assert not path.is_set("show")


def test_set_values_flat_dict():
    """set_values returns only explicitly set leaves, as a flat dict."""
    style = Style(opacity=0.5)
    style.path.line.width = 2
    assert style.set_values() == {"opacity": 0.5, "path.line.width": 2}
    assert style.is_set("path")
    assert not Style().is_set("path")


# dict round-tripping ----------------------------------------------------


def test_magic_underscore_init_update_and_as_dict():
    """Constructor kwargs, update() and dicts support magic underscores."""
    s1 = Style(path_line_width=2, opacity=0.5)
    s2 = Style().update({"path": {"line": {"width": 2}}, "opacity": 0.5})
    assert s1 == s2
    assert s1.as_dict()["path"]["line"]["width"] == 2
    assert s1.as_dict(flatten=True)["path.line.width"] == 2

    s1.update(path_line_style="dashed")
    assert s1.path.line.style == "dashed"
    assert s1.path.line.width == 2  # update merges, does not reset siblings

    with pytest.raises(AttributeError, match="no property"):
        s1.update(bananas=5)
    s1.update(bananas=5, _match_properties=False)  # silently ignored


def test_update_replace_none_only():
    """_replace_None_only fills only unset properties."""
    style = Style(opacity=0.5)
    style.update(opacity=0.9, path_show=False, _replace_None_only=True)
    assert style.opacity == 0.5  # already set, kept
    assert style.path.show is False  # was unset, filled


def test_nested_assignment_replaces():
    """Assigning a dict or None to a child node replaces it wholesale."""
    style = Style(path_line_width=2)
    style.path = {"line": {"style": "dotted"}}
    assert style.path.line.style == "dotted"
    assert style.path.line.width is None  # replaced, not merged
    style.path = None
    assert style.set_values() == {}
    with pytest.raises(TypeError, match="instance of Path"):
        style.path = "bad"


def test_copy_is_detached():
    """Copies are independent of the original and unobserved."""
    style = Style(path_line_width=2)
    clone = style.copy()
    clone.path.line.width = 5
    assert style.path.line.width == 2
    assert clone == Style(path_line_width=5)
    assert clone != style


# lazy children -----------------------------------------------------------


def test_children_are_lazy():
    """Child nodes are only instantiated on first access."""
    style = Style()
    assert "path" not in style.__dict__
    _ = style.path
    assert "path" in style.__dict__


# dotted-path access -------------------------------------------------------


def test_get_set_by_path():
    """get/set navigate dotted paths; unknown segments raise."""
    style = Style()
    style.set("path.line.width", 3).set("opacity", 0.1)
    assert style.get("path.line.width") == 3
    assert style.get("path.line") is style.path.line
    with pytest.raises(AttributeError):
        style.get("path.nope")
    with pytest.raises(ValueError, match=">=0"):
        style.set("path.line.width", -1)


# layered resolution --------------------------------------------------------


def test_merged_first_set_wins():
    """merged() fills unset values from layers, earlier layers first."""
    obj = Style(opacity=0.5)
    family = Style(opacity=0.9, path_line_width=2, label="family")
    base = Style(path_line_width=7, path_line_style="dashed")

    resolved = obj.merged(family, base)
    assert resolved.opacity == 0.5  # self wins
    assert resolved.path.line.width == 2  # first layer wins
    assert resolved.path.line.style == "dashed"  # second layer fills the rest
    assert resolved.label == "family"
    # inputs are untouched
    assert obj.set_values() == {"opacity": 0.5}
    assert base.path.line.width == 7


# observation ---------------------------------------------------------------


def test_observe_bubbles_with_dotted_path():
    """Observers receive dotted paths relative to the observed node, for
    both direct and nested changes, including on lazily created children."""
    style = Style()
    events = []
    style.observe(lambda path, value: events.append((path, value)))

    style.opacity = 0.5
    style.path.line.width = 2  # path/line created lazily after observe()
    style.set("path.line.width", None)
    assert events == [
        ("opacity", 0.5),
        ("path.line.width", 2),
        ("path.line.width", None),
    ]

    # observer on a subnode sees subtree changes only, with relative paths
    sub_events = []
    callback = lambda path, _: sub_events.append(path)
    style.path.observe(callback)
    style.path.line.style = "dotted"
    style.label = "x"
    assert sub_events == ["line.style"]

    style.path.unobserve(callback)
    style.path.line.style = "dashed"
    assert sub_events == ["line.style"]


def test_unobserved_sets_have_no_bookkeeping():
    """Without observers, assignments leave no observation state behind."""
    style = Style(path_line_width=2)
    assert style._observed is False
    assert style.path._observed is False


# node collections ------------------------------------------------------------


def test_node_sequence_is_always_a_tuple():
    """A collection field holds a tuple, whether unset, emptied or filled -
    so it can only change by assignment, never by mutating what it returns."""
    stack = Stack()
    assert stack.layers == ()
    assert stack.is_set("layers") is False

    stack.layers = [{"name": "a"}, Layer(name="b")]
    assert isinstance(stack.layers, tuple)
    assert [layer.name for layer in stack.layers] == ["a", "b"]

    stack.layers = {"name": "solo"}  # a single item is wrapped
    assert len(stack.layers) == 1

    stack.layers = []  # explicitly empty, and explicitly set
    assert stack.layers == ()
    assert stack.is_set("layers") is True

    stack.layers = None  # unset, back to the default
    assert stack.layers == ()
    assert stack.is_set("layers") is False

    with pytest.raises(AttributeError):  # in-place mutation is impossible
        stack.layers.append(Layer())
    with pytest.raises(TypeError, match="sequence of Layer"):
        stack.layers = ["not a layer"]


def test_node_sequence_rejected_assignment_changes_nothing():
    """A collection that fails validation keeps its elements, and keeps them
    reporting into the tree."""
    stack = Stack(layers=[{"name": "a"}])
    events = []
    stack.observe(lambda path, _: events.append(path))

    with pytest.raises(TypeError, match="sequence of Layer"):
        stack.layers = [Layer(name="b"), "not a layer"]

    assert [layer.name for layer in stack.layers] == ["a"]
    stack.layers[0].weight = 1
    assert events == ["layers.0.weight"]


def test_node_sequence_elements_join_the_tree():
    """Elements are adopted, so changes inside them bubble up under their
    index, and get/set reach them through that same path."""
    stack = Stack()
    events = []
    stack.observe(lambda path, value: events.append((path, value)))

    stack.layers = [{"name": "a"}, {"name": "b"}]
    stack.layers[1].weight = 2
    stack.set("layers.0.weight", 1)
    assert events[1:] == [("layers.1.weight", 2), ("layers.0.weight", 1)]
    assert stack.get("layers.0.weight") == 1
    assert stack.get("layers.-1") is stack.layers[1]

    with pytest.raises(ValueError, match="must be an index"):
        stack.get("layers.nope")
    # a path can only end on a property; the error says what it ran into
    with pytest.raises(TypeError, match="'layers' is a tuple"):
        stack.set("layers.0", Layer())
    stack.meta = {"origin": "file"}
    with pytest.raises(TypeError, match="'meta' is a dict"):
        stack.set("meta.origin", "hand")

    # a dropped element leaves the tree instead of reporting into it
    dropped = stack.layers[0]
    stack.layers = []
    dropped.weight = 9
    assert dropped._parent is None
    assert [path for path, _ in events][-1] == "layers"


def test_node_sequence_elements_are_adopted_before_notification():
    """The collection event fires on a fully linked tree, so a callback that
    edits a fresh element is itself reported."""
    stack = Stack()
    events = []

    def callback(path, value):
        events.append(path)
        if path == "layers":
            value[0].weight = 1  # edit an element the callback just received

    stack.observe(callback)
    stack.layers = [{"name": "a"}]
    assert events == ["layers", "layers.0.weight"]


def test_node_sequence_observed_after_the_fact():
    """observe() reaches elements that were already in the collection."""
    stack = Stack(layers=[{"name": "a"}])
    events = []
    stack.observe(lambda path, value: events.append((path, value)))
    stack.layers[0].weight = 3
    assert events == [("layers.0.weight", 3)]


def test_node_sequence_copy_is_detached():
    """Copies own their elements: editing one leaves the original alone."""
    stack = Stack(layers=[{"name": "a", "weight": 1}])
    clone = stack.copy()
    clone.layers[0].weight = 5
    assert stack.layers[0].weight == 1
    assert clone.layers[0]._parent is clone
    assert clone.layers[0]._name == "layers.0"


def test_node_sequence_element_deepcopy_is_detached():
    """copy.deepcopy of an element copies the element, not the tree it hangs
    in - otherwise the live tree's observers would fire for edits to the copy."""
    stack = Stack(layers=[{"name": "a", "weight": 1}])
    events = []
    stack.observe(lambda path, _: events.append(path))

    clone = deepcopy(stack.layers[0])
    clone.weight = 5

    assert clone._parent is None
    assert events == []
    assert stack.layers[0].weight == 1

    # a whole tree still deep-copies with its internal links intact
    twin = deepcopy(stack)
    twin.layers[0].weight = 7
    assert twin.layers[0]._parent is twin
    assert stack.layers[0].weight == 1
    assert events == []


def test_node_sequence_element_cannot_hold_two_slots():
    """A node has one parent link, so the same node twice becomes two
    independent elements rather than one reporting under both indices."""
    layer = Layer(name="a")
    stack = Stack(layers=[layer, layer])
    events = []
    stack.observe(lambda path, _: events.append(path))

    assert stack.layers[0] is not stack.layers[1]
    stack.layers[0].weight = 1
    stack.layers[1].weight = 2
    assert events == ["layers.0.weight", "layers.1.weight"]


def test_node_sequence_merged_keeps_layer_ownership():
    """A layer keeps its own elements: merged() takes copies, because
    assignment would otherwise adopt them out of the layer's tree."""
    stack = Stack()
    layer = Stack(layers=[{"name": "a", "weight": 1}])
    resolved = stack.merged(layer)

    assert [lay.name for lay in resolved.layers] == ["a"]
    assert resolved.layers[0] is not layer.layers[0]
    assert layer.layers[0]._parent is layer  # ownership not stolen

    resolved.layers[0].weight = 5
    assert layer.layers[0].weight == 1


# schema ---------------------------------------------------------------------


def test_schema():
    """schema() describes the tree as JSON-serializable JSON Schema."""
    schema = Style.schema()
    props = schema["properties"]
    assert schema["additionalProperties"] is False
    assert props["opacity"] == {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    assert props["label"]["description"] == "Object label."
    line = props["path"]["properties"]["line"]["properties"]
    assert line["style"]["enum"] == ["solid", "dashed", "dotted", None]
    assert line["color"]["format"] == "color"
    assert props["path"]["properties"]["show"]["default"] is True
    json.dumps(schema)  # JSON-serializable

    layers = Stack.schema()["properties"]["layers"]
    assert layers["type"] == ["array", "null"]
    assert layers["items"] == Layer.schema()
    assert layers["description"] == "Stacked layers."
    json.dumps(Stack.schema())


# base Property ---------------------------------------------------------------


def test_schema_omits_callable_defaults():
    """A schema is JSON, so a callable default is described by its field but
    never reported as a `default` value."""

    class WithCallable(PropertyNode):
        hook = Property(default=dict, doc="Called at render time.")

    field = WithCallable.schema()["properties"]["hook"]
    assert field == {"description": "Called at render time."}
    json.dumps(WithCallable.schema())


def test_plain_property_accepts_anything():
    """The untyped base Property performs no validation."""

    class Free(PropertyNode):
        anything = Property(doc="No validation.")

    free = Free(anything=object)
    assert free.anything is object
    assert Free.schema()["properties"]["anything"] == {"description": "No validation."}
