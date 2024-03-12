import numpy as np

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput

# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=no-member


def test_parent_setter():
    """setting and removing a parent"""
    child_labels = lambda x: [c.style.label for c in x]

    # default parent is None
    x1 = magpy.Sensor(style_label="x1")
    assert x1.parent is None

    # init collection gives parent
    c1 = magpy.Collection(x1, style_label="c1")
    assert x1.parent.style.label == "c1"
    assert child_labels(c1) == ["x1"]

    # remove parent with setter
    x1.parent = None
    assert x1.parent is None
    assert child_labels(c1) == []

    # set parent
    x1.parent = c1
    assert x1.parent.style.label == "c1"
    assert child_labels(c1) == ["x1"]

    # set another parent
    c2 = magpy.Collection(style_label="c2")
    x1.parent = c2
    assert x1.parent.style.label == "c2"
    assert child_labels(c1) == []
    assert child_labels(c2) == ["x1"]


def test_children_setter():
    """setting new children and removing old parents"""
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    x3 = magpy.Sensor()
    x4 = magpy.Sensor()

    c = magpy.Collection(x1, x2)
    c.children = [x3, x4]

    # remove old parents
    assert x1.parent is None
    assert x2.parent is None
    # new children
    assert c[0] == x3
    assert c[1] == x4


def test_setter_parent_override():
    """all setter should override parents"""
    x1 = magpy.Sensor()
    s1 = magpy.magnet.Cuboid()
    c1 = magpy.Collection()
    coll = magpy.Collection(x1, s1, c1)

    coll2 = magpy.Collection()
    coll2.children = coll.children
    assert coll2.children == [x1, s1, c1]

    coll3 = magpy.Collection()
    coll3.sensors = [x1]
    coll3.sources = [s1]
    coll3.collections = [c1]
    assert coll3.children == [x1, s1, c1]


def test_sensors_setter():
    """setting new sensors and removing old parents"""
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    x3 = magpy.Sensor()
    x4 = magpy.Sensor()
    s1 = magpy.magnet.CylinderSegment()

    c = magpy.Collection(x1, x2, s1)
    c.sensors = [x3, x4]

    # remove old parents
    assert x1.parent is None
    assert x2.parent is None
    # keep non-sensors
    assert s1.parent == c
    # new sensors
    assert c[0] == s1
    assert c.sensors[0] == x3
    assert c.sensors[1] == x4


def test_sources_setter():
    """setting new sources and removing old parents"""
    s1 = magpy.magnet.Cylinder()
    s2 = magpy.magnet.Cylinder()
    s3 = magpy.magnet.Cylinder()
    s4 = magpy.magnet.Cylinder()
    x1 = magpy.Sensor()

    c = magpy.Collection(x1, s1, s2)
    c.sources = [s3, s4]

    # old parents
    assert s1.parent is None
    assert s2.parent is None
    # keep non-sources
    assert x1.parent == c
    # new children
    assert c[0] == x1
    assert c.sources[0] == s3
    assert c[2] == s4


def test_collections_setter():
    """setting new sources and removing old parents"""
    c1 = magpy.Collection()
    c2 = magpy.Collection()
    c3 = magpy.Collection()
    c4 = magpy.Collection()
    x1 = magpy.Sensor()

    c = magpy.Collection(c1, x1, c2)
    c.collections = [c3, c4]

    # old parents
    assert c1.parent is None
    assert c2.parent is None
    # keep non-collections
    assert x1.parent == c
    # new children
    assert c[0] == x1
    assert c.collections[0] == c3
    assert c[2] == c4


def test_collection_inputs():
    """test basic collection inputs"""

    s1 = magpy.magnet.Cuboid(style_label="s1")
    s2 = magpy.magnet.Cuboid(style_label="s2")
    s3 = magpy.magnet.Cuboid(style_label="s3")
    x1 = magpy.Sensor(style_label="x1")
    x2 = magpy.Sensor(style_label="x2")
    c1 = magpy.Collection(x2, style_label="c1")

    c2 = magpy.Collection(c1, x1, s1, s2, s3)
    assert [c.style.label for c in c2.children] == ["c1", "x1", "s1", "s2", "s3"]
    assert [c.style.label for c in c2.sensors] == ["x1"]
    assert [c.style.label for c in c2.sources] == ["s1", "s2", "s3"]
    assert [c.style.label for c in c2.collections] == ["c1"]


def test_collection_parent_child_relation():
    """test if parent-child relations are properly set with collections"""

    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cuboid()
    s3 = magpy.magnet.Cuboid()
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    c1 = magpy.Collection(x2)
    c2 = magpy.Collection(c1, x1, s1, s2, s3)

    assert x1.parent == c2
    assert s3.parent == c2
    assert x2.parent == c1
    assert c1.parent == c2
    assert c2.parent is None


def test_collections_add():
    """test collection construction"""
    child_labels = lambda x: [c.style.label for c in x]

    x1 = magpy.Sensor(style_label="x1")
    x2 = magpy.Sensor(style_label="x2")
    x3 = magpy.Sensor(style_label="x3")
    x6 = magpy.Sensor(style_label="x6")
    x7 = magpy.Sensor(style_label="x7")

    # simple add
    c2 = magpy.Collection(x1, style_label="c2")
    c2.add(x2, x3)
    assert child_labels(c2) == ["x1", "x2", "x3"]

    # adding another collection
    c3 = magpy.Collection(x6, style_label="c3")
    c2.add(c3)
    assert child_labels(c2) == ["x1", "x2", "x3", "c3"]
    assert child_labels(c3) == ["x6"]

    # adding to child collection should not change its parent collection
    c3.add(x7)
    assert child_labels(c2) == ["x1", "x2", "x3", "c3"]
    assert child_labels(c3) == ["x6", "x7"]

    # add with parent override
    assert x7.parent == c3

    c4 = magpy.Collection(style_label="c4")
    c4.add(x7, override_parent=True)

    assert child_labels(c3) == ["x6"]
    assert child_labels(c4) == ["x7"]
    assert x7.parent == c4

    # set itself as parent should fail
    with np.testing.assert_raises(MagpylibBadUserInput):
        c2.parent = c2

    # add itself, also nested, should fail
    with np.testing.assert_raises(MagpylibBadUserInput):
        c2.add(magpy.Collection(c2))


def test_collection_plus():
    """
    testing collection adding and the += functionality
    """
    child_labels = lambda x: [c.style.label for c in x]

    s1 = magpy.magnet.Cuboid(style_label="s1")
    s2 = magpy.magnet.Cuboid(style_label="s2")
    x1 = magpy.Sensor(style_label="x1")
    x2 = magpy.Sensor(style_label="x2")
    x3 = magpy.Sensor(style_label="x3")
    c1 = magpy.Collection(s1, style_label="c1")

    # practical simple +
    c2 = c1 + s2
    assert child_labels(c2) == ["c1", "s2"]

    # useless triple addition consistency
    c3 = x1 + x2 + x3
    assert c3[0][0].style.label == "x1"
    assert c3[0][1].style.label == "x2"
    assert c3[1].style.label == "x3"

    # useless += consistency
    s3 = magpy.magnet.Cuboid(style_label="s3")
    c2 += s3
    assert [c.style.label for c in c2[0]] == ["c1", "s2"]
    assert c2[1] == s3


def test_collection_remove():
    """removing from collections"""
    child_labels = lambda x: [c.style.label for c in x]
    source_labels = lambda x: [c.style.label for c in x.sources]
    sensor_labels = lambda x: [c.style.label for c in x.sensors]

    x1 = magpy.Sensor(style_label="x1")
    x2 = magpy.Sensor(style_label="x2")
    x3 = magpy.Sensor(style_label="x3")
    x4 = magpy.Sensor(style_label="x4")
    x5 = magpy.Sensor(style_label="x5")
    s1 = magpy.misc.Dipole(style_label="s1")
    s2 = magpy.misc.Dipole(style_label="s2")
    s3 = magpy.misc.Dipole(style_label="s3")
    q1 = magpy.misc.CustomSource(style_label="q1")
    c1 = magpy.Collection(x1, x2, x3, x4, x5, style_label="c1")
    c2 = magpy.Collection(s1, s2, s3, style_label="c2")
    c3 = magpy.Collection(q1, c1, c2, style_label="c3")

    assert child_labels(c1) == ["x1", "x2", "x3", "x4", "x5"]
    assert child_labels(c2) == ["s1", "s2", "s3"]
    assert child_labels(c3) == ["q1", "c1", "c2"]

    # remove item from collection
    c1.remove(x5)
    assert child_labels(c1) == ["x1", "x2", "x3", "x4"]
    assert [c.style.label for c in c1.sensors] == ["x1", "x2", "x3", "x4"]

    # remove 2 items from collection
    c1.remove(x3, x4)
    assert child_labels(c1) == ["x1", "x2"]
    assert sensor_labels(c1) == ["x1", "x2"]

    # remove item from child collection
    c3.remove(s3)
    assert child_labels(c3) == ["q1", "c1", "c2"]
    assert child_labels(c2) == ["s1", "s2"]
    assert source_labels(c2) == ["s1", "s2"]

    # remove child collection
    c3.remove(c2)
    assert child_labels(c3) == ["q1", "c1"]
    assert child_labels(c2) == ["s1", "s2"]

    # attempt remove non-existent child
    c3.remove(s1, errors="ignore")
    assert child_labels(c3) == ["q1", "c1"]
    assert child_labels(c1) == ["x1", "x2"]

    # attempt remove child in lower level with recursion=False
    c3.remove(x1, errors="ignore", recursive=False)
    assert child_labels(c3) == ["q1", "c1"]
    assert child_labels(c1) == ["x1", "x2"]

    # attempt remove of non-existing child
    with np.testing.assert_raises(MagpylibBadUserInput):
        c3.remove(x1, errors="raise", recursive=False)


def test_collection_nested_getBH():
    """test if getBH functionality is self-consistent with nesting"""
    s1 = magpy.current.Circle(current=1, diameter=1)
    s2 = magpy.current.Circle(current=1, diameter=1)
    s3 = magpy.current.Circle(current=1, diameter=1)
    s4 = magpy.current.Circle(current=1, diameter=1)

    obs = [(1, 2, 3), (-2, -3, 1), (2, 2, -4), (4, 2, -4)]
    coll = s1 + s2 + s3 + s4  # nasty nesting

    B1 = s1.getB(obs)
    B4 = coll.getB(obs)
    np.testing.assert_allclose(4 * B1, B4)

    H1 = s1.getH(obs)
    H4 = coll.getH(obs)
    np.testing.assert_allclose(4 * H1, H4)


def test_collection_properties_all():
    """test _all properties"""
    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cylinder()
    s3 = magpy.current.Circle()
    s4 = magpy.current.Circle()
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    x3 = magpy.Sensor()
    c1 = magpy.Collection(s2)
    c3 = magpy.Collection(s4)
    c2 = magpy.Collection(s3, x3, c3)

    cc = magpy.Collection(s1, x1, c1, x2, c2)

    assert cc.children == [s1, x1, c1, x2, c2]
    assert cc.sources == [s1]
    assert cc.sensors == [x1, x2]
    assert cc.collections == [c1, c2]

    assert cc.children_all == [s1, x1, c1, s2, x2, c2, s3, x3, c3, s4]
    assert cc.sources_all == [s1, s2, s3, s4]
    assert cc.sensors_all == [x1, x2, x3]
    assert cc.collections_all == [c1, c2, c3]
