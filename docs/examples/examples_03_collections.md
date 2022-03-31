---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(examples-collections-construction)=

# Collections

## Constructing collections

The `Collection` class is a versatile way of grouping and manipulating Magpylib objects. When objects are added to a Collection they are added by reference (not copied) to the **attributes** `children` (list of all objects), `sources` (list of the sources), `sensors` (list of the sensors) and `collections`.

```{code-cell} ipython3
import magpylib as magpy

x1 = magpy.Sensor(style_label='x1')
s1 = magpy.magnet.Cuboid(style_label='s1')
c1 = magpy.Collection(style_label='c1')

coll = magpy.Collection(x1, s1, c1, style_label='coll')

print(f"children:    {coll.children}")
print(f"sources:     {coll.sources}")
print(f"sensors:     {coll.sensors}")
print(f"collections: {coll.collections}")
```

These attributes are ordered lists. New additions are always added at the end. Add objects to an existing collection using these parameters, or the **`add`** method.

```{code-cell} ipython3

# automatically adjusts object label
x2 = x1.copy()
s2 = s1.copy()
c2 = c1.copy()

# add objects with add method
coll.add(x2, s2)

# add objects with parameters
coll.collections += [c2]

print(f"children:    {coll.children}")
print(f"sources:     {coll.sources}")
print(f"sensors:     {coll.sensors}")
print(f"collections: {coll.collections}")
```

The **`describe`** method is a very convenient way to view a Collection structure, especially when the collection is nested, i.e. when containing other collections:

```{code-cell} ipython3
# add more objects
c1.add(x2.copy())
c2.add(s2.copy())

coll.describe(format='label')
```

For convenience, any two Magpylib object can be added up with `+` to form a collection:

```{code-cell} ipython3
import magpylib as magpy

x1 = magpy.Sensor(style_label='x1')
s1 = magpy.magnet.Cuboid(style_label='s1')

coll = x1 + s1

coll.describe(format='label')
```

## Child-parent relations

Objects that are part of a collection become children of that collection, and the collection itself becomes their parent. Every Magpylib object has the `parent` attribute, which is `None` by default. 

```{code-cell} ipython3
import magpylib as magpy

x1 = magpy.Sensor()
c1 = magpy.Collection(x1)

print(f"x1.parent:   {x1.parent}")
print(f"c1.parent:   {c1.parent}")
print(f"c1.children: {c1.children}")
```

Rather than adding objects to a collection, as described above, one can also set the `parent` parameter. A Magpylib object can only have a single parent, i.e. it can only be part of a single collection. As a result, changing the parent will automatically remove the object from it's previous collection.

```{code-cell} ipython3
import magpylib as magpy

x1 = magpy.Sensor(style_label='x1')
c1 = magpy.Collection(style_label='c1')
c2 = magpy.Collection(c1, style_label='c2')

print("Two empty, nested collections")
c2.describe(format='label')

print("\nSet x1 parent to c1")
x1.parent = c1
c2.describe(format='label')

print("\nChange x1 parent to c2")
x1.parent = c2
c2.describe(format='label')
```


## Working with collections

Collections have `__getitem__` through the attribute `children` defined which enables using collections directly as iterators,

```{code-cell} ipython3
import magpylib as magpy

x1 = magpy.Sensor()
x2 = magpy.Sensor()

coll = x1 + x2

for child in coll:
    print(child)
```

and makes it possible to directly reference to a child object by index:

```{code-cell} ipython3
print(coll[0])
```

How to work with collections in a practical way is demonstrated in the introduction section {ref}`intro-collections`.

How to make complex compound objects is documented in {ref}`examples-compounds`.

(examples-collections-efficient)=

## Efficient 3D models

The Matplotlib and Plotly libraries were not designed for complex 3D graphic outputs. As a result, it becomes often inconvenient and slow when attempting to display many 3D objects. One solution to this problem when dealing with large collections, is to represent the latter by a single encompassing body, and to deactivate the individual 3D models of all children. This is demonstrated in the following example.

```{code-cell} ipython3
import magpylib as magpy

# create collection
coll = magpy.Collection()
for index in range(10):
    cuboid = magpy.magnet.Cuboid(
        magnetization=(0, 0, 1000 * (index%2-.5)),
        dimension=(10,10,10),
        position=(index*10,0,0),
    )
    coll.add(cuboid)

# add 3D-trace
plotly_trace = magpy.graphics.model3d.make_Cuboid(
    backend='matplotlib',
    dimension=(104, 12, 12),
    position=(45, 0, 0),
    alpha=0.5,
)
coll.style.model3d.add_trace(plotly_trace)

coll.style.label='Collection with visible children'
coll.show()

# hide the children default 3D representation
coll.set_children_styles(model3d_showdefault=False)
coll.style.label = 'Collection with hidden children'
coll.show()
```

```{note}
The `Collection` position is set to (0,0,0) at creation time. Any added extra 3D-model will be bound to the local coordinate system of to the `Collection` and `rotated`/`moved` together with its parent object.
```
