(gallery-tutorial-collection)=

# Working with Collections

##

The top level class `Collection` allows users to group objects by reference for common manipulation. Objects that are part of a collection become **children** of that collection, and the collection itself becomes their **parent**. An object can only have a single parent. The child-parent relation is demonstrated with the `describe` method in the following example:

```{code-cell} ipython3
import magpylib as magpy

sens = magpy.Sensor(style_label='sens')
loop = magpy.current.Loop(style_label='loop')
line = magpy.current.Line(style_label='line')
cube = magpy.magnet.Cuboid(style_label='cube')

coll1 = magpy.Collection(sens, loop, line, style_label='Nested Collection')
coll2 = cube + coll1
coll2.style.label="Root Collection"
coll2.describe(format='label')
```


A detailed review of collection properties and construction is provided in the example gallery {ref}`examples-collections-construction`. It is specifically noteworthy in the above example, that any two Magpylib objects can simply be added up to form a collection.



This enables user-friendly manipulation of groups, sub-groups and individual objects, which is demonstrated in the following example:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
from magpylib.current import Loop

# construct two coil collections from windings
coil1 = magpy.Collection(style_label='coil1')
for z in np.linspace(-.5, .5, 5):
    coil1.add(Loop(current=1, diameter=20, position=(0,0,z)))
coil1.position = (0,0,-5)
coil2 = coil1.copy(position=(0,0,5))

# helmholtz consists of two coils
helmholtz = coil1 + coil2

# move the helmholz
helmholtz.position = np.linspace((0,0,0), (10,0,0), 30)
helmholtz.rotate_from_angax(np.linspace(0,180,30), 'x', start=0)

# move the coils
coil1.move(np.linspace((0,0,0), ( 5,0,0), 30))
coil2.move(np.linspace((0,0,0), (-5,0,0), 30))

# move the windings
for coil in [coil1, coil2]:
    for i,wind in enumerate(coil):
        wind.move(np.linspace((0,0,0), (0,0,2-i), 20))

magpy.show(*helmholtz, backend='plotly', animation=4, style_path_show=False)
```


Notice, that collections have their own `style` attributes, their paths are displayed in `show`, and all children are automatically assigned their parent color.

For magnetic field computation a collection with source children behaves like a single source object, and a collection with sensor children behaves like a flat list of it's sensors when provided as `sources` and `observers` input respectively. This is demonstrated in the following continuation of the previous Helmholtz example:

```{code-cell} ipython3
import matplotlib.pyplot as plt

B = helmholtz.getB((10,0,0))
plt.plot(B, label=['Bx', 'By', 'Bz'])

plt.gca().set(
    title='B-field (mT) at position (10,0,0)',
    xlabel='helmholtz path position index'
)
plt.gca().grid(color='.9')
plt.gca().legend()
plt.show()
```


One central motivation behind the `Collection` class is enabling users to build **compound objects**, which refer to custom classes that inherit `Collection`. They can represent complex magnet structures like magnetic encoders, motor parts, Halbach arrays, and other arrangements, and will naturally integrate into the Magpylib interface. An advanced tutorial how to sub-class `Collection` with dynamic properties and custom 3D models is given in reference `examples-compounds`.



The `Collection` class is a versatile way of grouping and manipulating multiple Magpylib objects. A basic introduction is given in {ref}`intro-collections`. Here things are explained in more detail with examples.

## Constructing collections

Collections have the attributes `children`, `sources`, `sensors` and `collections`. These attributes are ordered lists that contain objects that are added to the collection by reference (not copied). `children` returns is list of all objects in the collection. `sources` returns a list of the sources, `sensors` a list of the sensors and `collections` a list of "sub-collections" within the collection.

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

New additions are always added at the end. Add objects to an existing collection using these parameters, or the **`add`** method.

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

Collection nesting is powerful to create a self-consistent hierarchical structure, however, it is often in the way of simple construction and children access in nested trees. For this, the `children_all`, `sources_all`, `sensors_all` and `collections_all` read-only parameters, give quick access to all objects in the tree:

```{code-cell} ipython3
import magpylib as magpy

s1 = magpy.Sensor(style_label='s1')
s2 = s1.copy()
s3 = s2.copy()

# this creates anested collection
coll = s1 + s2 + s3
coll.describe(format='label')

# _all gives access to the whole tree
print([s.style.label for s in coll.sensors_all])
```

How to work with collections in a practical way is demonstrated in the introduction section {ref}`intro-collections`.

How to make complex compound objects is documented in reference `examples-compounds`.

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
extra_generic_trace = magpy.graphics.model3d.make_Cuboid(
    dimension=(104, 12, 12),
    position=(45, 0, 0),
    opacity=0.5,
)
coll.style.model3d.add_trace(extra_generic_trace)

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
