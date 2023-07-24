---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(docu-classes)=

# The Magpylib Classes


(intro-magpylib-objects)=

## The Magpylib classes

The most convenient way of working with Magpylib is through the **object oriented interface**. Magpylib objects represent magnetic field sources, sensors and collections with various defining attributes and methods. By default all objects are initialized with `position=(0,0,0)`, `orientation=None`, and default graphic `style` settings. Additional `**kwargs` mostly include style shortcuts, see {ref}`intro-graphic-output`. The following classes are implemented:

**Magnets**

All magnet objects have the `magnetization` attribute which must be of the format $(m_x, m_y, m_z)$ and denotes the homogeneous magnetization vector in the local object coordinates in units of mT. It is often referred to as the remanence ($B_r=\mu_0 M$) in material data sheets. All magnets can be used as Magpylib `sources` input.

- **`Cuboid`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cuboid shape. `dimension` has the format $(a,b,c)$ and denotes the sides of the cuboid in units of mm. By default the center of the cuboid lies in the origin of the global coordinates, and the sides are parallel to the coordinate axes.

- **`Cylinder`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cylindrical shape. `dimension` has the format $(d,h)$ and denotes diameter and height of the cylinder in units of mm. By default the center of the cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`CylinderSegment`**`(magnetization, dimension, position, orientation, style)` represents a magnet with the shape of a cylindrical ring section. `dimension` has the format $(r_1,r_2,h,\varphi_1,\varphi_2)$ and denotes inner radius, outer radius and height in units of mm and the two section angles $\varphi_1<\varphi_2$ in deg. By default the center of the full cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`Sphere`**`(magnetization, diameter, position, orientation, style)` represents a magnet of spherical shape. `diameter` is the sphere diameter $d$ in units of mm. By default the center of the sphere lies in the origin of the global coordinates.

- **`Tetrahedron`**`(magnetization, vertices, position, orientation, style)` represents a magnet of tetrahedral shape. `vertices` corresponds to the four corner points in units of mm. By default the vertex positions coincide in the local object coordinates and the global coordinates.

- **`TriangularMesh`**`(magnetization, vertices, faces, position, orientation, validate_closed, validate_connected, reorient_faces, style)` represents a magnet with surface given by a triangular mesh. The `vertices` correspond to the corner points in units of mm and the `faces` are index-triplets for each face. By default, input checks are performed to see if the mesh is closed, connected and if its faces are correctly oriented. At initialization, the vertex positions coincide in the local object coordinates and the global coordinates.

**Currents**

All current objects have the `current` attribute which must be a scalar $i_0$ and denotes the electrical current in units of A. All currents can be used as Magpylib `sources` input.

- **`Loop`**`(current, diameter, position, orientation, style)` represents a circular current loop where `diameter` is the loop diameter $d$ in units of mm. By default the loop lies in the xy-plane with it's center in the origin of the global coordinates.

- **`Line`**`(current, vertices, position, orientation, style)` represents electrical current segments that flow in a straight line from vertex to vertex. By default the vertex positions coincide in the local object coordinates and the global coordinates.

**Other**

- **`Dipole`**`(moment, position, orientation, style)` represents a magnetic dipole moment with moment $(m_x,m_y,m_z)$ given in mT mm³. For homogeneous magnets the relation moment=magnetization$\times$volume holds. Can be used as Magpylib `sources` input.

- **`Triangle`**`(magnetization, vertices, position, orientation, style)` represents a triangular surface with a homogeneous charge given by the projection of the `magnetization` vector onto the surface normal. `vertices` is a set of the three corners given in mm³. When multiple Triangles form a closed surface, on the outside their total magnetic field correponds to the one of a homogeneously charged magnet.

- **`CustomSource`**`(field_func, position, orientation, style)` is used to create user defined custom sources with their own field functions. Can be used as Magpylib `sources` input.

- **`Sensor`**`(position, pixel, orientation, style)` represents a 3D magnetic field sensor. The field is evaluated at the given pixel positions. By default (`pixel=(0,0,0)`) the pixel position coincide in the local object coordinates and the global coordinates. Can be used as Magpylib `observers` input.

- **`Collection`**`(*children, position, orientation, style)` is a group of source and sensor objects (children) that is used for common manipulation. Depending on the children, a collection can be used as Magpylib `sources` and/or `observers` input.


```{versionadded} 4.2
The `Triangle` class
```

(examples-triangle)=

# Complex shapes - Triangle

The field of a homogeneously charged magnet is, on the outside, the same as the field of a similarly shaped body with a magnetic surface charge. The surface charge is proportional to the projection of the magnetization vector onto the surface normal.

The `Triangle` class is set up so that it can easily be used for this purpose. Arbitrary surfaces are easily approximated by triangles, and given the magnetization vector, the surface charge density is automatically computed. One must be very careful to orient the triangles correctly, with surface normal vectors pointing outwards (right-hand-rule). The resulting H-field of such a collection is correct, but the B-field is only correct on the outside of the body. On the inside the magnetization must be added to the field.




```{versionadded} 4.3
The `TriangularMesh` class
```

(examples-triangularmesh)=

# Complex shapes - TrianglularMesh

The `TriangularMesh` class is used to create magnets from triangular surface meshes, instead of assembling them from individual `Triangle` objects as described in {ref}`examples-triangle`. This class is initialized with the `vertices` (an array_like of positions) and the `faces` (an array_like of index triplets) inputs. In addition, a set of useful classmethods enables initialization from various inputs:

- `TriangularMesh.from_mesh()`: from an array_like of triplets of vertices
- `TriangularMesh.from_triangles()`: from a list of `Triangle` objects
- `TriangularMesh.from_ConvexHull()`: from the convex hull of a given point cloud
- `TriangularMesh.from_pyvista()`: from a Pvista `PolyData` object

In contrast to a `Collection` of `Triangle` objects the `TriangularMesh` class performs several important checks at initialization by default to ensure that the given triangular mesh can form a proper magnet:

- `check_open`: checks if given mesh forms a closed surface
- `check_disconnected`: checks if given mesh is connected
- `check_selfintersecting`: checks if given mesh is self-intersecting
- `reorient_faces`: checks if faces are oriented outwards, and flips the ones wrongly oriented. This works only if the mesh is closed.

All four checks will throw warnings by default if the mesh is open, disconnected, self-intersecting, or cannot be reoriented. Four options enable error handling: `"skip"` (=`False`), `"ignore"`, `"warn"` (=default=`True`), `"raise"`. If skipped at initialization, the checks can be performed by hand via respective methods.

The mesh status is set by the checks, and can be viewed via the properties `status_open`, `status_disconnected` and `status_reoriented` with possible values `None`, `True`, `False`. Problems of the mesh (e.g. open edges) are stored in `status_open_data` and `status_disconnected_data`. Such problems can be viewed with `show`.

```{caution}
* `getB` and `getH` compute the fields correctly only if the mesh is closed, not self-intersecting, and all faces are properly oriented outwards.

* Input checks and face reorientation can be computationally expensive. The checks can be individually deactivated by setting `reorient_faces="skip"`, `check_open="skip"`, `check_disconnected="skip"`, and `check_selfintersecting="skip"` at initialization of `TriangularMesh` objects. The checks can also be performed by hand after initialization.

* Meshing tools such as the [Pyvista](https://docs.pyvista.org/) library can be very convenient for building complex shapes, but often do not guarantee that the mesh is properly closed or connected.

* Meshing tools often create meshes with a lot of faces, especially when working with curved surfaces. Keep in mind that the field computation takes of the order of a few microseconds per observer position per face, and that RAM is a limited ressource.
```

## Example - Tetrahedron magnet

```{code-cell} ipython3
import magpylib as magpy

# create faceted tetrahedron from vertices and faces
tmesh_tetra = magpy.magnet.TriangularMesh(
    magnetization=(0, 0, 1000),
    vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    faces=((2, 1, 0), (3, 0, 1), (3, 2, 0), (3, 1, 2)),
)

# print mesh status
print("mesh status open:", tmesh_tetra.status_open)
print("mesh status disconnected:", tmesh_tetra.status_disconnected)
print("mesh status selfintersecting:", tmesh_tetra.status_selfintersecting)
print("mesh status reoriented:", tmesh_tetra.status_reoriented)

tmesh_tetra.show()
```



## Prism magnet with open mesh

In some cases it may be desirable to generate a `TriangularMesh` object from an open mesh, as described in {ref}`examples-triangle`. In this case one has to be extremely careful because one cannot rely on the checks. Not to generate warnings or error messages, these checks can be disabled with `"skip"` or their outcome can be ignored with `"ignore"`. The `show` function can be used to view open edges and disconnected parts. In the following example we generate such an open mesh directly from `Triangle` objects.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

top = magpy.misc.Triangle(
    magnetization=(1000,0,0),
    vertices= ((-1,-1,1), (1,-1,1), (0,2,1)),
)
bottom = magpy.misc.Triangle(
    magnetization=(1000,0,0),
    vertices= ((-1,-1,-1), (0,2,-1), (1,-1,-1)),
)

# create faceted prism with open edges
prism = magpy.magnet.TriangularMesh.from_triangles(
    magnetization=(0, 0, 1000),   # overrides triangles magnetization
    triangles=[top, bottom],
    check_open="ignore",        # check but ignore open mesh
    check_disconnected="ignore",     # check but ignore disconnected mesh
    reorient_faces="ignore",      # check but ignore non-orientable mesh
    style_label="Open prism",
)
prism.style.magnetization.mode = "arrow"

print("mesh status open:", prism.status_open)
print("mesh status disconnected:", prism.status_disconnected)
print("mesh status self-intersecting:", prism.status_selfintersecting)
print("mesh status reoriented:", prism.status_reoriented)

prism.show(
    backend="plotly",
    style_mesh_open_show=True,
    style_mesh_disconnected_show=True,
)
```



# Collections


(intro-collections)=

## Collections

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

A collection object has its own `position` and `orientation` attributes and spans a local reference frame for all its children. An operation applied to a collection moves the frame, and is individually applied to all children such that their relative position in the local reference frame is maintained. This means that the collection functions only as a container for manipulation, but child position and orientation are always updated in the global coordinate system. After being added to a collection, it is still possible to manipulate the individual children, which will also move them to a new relative position in the collection frame.

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



(examples-collections-construction)=

# Collections

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
