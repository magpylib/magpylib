(docs-classes)=
# The Magpylib Classes

In Magpylib's [object oriented interface](docs-fieldcomp-oo) magnetic field **sources** (generate the field) and **observers** (read the field) are created as Python objects with various defining attributes and methods.

## Base properties

The following basic properties are shared by all Magpylib classes:

* The <span style="color: orange">**position**</span> and <span style="color: orange">**orientation**</span> attributes describe the object placement in the global coordinate system.

* The <span style="color: orange">**move()**</span> and <span style="color: orange">**rotate()**</span> methods enable relative object positioning.

* The <span style="color: orange">**reset_path()**</span> method sets position and orientation to default values.

* The <span style="color: orange">**barycenter**</span> property returns the object barycenter (often the same as position).

See {ref}`docs-position` for more information on these features.


* The <span style="color: orange">**style**</span> attribute includes all settings for graphical object representation.

* The <span style="color: orange">**show()**</span> method gives quick access to the graphical representation.

See {ref}`guide-graphics` for more information on graphic output, default styles and customization possibilities.

* The <span style="color: orange">**getB()**</span>, <span style="color: orange">**getH()**</span>, <span style="color: orange">**getJ()**</span> and <span style="color: orange">**getM()**</span> methods give quick access to field computation.

See {ref}`docs-fieldcomp` for more information.


* The <span style="color: orange">**parent**</span> attribute references a [Collection](guide-docs-classes-collections) that the object is part of.

* The <span style="color: orange">**copy()**</span> method creates a clone of any object where selected properties, given by kwargs, are modified.

* The <span style="color: orange">**describe()**</span> method provides a brief description of the object and returns the unique object id.


---------------------------------------------


## Local and global coordinates

::::{grid} 2
:::{grid-item}
:columns: 9
Magpylib objects span a local reference frame, and all object properties are defined within this frame, for example the vertices of a `Tetrahedron` magnet. The position and orientation attributes describe how the local frame lies within the global coordinates. The two frames coincide by default, when `position=(0,0,0)` and `orientation=None` (=unit rotation). The `position` and `orientation` attributes are described in detail in {ref}`docs-position`.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_global_local.png)
:::
::::


---------------------------------------------


(docu-magnet-classes)=
## Magnet classes

All magnets are sources. They have the <span style="color: orange">**polarization**</span> attribute which is of the format $\vec{J}=(J_x, J_y, J_z)$ and denotes a homogeneous magnetic polarization vector in the local object coordinates in units of T. Alternatively, the magnetization vector can be set via the  <span style="color: orange">**magnetization**</span> attribute of the format $\vec{M}=(M_x, M_y, M_z)$. These two parameters are codependent and Magpylib ensures that they stay in sync via the relation $\vec{J}=\mu_0\cdot\vec{M}$. Information on how this is related to material properties from data sheets is found in {ref}`examples-tutorial-modelling-magnets`.


### Cuboid
```python
magpylib.magnet.Cuboid(position, orientation, dimension, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Cuboid` objects represent magnets with cuboid shape. The <span style="color: orange">**dimension**</span> attribute has the format $(a,b,c)$ and denotes the sides of the cuboid units of meter. The center of the cuboid lies in the origin of the local coordinates, and the sides are parallel to the coordinate axes.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_cuboid.png)
:::
::::


### Cylinder
```python
magpylib.magnet.Cylinder(position, orientation, dimension, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Cylinder` objects represent magnets with cylindrical shape. The <span style="color: orange">**dimension**</span> attribute has the format $(d,h)$ and denotes diameter and height of the cylinder in units of meter. The center of the cylinder lies in the origin of the local coordinates, and the cylinder axis coincides with the z-axis.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_cylinder.png)
:::
::::


### CylinderSegment
```python
magpylib.magnet.CylinderSegment(position, orientation, dimension, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`CylinderSegment` objects represent magnets with the shape of a cylindrical ring section. The <span style="color: orange">**dimension**</span> attribute has the format $(r_1,r_2,h,\varphi_1,\varphi_2)$ and denotes inner radius, outer radius and height in units of meter, and the two section angles $\varphi_1<\varphi_2$ in °. The center of the full cylinder lies in the origin of the local coordinates, and the cylinder axis coincides with the z-axis.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_cylindersegment.png)
:::
:::{grid-item}
:columns: 12
**Info:** When the cylinder section angles span 360°, then the much faster `Cylinder` methods are used for the field computation.
:::
::::


### Sphere
```python
magpylib.magnet.Sphere(position, orientation, diameter, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Sphere` objects represent magnets of spherical shape. The <span style="color: orange">**diameter**</span> attribute is the sphere diameter $d$ in units of meter. The center of the sphere lies in the origin of the local coordinates.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_sphere.png)
:::
::::


### Tetrahedron
```python
magpylib.magnet.Tetrahedron(position, orientation, vertices, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Tetrahedron` objects represent magnets of tetrahedral shape. The <span style="color: orange">**vertices**</span> attribute stores the four corner points $(\vec{P}_1, \vec{P}_2, \vec{P}_3, \vec{P}_4)$ in the local object coordinates in units of m.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_tetra.png)
:::
:::{grid-item}
:columns: 12
**Info:** The `Tetrahedron` field is computed from four `Triangle` fields.
:::
::::

(docu-magpylib-api-trimesh)=

### TriangularMesh
```python
magpylib.magnet.TriangularMesh(position, orientation, vertices, faces, polarization, magnetization, check_open, check_disconnected, check_selfintersecting, reorient_faces, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`TriangularMesh` objects represent magnets with surface given by a triangular mesh. The mesh is defined by the <span style="color: orange">**vertices**</span> attribute, an array of all unique corner points $(\vec{P}_1, \vec{P}_2, ...)$ in units of meter, and the <span style="color: orange">**faces**</span> attribute, which is an array of index-triplets that define individual faces $(\vec{F}_1, \vec{F}_2, ...)$. The property <span style="color: orange">**mesh**</span> returns an array of all faces as point-triples $[(\vec{P}_1^1, \vec{P}_2^1, \vec{P}_3^1), (\vec{P}_1^2, \vec{P}_2^2, \vec{P}_3^2), ...]$.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_trimesh.png)
:::
:::{grid-item}
:columns: 12
At initialization the mesh integrity is automatically checked, and all faces are reoriented to point outwards. These actions are controlled via the kwargs
* <span style="color: orange">**check_open**</span>
* <span style="color: orange">**check_disconnected**</span>
* <span style="color: orange">**check_selfintersecting**</span>
* <span style="color: orange">**reorient_faces**</span>

which are all by default set to `"warn"`. Options are `"skip"` (don't perform check), `"ignore"` (ignore if check fails), `"warn"` (warn if check fails), `"raise"` (raise error if check fails).

Results of the checks are stored in the following object attributes
* <span style="color: orange">**status_open**</span> can be `True`, `False` or `None` (unchecked)
* <span style="color: orange">**status_open_data**</span> contains an array of open edges
* <span style="color: orange">**status_disconnected**</span> can be `True`, `False` or `None` (unchecked)
* <span style="color: orange">**status_disconnected_data**</span> contains an array of mesh parts
* <span style="color: orange">**status_selfintersecting**</span> can be `True`, `None` or `None` (unchecked)
* <span style="color: orange">**status_selfintersecting_data**</span> contains an array of self-intersecting faces
* <span style="color: orange">**status_reoriented**</span> can be `True` or `False`

The checks can also be performed after initialization using the methods
* <span style="color: orange">**check_open()**</span>
* <span style="color: orange">**check_disconnected()**</span>
* <span style="color: orange">**check_selfintersecting()**</span>
* <span style="color: orange">**reorient_faces()**</span>

The following class methods enable easy mesh creating and mesh loading.

* <span style="color: orange">**TriangularMesh.from_mesh()**</span> generates a `TriangularMesh` objects from the input <span style="color: orange">**mesh**</span>, which is an array in the mesh format $[(\vec{P}_1^1, \vec{P}_2^1, \vec{P}_3^1), (\vec{P}_1^2, \vec{P}_2^2, \vec{P}_3^2), ...]$.
* <span style="color: orange">**TriangularMesh.from_ConvexHull()**</span> generates a `TriangularMesh` object from the input <span style="color: orange">**points**</span>, which is an array of positions $(\vec{P}_1, \vec{P}_2, \vec{P}_3, ...)$ from which the convex Hull is computed via the [Scipy ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) implementation.
* <span style="color: orange">**TriangularMesh.from_triangles()**</span> generates a `TriangularMesh` object from the input <span style="color: orange">**triangles**</span>, which is a list or a `Collection` of `Triangle` objects.
* <span style="color: orange">**TriangularMesh.from_pyvista()**</span> generates a `TriangularMesh` object from the input <span style="color: orange">**polydata**</span>, which is a [Pyvista PolyData](https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyData.html) object.

The method <span style="color: orange">**to_TriangleCollection()**</span> transforms a `TriangularMesh` object into a `Collection` of `Triangle` objects.

**Info:** While the checks may be disabled, the field computation guarantees correct results only if the mesh is closed, connected, not self-intersecting and all faces are oriented outwards. Examples of working with the `TriangularMesh` class are found in {ref}`examples-shapes-triangle` and in {ref}`examples-shapes-pyvista`.
:::
::::


---------------------------------------------


## Current classes

All currents are sources. Current objects have the <span style="color: orange">**current**</span> attribute which is a scalar that denotes the electrical current in units of ampere.

### Circle
```python
magpylib.current.Circle(position, orientation, diameter, current, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Circle` objects represent circular line current loops. The <span style="color: orange">**diameter**</span> attribute is the loop diameter $d$ in units of meter. The loop lies in the xy-plane with it's center in the origin of the local coordinates.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_loop.png)
:::
::::

### Polyline
```python
magpylib.current.Polyline(position, orientation, vertices, current, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Polyline` objects represent line current segments where the electric current flows in straight lines from vertex to vertex. The <span style="color: orange">**vertices**</span> attribute is a vector of all vertices $(\vec{P}_1, \vec{P}_2, ...)$ given in the local coordinates in units of meter.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_line.png)
:::
::::

---------------------------------------------

## Miscellaneous classes

There are classes listed hereon that function as sources, but they do not represent physical magnets or current distributions.


### Dipole
```python
magpylib.misc.Dipole(position, orientation, moment, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Dipole` objects represent magnetic dipole moments with the <span style="color: orange">**moment**</span> attribute that describes the magnetic dipole moment $\vec{m}=(m_x,m_y,m_z)$ in SI-units of Am², which lies in the origin of the local coordinates.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_dipole.png)
:::
:::{grid-item}
:columns: 12
**Info:** The total dipole moment of a homogeneous magnet with body volume $V$ is given by $\vec{m}=\vec{M}\cdot V$.
:::
::::


### Triangle
```python
magpylib.misc.Triangle(position, orientation, vertices, polarization, magnetization, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Triangle` objects represent triangular surfaces with homogeneous charge density given by the projection of the polarization or magnetization vector onto the surface normal. The attributes <span style="color: orange">**polarization**</span> and <span style="color: orange">**magnetization**</span> are treated similar as by the {ref}`docu-magnet-classes`. The <span style="color: orange">**vertices**</span> attribute is a set of the three triangle corners $(\vec{P}_1, \vec{P}_2, \vec{P}_3)$ in units of meter in the local coordinates.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_triangle.png)
:::
:::{grid-item}
:columns: 12
**Info:** When multiple Triangles with similar magnetization/polarization vectors form a closed surface, and all their orientations (right-hand-rule) point outwards, their total H-field is equivalent to the field of a homogeneous magnet of the same shape. In this case, the B-field is only correct on the outside of the body. On the inside the polarization must be added to the field. This is demonstrated in the tutorial {ref}`examples-shapes-triangle`.
:::
::::

(guide-docs-classes-custom-source)=
### CustomSource
```python
magpylib.misc.CustomSource(field_func, position, orientation, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
The `CustomSource` class is used to create user defined sources provided with with custom field computation functions. The argument <span style="color: orange">**field_func**</span> takes a function that is then automatically called for the field computation. This custom field function is treated like a [core function](docs-field-core). It must have the positional arguments `field` with values `"B"` or `"H"`, and `observers` (must accept array with shape (n,3)) and return the B-field and the H-field with a similar shape.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_custom.png)
:::
:::{grid-item}
:columns: 12
**Info:** A tutorial {ref}`examples-tutorial-custom` is found in the examples.
:::
::::


---------------------------------------------


## Sensor
```python
magpylib.Sensor(position, orientation, pixel, handedness, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
`Sensor` objects represent observers of the magnetic field and can be used as Magpylib `observers` input for magnetic field computation. The <span style="color: orange">**pixel**</span> attribute is an array of positions $(\vec{P}_1, \vec{P}_2, ...)$ provided in units of meter in the local sensor coordinates. A sensor returns the magnetic field at these pixel positions. By default `pixel=(0,0,0)` and the sensor simply returns the field at it's position. The <span style="color: orange">**handedness**</span> attribute can be `"left"` or `"right"` (default) to set a left- or right-handed sensor coordinate system for the field computation.
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_sensor.png)
:::
:::{grid-item}
:columns: 12
**Info:** Sensors can have their own position and orientation and enable easy relative positioning between sources and observers. The field is always computed in the reference frame of the sensor, which might itself be moving in the global coordinate system. Magpylib sensors can be understood as perfect magnetic field sensors with infinitesimally sensitive elements. An example how to use sensors is given in {ref}`examples-tutorial-field-computation-sensors`.
:::
::::


---------------------------------------------


(guide-docs-classes-collections)=
## Collection
```python
magpylib.Collection(*children, position, orientation, override_parent, style)
```

::::{grid} 2
:::{grid-item}
:columns: 9
A `Collection` is a group of Magpylib objects that is used for common manipulation. All these objects are stored by reference in the <span style="color: orange">**children**</span> attribute. The collection becomes the <span style="color: orange">**parent**</span> of the object. An object can only have one parent. There are several options for accessing only specific children via the following properties

* <span style="color: orange">**sources**</span>: return only sources
* <span style="color: orange">**observers**</span>: return only observers
* <span style="color: orange">**collections**</span>: return only collections
* <span style="color: orange">**sources_all**</span>: return all sources, including the ones from sub-collections
* <span style="color: orange">**observers_all**</span>: return all observers, including the ones from sub-collections
* <span style="color: orange">**collections_all**</span>: return all collections, including the ones from sub-collections

Additional methods for adding and removing children:

- <span style="color: orange">**add()**</span>: Add an object to the collection
- <span style="color: orange">**remove()**</span>: Remove an object from the collection
:::
:::{grid-item}
:columns: 3
![](../../../_static/images/docu_classes_init_collection.png)
:::
:::{grid-item}
:columns: 12
**Info:** A collection object has its own `position` and `orientation` attributes and spans a local reference frame for all its children. An operation applied to a collection moves the frame and is individually applied to all children such that their relative position in the local reference frame is maintained. This means that the collection functions as a container for manipulation, but child position and orientation are always updated in the global coordinate system. After being added to a collection, it is still possible to manipulate the individual children, which will also move them to a new relative position in the collection frame.

Collections have **format** as an additional argument for **describe()** method. Default value is `format="type+id+label"`. Any combination of `"type"`, `"id"`, and `"label"` is allowed.

A tutorial {ref}`examples-tutorial-collection` is provided in the example examples.
:::
::::
