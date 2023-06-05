---
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

+++ {"user_expressions": []}

```{versionadded} 4.3
The `TriangularMesh` class
```

+++ {"user_expressions": []}

(examples-triangularmesh)=

# Complex shapes - The TrianglularMesh class

The outside field of a homogeneously charged magnet is the same as the field of a similarly shaped body with a magnetic surface charge. In turn, the surface charge is proportional to the projection of the magnetization vector onto the surface normal.

The `TriangularMesh` class is set up so that a magnet with an arbitrary surface can be approximated by triangular faces. Given the magnetization vector, the surface charge density is automatically computed and summed up over all triangles. This source is homologous to a collection of `Triangle` objects where its geometry is defined via `vertices` and `triangles`. The `vertices` correspond to the corner points in units of \[mm\] and the `triangles` define indices triples corresponding to the the vertices coordinates of each triangle.
Additionally, useful input checks, enabled by default, are implemented in order to ensure the validity of a faceted magnet source, such as:
- `reorient_triangles`: checks if faces are facing outwards and flip the ones wrongly oriented
- `validate_closed`: checks if set of provided set of `vertices` and `triangles` form a closed body.
- `validate_connected`: checks if set of provided set of `vertices` and `triangles` is not disjoint.

+++ {"user_expressions": []}

```{caution}
* Input checks can be computationally intensive, especially for the automatic reorientation of triangles but can be deactivated. However, one must be sure that all the faces are oriented correctly before any field calculation, otherwise it may yield erroneous values.

* Meshing tools such as the [Pyvista](https://docs.pyvista.org/) library can be very convenient for building complex shapes but no guarantee is given that the produced mesh is a valid as a Magpylib `TriangularMesh` input. Deactivating mesh input checks may lead to unwanted results.

* To date, there is no self-intersecting check but it is planed to be implemented in the future. Do not use the Pyvista `merge` operator to construct complex meshes, since it easily leads to self-intersection!
```

+++ {"user_expressions": []}

## Example - Faceted Tetrahedron

```{code-cell} ipython3
import magpylib as magpy

# create faceted tetrahedron from vertices and triangles
tetra_facet = magpy.magnet.TriangularMesh(
    magnetization=(0, 0, 1000),
    vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    triangles=[[2, 1, 0], [3, 0, 1], [3, 2, 0], [3, 1, 2]],
)

# print input checks attributes
print("is closed:", tetra_facet.is_closed)
print("is connected:", tetra_facet.is_connected)
print("is reoriented:", tetra_facet.is_reoriented)

tetra_facet.show()
```

+++ {"user_expressions": []}

## Example - Dodecahedron from pyvista

In order to easily pass Pyvista objects to the `TriangularMesh` constructor a special `classmethod` has been implemented. This will take care of the transforming of a `polydata` object into an internal `TriangularMesh` magnet.

+++ {"user_expressions": []}

```{note}
The Pyvista library used in the following examples does not get installed with Magpylib. You will need to [install](https://docs.pyvista.org/getting-started/installation.html) it on your own if you want to run them on your machine.
```

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=dodec_mesh,
)

dodec.show()
```

+++ {"user_expressions": []}

We can now add a sensor and plot the B-field value along the sensor path.

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

# create TriangularMesh object
dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(1000, 0, 0),
    polydata=dodec_mesh,
)

# add sensor and rotate around source
sens = magpy.Sensor(position=(2, 0, 0))
sens.rotate_from_angax(angle=np.linspace(0, 320, 50), axis="z", start=0, anchor=0)

# get B-field and plot it along the 3d model
B = dodec.getB(sens)

# define matplotlib figure and axes
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(
    121,
    projection="3d",
    azim=-80,
    elev=15,
)
ax2 = fig.add_subplot(
    122,
)

# plot 3D model into first axis
magpy.show(dodec, sens, canvas=ax1)
ax2.plot(B, label=["Bx", "By", "Bz"])


# plot styling
ax1.set(
    title="3D model",
    aspect="equal",
)
ax2.set(
    title="Dodecahedron field",
    xlabel="path index",
    ylabel="B-field [mT]",
    aspect=1,
)

plt.gca().grid(color=".9")
plt.gca().legend()
plt.tight_layout()
plt.show()
```

+++ {"user_expressions": []}

## Example - Trapezoidal prism from ConvexHull

The `from_ConvexHull` classmethod has been added to the `TriangularMesh` constructor to easily build a convex body from a point could. In The following example a trapezoidal prism is built taking advantage of this feature. Under the hood the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) class is used. Note that the Scipy method does not guarantee correct facet orientations if `reorient_triangles` is disabled.

```{code-cell} ipython3
import magpylib as magpy

# create trapezoidal prism from point cloud
points = [[-2,-2, 0], [-2,2,0], [2,-2,0], [2,2,0], [-1,-1, 3], [-1,1,3], [1,-1,3], [1,1,3]]
cube_from_ConvexHull = magpy.magnet.TriangularMesh.from_ConvexHull(
    magnetization=(0, 0, 1000),
    points=points,
)

# show and style the created source
cube_from_ConvexHull.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_size=1.5,
)
```

+++ {"user_expressions": []}

Lets see what happens if we disable the triangle reorientation. Like shown below, the orientation of the mesh is wrong and some triangles are pointing inwards.

```{code-cell} ipython3
import magpylib as magpy

# create trapezoidal prism from point cloud
points = [[-2,-2, 0], [-2,2,0], [2,-2,0], [2,2,0], [-1,-1, 3], [-1,1,3], [1,-1,3], [1,1,3]]
cube_from_ConvexHull = magpy.magnet.TriangularMesh.from_ConvexHull(
    magnetization=(0, 0, 1000),
    points=points,
    reorient_triangles=False,
)

# show and style the created source
cube_from_ConvexHull.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_size=1.5,
)
```

+++ {"user_expressions": []}

## Example - Open Prism from mesh

+++ {"user_expressions": []}

In some cases it may be desirable to create an open mesh. By disabling the corresponding `validate_closed` input check, the object initialization will still pass. It is yet possible to highlight the open edges with the `show` function. The same idea holds true for the `validate_connected` check.

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
cube = magpy.magnet.TriangularMesh.from_triangles(
    magnetization=(0, 0, 1000), # overrides triangles magnetization
    faces=[top, bottom],
    validate_closed=False,  # disable check
    validate_connected=False,  # disable check
    style_label="Open prism",
)
cube.style.magnetization.mode = "arrow"

print("is closed:", cube.is_closed)
print("is connected:", cube.is_connected)
cube.show(backend="plotly")
```

+++ {"user_expressions": []}

```{seealso}
For more information about `Triangle` objects and when to use them see {ref}`examples-triangle`
```

+++ {"user_expressions": []}

## Example - Boolean operation

+++ {"user_expressions": []}

With the help of the Pyvista package it is possible to build even more complex shapes with boolean operations. However this comes with some caveats and will require some refinement in order to produce a clean mesh, as shown below

+++ {"user_expressions": []}

* Deactivate all checks to see the mesh issues

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.85)
dodec = pv.Dodecahedron().triangulate()
obj = dodec.boolean_difference(sphere)

# use the `from_pyvista` classmethod to construct our magnet
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    validate_connected=False,  # disable
    validate_closed=False,  # disable
    style_label="Dodecahedron cut by Sphere",
)

magnet.show(backend="plotly")
```

+++ {"user_expressions": []}

* Add mesh cleaning which resolves the disjoint part problem

+++ {"user_expressions": []}

```{tip}
If the `TriangularMesh` object has disjoint parts, the magnetization coloring is overruled and colors cycle through a predefined colorsequence (`style.mesh.disjoint.colorsequence`) to match each subset.
```

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.85)

# triangulate the dodecahedron
dodec = pv.Dodecahedron().triangulate()

# perform boolean operation
obj = dodec.boolean_difference(sphere)

# Additional step -> clean the mesh to avoid disjoint parts
obj = obj.clean()

magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    validate_closed=False,  # disabled
    style_label="Dodecahedron cut by Sphere",
)

magnet.show(backend="plotly")
```

+++ {"user_expressions": []}

* Perfom mesh subdivision before boolean operation to solve the lasting open mesh problem

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.85)

# triangulate the dodecahedron
dodec = pv.Dodecahedron().triangulate()

# Additional step -> subdivide before boolean operation, to avoid open edges
dodec = dodec.subdivide(2)

# perform boolean operation
obj = dodec.boolean_difference(sphere)

# Additional step ->  clean the mesh
obj = obj.clean()

# use the `from_pyvista` classmethod to construct our magnet
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    style_label="Dodecahedron cut by Sphere",
)

magnet.show(backend="plotly")
```

+++ {"user_expressions": []}

```{hint}
The mesh cleaning methods as described are only rely on the Pyvista library implementations and may evolve in the future. This process is independent of Magpylib and may not work for some other shapes. In deed, other library such as [Open3D](http://www.open3d.org/docs/release/index.html#python-api-index) can also be used to create mesh objects. The provided set of input checks provided by Magpylib should help make sure created `TriangularMesh` objects are valid magnet sources.
