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

(examples-triangularmesh)=

# Complex shapes - The TrianglularMesh class

The field of a homogeneously charged magnet is, on the outside, the same as the field of a similarly shaped body with a magnetic surface charge. The surface charge is proportional to the projection of the magnetization vector onto the surface normal.

The `TriangularMesh` class is set up so that a magnet with an arbitrary surface can be approximated by triangular facets. Given the magnetization vector, the surface charge density is automatically computed. This source is homologous to a collection of `Triangle` objects where its mesh is defined via `vertices` and `triangles`. The `vertices` correspond to the corner points in units of \[mm\] and the `triangles` define indices triples corresponding to the the vertices coordinates of each triangle. 
Additionaly, convenient input checks, enabled by default, are implemented in order to ensure the proper constructing of a facetet magnet source, such as:
- `reorient_triangles`: checks if facets are facing outwards and flip the ones wrongly orientated
- `validate_closed`: checks if set of provided set of `vertices` and `triangles` form a closed body.
- `validate_connected`: checks if set of provided set of `vertices` and `triangles` is not disjoint.

+++ {"user_expressions": []}

```{warning}
* The automatic reorientation of triangles is the most computationally intensive check and can be deactivated. However you must be sure that all the facets are oriented correctly on your own, otherwise field calculation may be completely wrong.
Meshing tools such as the [Pyvista](https://docs.pyvista.org/) library are useful for building complex shapes but do not guarantee that the mesh is valid as a Magpylib magnet and deactivating mesh input checks may lead to unwnanted results.
* To date, there is no self-intersecting check but it is planed to be implemented in the future. Do not use the Pyvista `merge` operator to construct complex meshes, since it easily leads to self-intersection!
```

+++ {"user_expressions": []}

## Example - Dodecahedron

In order to easily pass Pyvista objects to the `TriangularMesh` constructor a special `classmethod` has been implemented. This will take care of the transforming of a `polydata` object into an internal `TriangularMesh` magnet.

+++ {"user_expressions": []}

```{note}
The Pyvista library used in the following examples does not get installed with Magpylib. You will need to [install](https://docs.pyvista.org/getting-started/installation.html) it on your own if you want to run them.
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

# print input checks attributes
print("is closed:", dodec.is_closed)
print("is connected:", dodec.is_connected)
print("is reoriented:", dodec.is_reoriented)
dodec.show()
```

+++ {"user_expressions": []}

We can now add a sensor and plot the B-field value along the sensor path

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
sens.rotate_from_angax(angle=np.linspace(0, 270, 50), axis="z", start=0, anchor=0)

# get B-field and plot it along the 3d model
B = dodec.getB(sens)

# define matplotlib figure and axes
fig = plt.figure(figsize=(12, 5))
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
plt.show()
```

+++ {"user_expressions": []}

## Example - ConvexHull

The `from_ConvexHull_points` classmethod has been added to the `TriangularMesh` constructor to easily build a convex body from a point could. In The following example a tetrahedron is built taking advantage of this feature. Under the hood the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) class is used. Note that the Scipy method will produce random facet orientations most of the time as shown below if reorientation is disabled.

```{code-cell} ipython3
import magpylib as magpy

# create tetrahedron from point cloud
points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
tetra_from_ConvexHull = magpy.magnet.TriangularMesh.from_ConvexHull_points(
    magnetization=(0, 0, 1000), points=points
)

tetra_from_ConvexHull.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_size=1.5,
)
```

+++ {"user_expressions": []}

Lets see what happens if we disable the triangle reorientation. Like shown below the orientation of the facets is wrong. In this case all triangles are pointing inwards.

```{code-cell} ipython3
import magpylib as magpy

# create tetrahedron from point cloud
points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
tetra_from_ConvexHull = magpy.magnet.TriangularMesh.from_ConvexHull_points(
    magnetization=(0, 0, 1000), points=points, reorient_triangles=False
)

tetra_from_ConvexHull.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_offset=0,
    style_orientation_size=1.5,
)
```

+++ {"user_expressions": []}

## Example - Open Cube

+++ {"user_expressions": []}

In some cases the produced mesh may be open and if we disable the corresponding input check, the object initialization goes through. By displaying the object it is possible to highlight the open edges. The same idea holds true for the `validate_connected` check.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

# get triangles and vertices from a cuboid mesh from
kw = magpy.graphics.model3d.make_Cuboid()["kwargs"]
triangles = np.array([kw[k] for k in "ijk"]).T
vertices = np.array([kw[k] for k in "xyz"]).T

# open the mesh by removing the 2nd triangle
triangles = np.delete(triangles, 1, axis=0)

# create faceted cuboid magnet with open edges
cube = magpy.magnet.TriangularMesh(
    magnetization=(1000, 0, 0),
    vertices=vertices,
    triangles=triangles,
    validate_closed=False,  # disable check
)
print("is closed:", cube.is_closed)
cube.show(backend="plotly")
```

+++ {"user_expressions": []}

## Example - Boolean operation

+++ {"user_expressions": []}

With the help of the Pyvista package it is possible to build even more complex shapes by boolean operations. However this comes with some caveats and will require some refinement in order to produce a clean mesh as shown below 

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

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.85)

# triangulate the dodecahedron
dodec = pv.Dodecahedron().triangulate()

# perform boolean operation
obj = dodec.boolean_difference(sphere)

# clean the mesh to avoid disjoint parts
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

# triangulate and subdivide the dodecahedron
dodec = pv.Dodecahedron().triangulate()

# subdivide before boolean to avoid open edges
dodec = dodec.subdivide(2)

# perform boolean operation
obj = dodec.boolean_difference(sphere)

# clean the mesh
obj = obj.clean()

# use the `from_pyvista` classmethod to construct our magnet
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    style_label="Dodecahedron cut by Sphere",
)

magnet.show(backend="plotly")
```

```{note}
The mesh cleaning methods as described are highly dependent on the Pyvista library behavior and may evolve in the future. This is independent of Magpylib and may not work for some other shapes. Other library can also be used to create mesh objects and the provided input checks should avoid creating invalid `TriangularMesh` magnets.
