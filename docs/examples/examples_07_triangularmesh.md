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

```{versionadded} 4.3
The `TriangularMesh` class
```

(examples-triangularmesh)=

# Complex shapes - The TrianglularMesh class

The `TriangularMesh` class is used to create magnets from a triangular surface meshes, instead of assembling them from individual `Triangle` objects as described in {ref}`examples-triangle`. In contrast to a `Collection` of `Triangle` objects the `TriangularMesh` class performs several important checks at initialization to ensure that the given triangular mesh forms a proper magnet:

- `reorient_faces`: checks if faces are facing outwards, and flips the ones wrongly oriented
- `validate_closed`: checks if given mesh is closed.
- `validate_connected`: checks if given mesh is connected.

```{caution}
* Only if the mesh is closed and all faces are properly oriented outwards it is possible to compute the magnetic fields B and H correctly (requires inside-outside checks).

* Input checks can be computationally expensive, especially for the automatic reorientation of faces. The checks can be individually deactivated by setting `reorient_faces=False`, `validate_closed=False` and `validate_connected=False` at initialization of `TriangularMesh` objects.

* Meshing tools such as the [Pyvista](https://docs.pyvista.org/) library can be very convenient for building complex shapes, but often do not guarantee that the mesh is properly closed or connected. Deactivating mesh input checks may lead to unwanted results.

* Triangular meshes from meshing tools often create very large meshes with a lot of faces, especially when working with curved surfaces. Keep in mind that the computation takes of the order of a few microseconds per observer position per face, and that RAM is a limited ressource. On standard computers the checks and field computations might start to fail when meshes tend towards a Million faces.

* There is no self-intersecting check which might lead to unphysical bodies.
```

## Example - Tetrahedron

```{code-cell} ipython3
import magpylib as magpy

# create faceted tetrahedron from vertices and faces
tmesh_tetra = magpy.magnet.TriangularMesh(
    magnetization=(0, 0, 1000),
    vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    faces=((2, 1, 0), (3, 0, 1), (3, 2, 0), (3, 1, 2)),
)

# print input checks attributes
print("is closed:", tmesh_tetra.is_closed)
print("is connected:", tmesh_tetra.is_connected)
print("is reoriented:", tmesh_tetra.is_reoriented)

tmesh_tetra.show()
```

## Example - Dodecahedron from pyvista

`TriangularMesh` magnets can be directly created from Pyvista `PolyData` objects via the classmethod `from_pyvista`.

```{note}
The Pyvista library used in the following examples is not automatically installed with Magpylib. A Pyvista installation guide is found [here](https://docs.pyvista.org/getting-started/installation.html).
```

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=dodec_mesh,
)

dodec.show()
```

We can now add a sensor and plot the B-field value along the sensor path.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import magpylib as magpy

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

# create TriangularMesh object
tmesh_dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(1000, 0, 0),
    polydata=dodec_mesh,
)

# add sensor and rotate around source
sens = magpy.Sensor(position=(2, 0, 0))
sens.rotate_from_angax(angle=np.linspace(0, 320, 50), axis="z", start=0, anchor=0)

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

# plot 3D model and B-field
magpy.show(tmesh_dodec, sens, canvas=ax1)
B = tmesh_dodec.getB(sens)
ax2.plot(B, label=["Bx", "By", "Bz"])

# plot styling
ax1.set(
    title="3D model",
    aspect="equal",
)
ax2.set(
    title="Dodecahedron field",
    xlabel="path index ()",
    ylabel="B-field (mT)",
    aspect=1,
)

plt.gca().grid(color=".9")
plt.gca().legend()
plt.tight_layout()
plt.show()
```

## Example - Pyramid from ConvexHull

`TriangularMesh` objects are easily constructed from the convex hull of a given point cloud using the classmethod `from_ConvexHull`. This classmethod  makes use of the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) class. Note that the Scipy method does not guarantee correct facet orientations if `reorient_faces` is disabled.

```{code-cell} ipython3
import magpylib as magpy

# create Pyramid
points = [[-2,-2, 0], [-2,2,0], [2,-2,0], [2,2,0], [0,0,3]]
tmesh_pyramid = magpy.magnet.TriangularMesh.from_ConvexHull(
    magnetization=(0, 0, 1000),
    points=points,
)

#display graphically
tmesh_pyramid.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_size=1.5,
)
```

## Example - Open Prism

In some cases it may be desirable to generate a `TriangularMesh` object from an open mesh, as described in {ref}`examples-triangle`. In this case the `validate_closed` check must be disabled. 


 By disabling the corresponding `validate_closed` input check, the object initialization will still pass. It is yet possible to highlight the open edges with the `show` function. The same idea holds true for the `validate_connected` check.

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
