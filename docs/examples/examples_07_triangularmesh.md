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

The `TriangularMesh` class is used to create magnets from triangular surface meshes, instead of assembling them from individual `Triangle` objects as described in {ref}`examples-triangle`. This class is initialized with the `vertices` (an array_like of positions) and the `faces` (an array_like of index triplets) inputs. In addition, a set of useful classmethods enables initialization from various inputs:

- `TriangularMesh.from_mesh()`: from an array_like of triplets of vertices
- `TriangularMesh.from_triangles()`: from a list of `Triangle` objects
- `TriangularMesh.from_ConvexHull()`: from the convex hull of a given point cloud
- `TriangularMesh.from_pyvista()`: from a Pvista `PolyData` object

In contrast to a `Collection` of `Triangle` objects the `TriangularMesh` class performs several important checks at initialization by default to ensure that the given triangular mesh forms a proper magnet:

- `check_closed`: checks if given mesh forms a closed surface
- `check_connected`: checks if given mesh is connected
- `reorient_faces`: checks if faces are oriented outwards, and flips the ones wrongly oriented. This works only if the mesh is closed.

All three checks will throw warnings by default if the mesh is open, disconnected, or cannot be reoriented. Four options enable error handling: `"skip"`, `"ignore"`, `"warn"` (default), `"raise"`. If skipped at initialization, the checks can be performed by hand via respective methods.

The mesh status is set by the checks, and can be viewed via the properties `status_closed`, `status_connected` and `status_reoriented` with possible values `None`, `True`, `False`. Problems of the mesh (e.g. open edges) are stored in `status_closed_data` and `status_connected_data`. Such problems can be viewed with `show`.

```{caution}
* Only if the mesh is closed and all faces are properly oriented outwards, `getB` and `getH` compute the fields correctly.

* Input checks and face reorientation can be computationally expensive. The checks can be individually deactivated by setting `reorient_faces="skip"`, `check_closed="skip"` and `check_connected="skip"` at initialization of `TriangularMesh` objects. The checks can also be performed by hand after initialization.

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
print("mesh status closed:", tmesh_tetra.status_closed)
print("mesh status connected:", tmesh_tetra.status_connected)
print("mesh status reoriented:", tmesh_tetra.status_reoriented)

tmesh_tetra.show()
```

## Example - Dodecahedron magnet from pyvista

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
)

plt.gca().grid(color=".9")
plt.gca().legend()
plt.tight_layout()
plt.show()
```

## Example - Pyramid magnet from ConvexHull

`TriangularMesh` objects are easily constructed from the convex hull of a given point cloud using the classmethod `from_ConvexHull`. This classmethod  makes use of the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) class. Note that the Scipy method does not guarantee correct face orientations if `reorient_faces` is disabled.

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

## Example - Prism magnet with open mesh

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
    check_closed="ignore",        # check but ignore open mesh
    check_connected="ignore",     # check but ignore disconnected mesh
    reorient_faces="ignore",      # check but ignore non-orientable mesh
    style_label="Open prism",
)
prism.style.magnetization.mode = "arrow"

print("mesh status closed:", prism.status_closed)
print("mesh status connected:", prism.status_connected)
print("mesh status reoriented:", prism.status_reoriented)

prism.show(
    backend="plotly",
    style_mesh_open_show=True,
    style_mesh_disjoint_show=True,
)
```

## Example - Boolean operations with Pyvista

It is noteworthy that with [Pyvista](https://docs.pyvista.org/) it is possible to build complex shapes with boolean geometric operations. However, such operations often result in open and disconnected meshes that require some refinement to produce solid magnets. The following example demonstrates the problem, and how to view it with `show`.

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.6)
cube = pv.Cube().triangulate()
obj = cube.boolean_difference(sphere)

# use the `from_pyvista` classmethod to construct our magnet
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    check_connected="ignore",
    check_closed="ignore",
    reorient_faces="ignore",
    style_label="magnet",
)

print(f'mesh status closed: {magnet.status_closed}')
print(f'mesh status connected: {magnet.status_connected}')
print(f'mesh status reoriented: {magnet.status_reoriented}')

magnet.show(
    backend="plotly",
    style_mesh_open_show=True,
    style_mesh_disjoint_show=True,
)
```

Such problems can typically be avoided by
1. Subdivision of the initial triangulation (give Pyvista a finer mesh to work with from the start)
2. Cleaning (merge duplicate points, remove unused points, remove degenerate faces)

The following code solves these problems and produces a clean magnet.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.6)
dodec = pv.Cube().triangulate().subdivide(2)
obj = dodec.boolean_difference(sphere)
obj = obj.clean()

# use the `from_pyvista` classmethod to construct our magnet
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    style_label="magnet",
)

print(f'mesh status closed: {magnet.status_closed}')
print(f'mesh status connected: {magnet.status_connected}')
print(f'mesh status reoriented: {magnet.status_reoriented}')

magnet.show(backend="plotly")
```