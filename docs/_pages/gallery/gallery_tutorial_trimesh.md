---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(gallery-tutorial-trimesh)=

# TriangularMesh Class

The `TriangularMesh` class is used to create magnets from triangular surface meshes, instead of assembling them from individual `Triangle` objects as described in {ref}`gallery-shapes-triangle`. This class is initialized with the `vertices` (an array_like of positions) and the `faces` (an array_like of index triplets) inputs. In addition, a set of useful classmethods enables initialization from various inputs:

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

* Meshing tools often create meshes with a lot of faces, especially when working with curved surfaces. Keep in mind that the field computation takes of the order of a few microseconds per observer position per face, and that RAM is a limited resource.
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

In some cases it may be desirable to generate a `TriangularMesh` object from an open mesh, as described in {ref}`gallery-shapes-triangle`. In this case one has to be extremely careful because one cannot rely on the checks. Not to generate warnings or error messages, these checks can be disabled with `"skip"` or their outcome can be ignored with `"ignore"`. The `show` function can be used to view open edges and disconnected parts. In the following example we generate such an open mesh directly from `Triangle` objects.

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

