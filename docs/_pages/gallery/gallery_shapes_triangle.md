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

(gallery-shapes-triangle)=

# Triangular Meshes

The magnetic field of a homogenously magnetized body is equivalent to the field of a charged surface. The surface is the hull of the body and the charge density is proportional to the projection of the magnetization vector onto the surface normal.

It is very common to approximate the surface of bodies by triangular meshes, which can then be transformed into magnets using the `Triangle` and the `TriangularMesh` classes. When using these classes one should abide by the following rules:

1. The surface must be closed, or, all missing faces must have zero charge (magnetization vector perpendicular to surface normal).
2. All triangles are oriented outwards (right-hand-rule)
3. The surface must not be self-intersecting.
4. For the B-field the magnetic polarization must be added on the inside of the body.

## Cubeoctahedron Magnet

In this example `Triangle` is used to create a magnet with cuboctahedral shape. Notice that triangle orientation is displayed by default for convenience.

```{code-cell} ipython3
import magpylib as magpy

# Create collection of triangles
triangles = [
    ([ 0, 1,-1], [-1, 1, 0], [ 1, 1, 0]),
    ([ 0, 1, 1], [ 1, 1, 0], [-1, 1, 0]),
    ([ 0, 1, 1], [-1, 0, 1], [ 0,-1, 1]),
    ([ 0, 1, 1], [ 0,-1, 1], [ 1, 0, 1]),
    ([ 0, 1,-1], [ 1, 0,-1], [ 0,-1,-1]),
    ([ 0, 1,-1], [ 0,-1,-1], [-1, 0,-1]),
    ([ 0,-1, 1], [-1,-1, 0], [ 1,-1, 0]),
    ([ 0,-1,-1], [ 1,-1, 0], [-1,-1, 0]),
    ([-1, 1, 0], [-1, 0,-1], [-1, 0, 1]),
    ([-1,-1, 0], [-1, 0, 1], [-1, 0,-1]),
    ([ 1, 1, 0], [ 1, 0, 1], [ 1, 0,-1]),
    ([ 1,-1, 0], [ 1, 0,-1], [ 1, 0, 1]),
    ([ 0, 1, 1], [-1, 1, 0], [-1, 0, 1]),
    ([ 0, 1, 1], [ 1, 0, 1], [ 1, 1, 0]),
    ([ 0, 1,-1], [-1, 0,-1], [-1, 1, 0]),
    ([ 0, 1,-1], [ 1, 1, 0], [ 1, 0,-1]),
    ([ 0,-1,-1], [-1,-1, 0], [-1, 0,-1]),
    ([ 0,-1,-1], [ 1, 0,-1], [ 1,-1, 0]),
    ([ 0,-1, 1], [-1, 0, 1], [-1,-1, 0]),
    ([ 0,-1, 1], [ 1,-1, 0], [ 1, 0, 1]),
]
cuboc = magpy.Collection()
for t in triangles:
    cuboc.add(
        magpy.misc.Triangle(
            magnetization=(100, 200, 300),
            vertices=t,
        )
    )

# Display collection of triangles
magpy.show(
    cuboc,
    backend='pyvista',
    style_magnetization_mode='arrow',
    style_orientation_color='yellow',
    jupyter_backend="panel", # better pyvista rendering in a jupyter notebook
)
```

## Triangular Prism Magnet

Consider a prism with triangular base that is magnetized orthogonal to the base. All surface normals of the sides of the prism are orthogonal to the magnetization vector. As a result the sides do not contribute to the magnetic field because their charge density disappears. Only top and bottom surfaces contribute. One must be very careful when defining those surfaces in such a way that the surface normals point outwards.

Leaving out parts of the surface that do not contribute to the field is beneficial for the computation speed.

```{code-cell} ipython3
import magpylib as magpy

# Create prism magnet as triangle collection
top = magpy.misc.Triangle(
    magnetization=(0,0,1000),
    vertices=((-1,-1,1), (1,-1,1), (0,2,1)),
    style_label="top"
)
bott = magpy.misc.Triangle(
    magnetization=(0,0,1000),
    vertices=((-1,-1,-1), (0,2,-1), (1,-1,-1)),
    style_label="bottom"
)
prism = magpy.Collection(top, bott)

# Display graphically
magpy.show(
    *prism,
    backend='plotly',
    style_opacity=0.5,
    style_magnetization_show=False
)
```

## TriangularMesh class

While `Triangle` simply provides the field of a charged triangle and can be used to contruct complex forms, it is prone to error and tedious to work with when meshes become large. For this purpose the `TriangularMesh` class ensures proper and convenient magnet creation by automatically checking mesh integrity and by orienting the faces at initialization.

```{attention}
Automatic face reorientation of `TriangularMesh` may fail when the mesh is open.
```

In this example we revisit the cubeoctahedron, but generate it through the `TriangularMesh` class.

```{code-cell} ipython3
import magpylib as magpy

# Create cubeoctahedron magnet 
vertices = [
    ( 0, 1,-1), (-1, 1, 0), ( 1, 1, 0),
    ( 0, 1, 1), (-1, 0, 1), ( 0,-1, 1),
    ( 1, 0, 1), ( 1, 0,-1), ( 0,-1,-1),
    (-1, 0,-1), (-1,-1, 0), ( 1,-1, 0),
]
faces = [
    (0,1,2),   (3,2,1),   (3,4,5),   (3,5,6),
    (0,7,8),   (0,8,9), (5,10,11), (8,11,10),
    (1,9,4),  (10,4,9),   (2,6,7),  (11,7,6),
    (3,1,4),   (3,6,2),   (0,9,1),   (0,2,7),
    (8,10,9), (8,7,11),  (5,4,10),  (5,11,6),
]
cuboc = magpy.magnet.TriangularMesh(
    magnetization=(100, 200, 300),
    vertices=vertices,
    faces=faces
)

# Display TriangularMesh body
magpy.show(
    cuboc,
    backend='plotly',
    style_mesh_grid_show=True,
    style_mesh_grid_line_width=4
)
```

The `TriangularMesh` class is extremely powerful as it enables almost arbitrary magnet shapes. It is described in detail in {ref}`docu-magpylib-api-trimesh`. There are many ways to generate such triangular meshes. An example thereof is shown in {ref}`gallery-shapes-pyvista`.

```{caution}
* `getB` and `getH` compute the fields correctly only if the mesh is closed, not self-intersecting, and all faces are properly oriented outwards.

* Input checks and face reorientation can be computationally expensive. The checks can individually be deactivated by setting `reorient_faces="skip"`, `check_open="skip"`, `check_disconnected="skip"`, and `check_selfintersecting="skip"` at initialization of `TriangularMesh` objects. The checks can also be performed by hand after initialization.

* Meshing tools such as the [Pyvista](https://docs.pyvista.org/) library can be very convenient for building complex shapes, but often do not guarantee that the mesh is properly closed or connected - see {ref}`gallery-shapes-pyvista`.

* Meshing tools often create meshes with a lot of faces, especially when working with curved surfaces. Keep in mind that the field computation takes of the order of a few microseconds per observer position per face, and that RAM is a limited resource.
```

## Open TriangularMesh

In some cases it may be desirable to generate a `TriangularMesh` object from an open mesh (see Prism example above). In this case one has to be extremely careful because one cannot rely on the checks. Not to generate warnings or error messages, these checks can be disabled with `"skip"` or their outcome can be ignored with `"ignore"`. The `show` function can be used to view open edges and disconnected parts. In the following example we generate such an open mesh directly from `Triangle` objects.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

# Create top and bottom faces of a prism magnet
top = magpy.misc.Triangle(
    magnetization=(1000,0,0),
    vertices= ((-1,-1,1), (1,-1,1), (0,2,1)),
)
bottom = magpy.misc.Triangle(
    magnetization=(1000,0,0),
    vertices= ((-1,-1,-1), (0,2,-1), (1,-1,-1)),
)

# Create prism with open edges
prism = magpy.magnet.TriangularMesh.from_triangles(
    magnetization=(0, 0, 1000),   # overrides triangles magnetization
    triangles=[top, bottom],
    check_open="ignore",          # check but ignore open mesh
    check_disconnected="ignore",  # check but ignore disconnected mesh
    reorient_faces="ignore",      # check but ignore non-orientable mesh
)
prism.style.label = "Open Prism",
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

```{caution}
Keep in mind that the inside-outside check will fail, so that `getB` may yield wrong results on the inside of the prism where the polarization vector should be added.
```