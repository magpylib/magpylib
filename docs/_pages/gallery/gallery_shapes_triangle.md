---
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

## Triangular Mesh Magnet

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