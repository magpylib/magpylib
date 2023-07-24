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

(gallery-ext-complex-shapes-triangle)=

# Complex shapes with Triangle

## Example: Triangular prism magnet

Consider a prism with triangular base that is magnetized orthogonal to the base. All surface normals of the sides of the prism are orthogonal to the magnetization vector. As a result the sides do not contribute to the magnetic field because their charge density disappears. Only top and bottom surfaces contribute. One must be very careful when defining those surfaces in such a way that the surface normals point outwards. For this purpose the surface normal of `Triangle` objects is graphically displayed by default. The following examples shows how the `Triangle` class can be used to create a prism magnet object.

```{code-cell} ipython3
import magpylib as magpy

top = magpy.misc.Triangle(
    magnetization=(0,0,1000),
    vertices= ((-1,-1,1), (1,-1,1), (0,2,1)),
)
bott = magpy.misc.Triangle(
    magnetization=(0,0,1000),
    vertices= ((-1,-1,-1), (0,2,-1), (1,-1,-1)),
)

prism = magpy.Collection(top, bott)

prism.show(
    style_opacity=0.5,
    style_magnetization_size=0.2,
)
```

## Example: Cuboctahedron magnet

More complex bodies are easy constructed from `Triangle` objects. The following code shows how a magnet with cuboctahedral shape can be constructed. Be aware that the B-field requires addition of the magnetization vector on the inside.

```{code-cell} ipython3
import magpylib as magpy

vertices = [
    ([0, 1, -1], [-1, 1, 0], [1, 1, 0]),
    ([0, 1, 1], [1, 1, 0], [-1, 1, 0]),
    ([0, 1, 1], [-1, 0, 1], [0, -1, 1]),
    ([0, 1, 1], [0, -1, 1], [1, 0, 1]),
    ([0, 1, -1], [1, 0, -1], [0, -1, -1]),
    ([0, 1, -1], [0, -1, -1], [-1, 0, -1]),
    ([0, -1, 1], [-1, -1, 0], [1, -1, 0]),
    ([0, -1, -1], [1, -1, 0], [-1, -1, 0]),
    ([-1, 1, 0], [-1, 0, -1], [-1, 0, 1]),
    ([-1, -1, 0], [-1, 0, 1], [-1, 0, -1]),
    ([1, 1, 0], [1, 0, 1], [1, 0, -1]),
    ([1, -1, 0], [1, 0, -1], [1, 0, 1]),
    ([0, 1, 1], [-1, 1, 0], [-1, 0, 1]),
    ([0, 1, 1], [1, 0, 1], [1, 1, 0]),
    ([0, 1, -1], [-1, 0, -1], [-1, 1, 0]),
    ([0, 1, -1], [1, 1, 0], [1, 0, -1]),
    ([0, -1, -1], [-1, -1, 0], [-1, 0, -1]),
    ([0, -1, -1], [1, 0, -1], [1, -1, 0]),
    ([0, -1, 1], [-1, 0, 1], [-1, -1, 0]),
    ([0, -1, 1], [1, -1, 0], [1, 0, 1]),
]

cuboc = magpy.Collection(style_label="Cuboctahedron")
for ind, vert in enumerate(vertices):
    cuboc.add(
        magpy.misc.Triangle(
            magnetization=(100, 200, 300),
            vertices=vert,
            style_label=f"Triangle_{ind+1:02d}",
        )
    )

magpy.show(
    *cuboc,
    backend="pyvista",
    style_orientation_size=2,
    style_orientation_color='yellow',
    style_orientation_symbol='cone',
    style_magnetization_mode="arrow",
    jupyter_backend="panel", # better pyvista rendering in a jupyter notebook
)
```

However, despite the potential of the `Triangle` class to build complex shapes, its application is prone to error because there is no intrinsic check of face orientation, connectedness, and being inside or outside of the magnet. For convenience the `TriangularMesh` class provides all these features, and enables users to quickly import complex triangular meshes as single magnet objects, see {ref}`examples-triangularmesh`.
