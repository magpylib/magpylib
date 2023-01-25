---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(examples-triangle)=

# Complex shapes - The Triangle class

The field of a homogeneously charged magnet is, on the outside, the same as the field of a similarly shaped body with a magnetic surface charge. The surface charge is proportional to the projection of the magnetization vector onto the surface normal. The `Triangle` class is set up so that it can easily be used to approximate surfaces, and given the magnetization vector, the charge density is automatically computed. The resulting H-field is correct, but the B-field is only correct on the outside of the body, because on the inside the magnetization must be added to it.

## Example: Triangular prisma magnet

Consider a Prisma with triangular base that is magnetized orthogonal to the base. All surface normals of the sides of the prisma are orthogonal to the magnetization vector. As a result the sides do not contribute to the magnetic field because their charge desity dissappears. Only top and bottom surfaces contribute. One must be very careful when defining those surfaces in such a way that the surface normals point outwards. The following examples shows how the `Triangle` class can be used to create such a prisma.

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

prisma = magpy.Collection(top, bott)

prisma.show(
    style_opacity=0.5,
    style_magnetization_size=0.2,
)
```

+++ {"tags": []}

## Example: Cuboctahedron magnet

More complex bodies are easy constructed from Triangles. The following code shows how a magnet with cuboc shape can be constructed. Be aware that the B-field is only correcto on the outside.

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
    style_orientation=dict(size=3, symbol="arrow3d", color="magenta"),
    style_magnetization_mode="arrow",
    jupyter_backend="panel", # better pyvista rendering in a jupyter notebook
)
```

```{code-cell} ipython3

```
