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

```{versionadded} 4.2
The `Triangle` class
```

+++ {"user_expressions": []}

(examples-triangle)=

# Complex shapes - The Triangle class

The field of a homogeneously charged magnet is, on the outside, the same as the field of a similarly shaped body with a magnetic surface charge. The surface charge is proportional to the projection of the magnetization vector onto the surface normal.

The `Triangle` class is set up so that it can easily be used for this purpose. Arbitrary surfaces are easily approximated by triangles, and given the magnetization vector, the surface charge density is automatically computed. One must be very careful to orient the triangles correctly, with surface normal vectors pointing outwards (right-hand-rule). The resulting H-field of such a collection is correct, but the B-field is only correct on the outside of the body. On the inside the magnetization must be added to the field.

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

+++ {"user_expressions": []}

```{seealso}
Building a source with a set of triangles is error prone, since there is no check if the collection produces a closed body, if the triangles are all pointing outwards and if the manifold is connected or self-intersecting. A more feature rich and robust `TriangularMesh` magnet class allows for combining triangular faces into a single object and implements useful features to build complex-shaped magnets more easily.
See some examples {ref}`examples-triangularmesh`
```
