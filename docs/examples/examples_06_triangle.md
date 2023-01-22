---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(examples-triangle)=

# Complex shapes - The Triangle class

The field of a homogeneously charged magnet is, on the outside, the same as the field of a similarly shaped body with a magnetic surface charge. The surface charge is proportional to the projection of the magnetization vector onto the surface normal. The `Triangle` class is set up so that it can easily be used to approximate surfaces, and given the magnetization vector, the charge density is automatically computed. The resulting H-field is correct, but the B-field is only correct on the outside of the body, because on the inside the magnetization must be added to it.

## Example: Triangular Prisma

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

prisma.show()
```

## Example: Cuboctahedron Magnet

More complex bodies are easy constructed from Triangles. The following code shows how magnet with cuboctahedron shape can be constructed.

```{code-cell} ipython3
import magpylib as magpy

vertices = (((0.0,1.0,-1.0),(1.0,1.0,0.0),(-1.0,1.0,0.0)),
    ((0.0,1.0,1.0),(-1.0,1.0,0.0),(1.0,1.0,0.0)),
    ((0.0,1.0,1.0),(0.0,-1.0,1.0),(-1.0,0.0,1.0)),
    ((0.0,1.0,1.0),(1.0,0.0,1.0),(0.0,-1.0,1.0)),
    ((0.0,1.0,-1.0),(0.0,-1.0,-1.0),(1.0,0.0,-1.0)),
    ((0.0,1.0,-1.0),(-1.0,0.0,-1.0),(0.0,-1.0,-1.0)),
    ((0.0,-1.0,1.0),(1.0,-1.0,0.0),(-1.0,-1.0,0.0)),
    ((0.0,-1.0,-1.0),(-1.0,-1.0,0.0),(1.0,-1.0,0.0)),
    ((-1.0,1.0,0.0),(-1.0,0.0,1.0),(-1.0,0.0,-1.0)),
    ((-1.0,-1.0,0.0),(-1.0,0.0,-1.0),(-1.0,0.0,1.0)),
    ((1.0,1.0,0.0),(1.0,0.0,-1.0),(1.0,0.0,1.0)),
    ((1.0,-1.0,0.0),(1.0,0.0,1.0),(1.0,0.0,-1.0)),
    ((0.0,1.0,1.0),(-1.0,0.0,1.0),(-1.0,1.0,0.0)),
    ((0.0,1.0,1.0),(1.0,1.0,0.0),(1.0,0.0,1.0)),
    ((0.0,1.0,-1.0),(-1.0,1.0,0.0),(-1.0,0.0,-1.0)),
    ((0.0,1.0,-1.0),(1.0,0.0,-1.0),(1.0,1.0,0.0)),
    ((0.0,-1.0,-1.0),(-1.0,0.0,-1.0),(-1.0,-1.0,0.0)),
    ((0.0,-1.0,-1.0),(1.0,-1.0,0.0),(1.0,0.0,-1.0)),
    ((0.0,-1.0,1.0),(-1.0,-1.0,0.0),(-1.0,0.0,1.0)),
    ((0.0,-1.0,1.0),(1.0,0.0,1.0),(1.0,-1.0,0.0)),
)
ica = magpy.Collection()
for vert in vertices:
    ica.add(magpy.misc.Triangle(
        magnetization=(100,200,300),
        vertices=vert,
    ))

magpy.show(*ica)
```


<!-- ## Pyvista mesh and Facet class # we leave this for the TriMesh class :)

Contemporary tools like [Pyvista](https://docs.pyvista.org/) offer powerful meshing options. The following example shows how a complex Pyvista object can be used together with the `Facet` class to create a magnet object with little effort. However, contrary to a an actual magnet object there is no insode-outside check when transofming between B- and H- field.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# create a complex pyvista PolyData object
sphere = pv.Sphere(radius=0.85)
dodec = pv.Dodecahedron().triangulate().subdivide(5)
object = dodec.boolean_difference(sphere)

# extract triangles and create Facet vertices input
points = object.points
faces = object.faces.reshape(-1,4)[:,1:]
verts = [[points[f[i]] for i in range(3)] for f in faces]

magnet = magpy.misc.Facet(
    magnetization=(0,0,100),
    vertices=verts
)

magnet.show(backend='plotly')
``` -->


