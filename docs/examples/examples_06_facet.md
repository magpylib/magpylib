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

(examples-facet)=

# Complex shapes from meshes

## Application of the Facet class

When a `Facet` object has a closed surface, where all individual facets are oriented outwards, their combined field corresponds to the one of a homogeneously charged magnet. Therefore, the `Facet` class can be used to approximate any form using a mesh of triangular facets. The following example shows an Icosahedron constructed from multiple facets. However, it must be understood that with such Facet objects an inside-outside check is not possible for observer points, so that the inside B-field is missing the magnetization.

```{code-cell} ipython3
import magpylib as magpy

verts = (((0.0,1.0,-1.0),(1.0,1.0,0.0),(-1.0,1.0,0.0)),
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
ica = magpy.misc.Facet(
    magnetization=(0,0,100),
    vertices=verts
)
ica.show()
```

## Pyvista mesh and Facet class

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
```


