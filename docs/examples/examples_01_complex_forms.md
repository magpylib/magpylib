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

(examples-complex-forms)=

# Complex forms

The [**superposition principle**](https://en.wikipedia.org/wiki/Superposition_principle) states that the net response caused by two or more stimuli is the sum of the responses caused by each stimulus individually. This principle holds in Magneto statics when there is no material response, and simply means that the total field created by multiple magnets and currents is the sum of the individual fields.

It is critical to understand that the superposition principle holds for the magnetization itself. When two magnets overlap geometrically, the magnetization in the overlap region is given by the vector sum of the two individual magnetizations.

(examples-union-operation)=

## Union operation

Based on the superposition principle we can build complex forms by aligning simple base shapes (no overlap), similar to a geometric union. This is demonstrated in the following example, where a hollow cylinder magnet is constructed from cuboids. The field is then compare to the exact solution implemented through `CylinderSegment`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cuboid, CylinderSegment

fig = plt.figure(figsize=(14,5))
ax1 = fig.add_subplot(131, projection='3d', elev=24)
ax2 = fig.add_subplot(132, projection='3d', elev=24)
ax3 = fig.add_subplot(133)

sensor = magpy.Sensor(position=np.linspace((-4,0,3), (4,0,3), 50))

# ring with cuboid shapes
ts = np.linspace(-3, 3, 31)
grid = [(x,y,0) for x in ts for y in ts]

coll = magpy.Collection()
for pos in grid:
    r = np.sqrt(pos[0]**2 + pos[1]**2)
    if 2<r<3:
        coll.add(Cuboid(magnetization=(0,0,100), dimension=(.2,.2,1), position=pos))
magpy.show(coll, sensor, canvas=ax1, style_magnetization_show=False)

# ring with CylinderSegment
ring = CylinderSegment(magnetization=(0,0,100), dimension=(2,3,1,0,360))
magpy.show(ring, sensor, canvas=ax2, style_magnetization_show=False)

# compare field at sensor
ax3.plot(sensor.getB(coll).T[2], label='Bz from Cuboids')
ax3.plot(sensor.getB(ring).T[2], ls='--', label='Bz from CylinderSegment')
ax3.grid(color='.9')
ax3.legend()

plt.tight_layout()
plt.show()
```

Construction of complex forms from base shapes is a powerful tool, however, there is always a geometry approximation error, visible in the above figure. The error can be reduced by increasing the discretization finesse, but this also requires additional computation effort.

## Cut-out operation

When two objects with opposing magnetization vectors of similar amplitude overlap, they will just cancel in the overlap region. This enables geometric cut-out operations. In the following example we construct an exact hollow cylinder solution from two concentric cylinder shapes with opposite magnetizations, and compare the result to the `CylinderSegment` class solution.

```{code-cell} ipython3
from magpylib.magnet import Cylinder, CylinderSegment

# ring from CylinderSegment
ring0 = CylinderSegment(magnetization=(0,0,100), dimension=(2,3,1,0,360))

# ring with cut-out
inner = Cylinder(magnetization=(0,0,-100), dimension=(4,1))
outer = Cylinder(magnetization=(0,0, 100), dimension=(6,1))
ring1 = inner + outer

print('getB from Cylindersegment', ring0.getB((1,2,3)))
print('getB from Cylinder cut-out', ring1.getB((1,2,3)))
```

Note that, it is faster to compute the `Cylinder` field two times than computing the complex `CylinderSegment` field one time. This is why Magpylib automatically falls back to the `Cylinder` solution whenever `CylinderSegment` is called with 360 deg section angles. Unfortunately, cut-out operations cannot be displayed graphically at the moment, but {ref}`examples-own-3d-models` offer a solution here.

## Complex shapes with the Facet class

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


