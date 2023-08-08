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

(gallery-ext-complex-shapes-superposition)=

## Superposition

The [superposition principle](https://en.wikipedia.org/wiki/Superposition_principle) states that the net response caused by two or more stimuli is the sum of the responses caused by each stimulus individually. This principle holds in magnetostatics when there is no material response, and simply states that the total field created by multiple magnets and currents is the sum of the individual fields.

When two magnets overlap geometrically, the magnetization in the overlap region is given by the vector sum of the two individual magnetizations. This enables two geometric operations,

:::::{grid} 2

::::{grid-item-card} Union
:img-bottom: ../../_static/images/docu_field_superpos_union.png
:shadow: None
Build complex forms by aligning base shapes (no overlap) with each other with similar magnetization vector.
::::

::::{grid-item-card} Cut-Out
:img-bottom: ../../_static/images/docu_field_superpos_cutout.png
:shadow: None
When two objects with opposing magnetization vectors of similar amplitude overlap, they will just cancel in the overlap region. This enables geometric cut-out operations.
::::
:::::

Working with union and cut-out operations is demonstrated in the {ref}`gallery-tutorial-field-computation` tutorial.


### Union operation

Based on the superposition principle we can build complex forms by aligning simple base shapes (no overlap), similar to a geometric union. This is demonstrated in the following example, where a hollow cylinder magnet is constructed from cuboids. The field is then compare to the exact solution implemented through `CylinderSegment`.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# create magnet parts with similar magnetization
mag = (500,0,0)
pt1 = magpy.magnet.CylinderSegment(
    magnetization=mag,
    dimension=(0,4,2,90,270),
    position=(0,0,0)
)
pt2 = magpy.magnet.Cuboid(
    magnetization=mag,
    dimension=(2,8,2),
    position=(1,0,0)
)
pt3 = magpy.magnet.TriangularMesh.from_ConvexHull(
    magnetization=mag,
    points=[(2,4,-1),(2,4,1),(2,-4,-1),(2,-4,1),(6,0,1),(6,0,-1)]
)

# combine parts in a Collection (=union)
magnet = magpy.Collection(pt1, pt2, pt3)

# add sensor an plot
sensor = magpy.Sensor(position=np.linspace((7,-10,0), (7,10,0), 100))
with magpy.show_context(magnet, sensor, backend='plotly') as s:
    s.show(col=1)
    s.show(output='B', col=2)
```

Construction of complex forms from base shapes is a powerful tool, however, there is always a geometry approximation error, visible in the above figure. The error can be reduced by increasing the discretization finesse, but this also requires additional computation effort.

### Cut-out operation

When two objects with opposing magnetization vectors of similar amplitude overlap, they will just cancel in the overlap region. This enables geometric cut-out operations. In the following example we construct an exact hollow cylinder solution from two concentric cylinder shapes with opposite magnetizations, and compare the result to the `CylinderSegment` class solution.

```{code-cell} ipython3
from magpylib.magnet import Cylinder, CylinderSegment

# ring from CylinderSegment
ring0 = CylinderSegment(magnetization=(0,0,100), dimension=(2,3,1,0,360))

# ring with cut-out
inner = Cylinder(magnetization=(0,0,-100), dimension=(4,1))
outer = Cylinder(magnetization=(0,0, 100), dimension=(6,1))
ring1 = inner + outer

print('getB from CylinderSegment', ring0.getB((1,2,3)))
print('getB from Cylinder cut-out', ring1.getB((1,2,3)))
```

Note that, it is faster to compute the `Cylinder` field two times than computing the complex `CylinderSegment` field one time. This is why Magpylib automatically falls back to the `Cylinder` solution whenever `CylinderSegment` is called with 360 deg section angles. Unfortunately, cut-out operations cannot be displayed graphically at the moment, but {ref}`examples-own-3d-models` offer a solution here.

Finally, it is explained in {ref}`examples-triangle`, how complex shapes are achieved based on triangular meshes.
