---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-complex-forms)=

# Complex forms

## Superposition

The [**superposition principle**](https://en.wikipedia.org/wiki/Superposition_principle) states that the net response caused by two or more stimuli is the sum of the responses that would have been caused by each stimulus individually. This principle holds in Magnetostatics when there is no material response, and simply means that the total field created by two magnets is the (vector) sum of the two individual fields.

We demonstrate this by showing that the field of one large magnet `src0` is the same as the sum of the fields of the two pieces `src1` and `src2` with the total same shape:

```{code-cell} ipython3
import magpylib as magpy

# the field of one large magnet
src0 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(1,1,4))
print(magpy.getB(src0, (1,2,3)))

# the combined field of two pieces
src1 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(1,1,2), position=(0,0,1))
src2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(1,1,2), position=(0,0,-1))
print(magpy.getB([src1, src2], (1,2,3), sumup=True))
```

## Complex forms from base shapes

The above example shows that we can build (complex) forms from base shapes. In the following example we construct a hollow cylinder magnet from cuboid base shapes and compare it to the exact solution implemented through `CylinderSegment`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define Matplotlib figure
fig = plt.figure(figsize=(14,5))
ax1 = fig.add_subplot(131, projection='3d', elev=24)
ax2 = fig.add_subplot(132, projection='3d', elev=24)
ax3 = fig.add_subplot(133)

# magnetization vector
mag = (0,0,100)

# sensor
sens = magpy.Sensor(position=np.linspace((-4,0,3), (4,0,3), 50))

# construct and display hollow cylinder from cuboid shapes
ts = np.linspace(-3, 3, 31)
grid = [(x,y,0) for x in ts for y in ts]

col = magpy.Collection()
for pos in grid:
    r = np.sqrt(pos[0]**2 + pos[1]**2)
    if 2<r<3:
        col.add(magpy.magnet.Cuboid(mag, [.2,.2,1], position=pos))
magpy.show(col, sens, canvas=ax1, style_magnetization_show=False)

# display hollow Cylinder from CylinderSegment class
src = magpy.magnet.CylinderSegment(mag, [2,3,1,0,360])
magpy.show(src, sens, canvas=ax2, style_magnetization_show=False)

# compute and display field at sensor
ax3.plot(sens.getB(col), label='from Cuboid')
ax3.plot(sens.getB(src), ls='--', label='from CylinderSegment')
ax3.grid(color='.9')
ax3.legend()

plt.tight_layout()
plt.show()
```

Construction of complex forms from base shapes is a powerful tool, however there is always a geometry approximation error, visible in the above plots. The error can be reduced by increasing the mesh finesse, which, requires additional computation effort.

## Cut-out operations

Finally, it is critical to understand that the superposition principle holds for the magnetization itself. When two magnets overlap geometrically, the magnetization in the overlap region is given by the vector sum of the two individual magnetizations. Specifically, if those magnetizations are of similar amplitude but point in opposite directions, they will just cancel which enables **geometric cut out** operations.

In the following example we construct an exact hollow cylinder solution from two cocentric cylinder shapes with opposite magnetizations, and compare the result to the `CylinderSegment` class solution.

```{code-cell} ipython3
import magpylib as magpy

# the exact field of a hollow cylinder from the CylinderSegment class
src0 = magpy.magnet.CylinderSegment(mag, [2,3,1,0,360])
print(magpy.getB(src0, (1,2,3)))

# cutting out one Cylinder from another Cylinder
src1 = magpy.magnet.Cylinder(magnetization=(0,0, 100), dimension=(6,1))
src2 = magpy.magnet.Cylinder(magnetization=(0,0,-100), dimension=(4,1))
print(magpy.getB([src1, src2], (1,2,3), sumup=True))
```

Note that, computing the `Cylinder` field two times is faster than computing the complex `CylinderSegment` field one time.
