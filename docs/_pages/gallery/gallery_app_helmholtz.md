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

(gallery-app-helmholtz)=

# Helmholtz Coils

- coil modeling
- visualization of homogeneity



<!--
In this example we model the **magnetic field of a coil**, and show how to display it with spectacular **field line** representations.

## Coil models

**Model 1:** The coil consists of multiple windings, each of which can be modeled with a circular current loop which is realized by the `Circle` class. The individual windings are combined into a `Collection` which itself behaves like a single magnetic field source.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

coil1 = magpy.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpy.current.Circle(
        current=100,
        diameter=10,
        position=(0,0,z),
    )
    coil1.add(winding)

coil1.show()
```

**Model 2:** The coil is in reality more like a spiral, which can be modeled using the `Polyline` class. However, a good spiral approximation requires many small line segments, which makes the computation slower.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

ts = np.linspace(-8, 8, 1000)
vertices = np.c_[5*np.cos(ts*2*np.pi), 5*np.sin(ts*2*np.pi), ts]
coil2 = magpy.current.Polyline(
    current=100,
    vertices=vertices
)

coil2.show()
```

## Matplotlib streamplot

Streamplot from Matplotlib is a powerful tool to outline the field lines. However, it must be understood that streamplot shows only a projection of the field onto the observation plane. All field components that point out of the plane become invisible. In out example we choose symmetry planes, where the perpendicular component is negligible.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(13,5))

# create grid
ts = np.linspace(-20, 20, 20)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute and plot field of coil1
B = magpy.getB(coil1, grid)
Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

sp = ax1.streamplot(
    grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2,
    color=Bamp,
    linewidth=np.sqrt(Bamp)*3,
    cmap='coolwarm',
)

# compute and plot field of coil2
B = magpy.getB(coil2, grid)
Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

cp = ax2.contourf(
    grid[:,:,0], grid[:,:,2], Bamp,
    levels=100,
    cmap='coolwarm',
)
ax2.streamplot(
    grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2,
    color='black',
)

# figure styling
ax1.set(
    title='Magnetic field of coil1',
    xlabel='x-position (mm)',
    ylabel='z-position (mm)',
    aspect=1,
)
ax2.set(
    title='Magnetic field of coil2',
    xlabel='x-position (mm)',
    ylabel='z-position (mm)',
    aspect=1,
)

plt.colorbar(sp.lines, ax=ax1, label='(mT)')
plt.colorbar(cp, ax=ax2, label='(mT)')

plt.tight_layout()
plt.show()
```

## Pyvista streamlines

[Pyvista](https://docs.pyvista.org/) is an incredible VTK based tool for 3D plotting and mesh analysis.

The following example shows how to compute and display 3D field lines of `coil1` with Pyvista. To run this example, the user must install Pyvista (`pip install pyvista`). By removing the command `jupyter_backend='static'` in `show`, the 3D figure becomes interactive.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
import pyvista as pv

pv.set_jupyter_backend('panel') # improve rending in a jupyter notebook

coil1 = magpy.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpy.current.Circle(
        current=100,
        diameter=10,
        position=(0,0,z),
    )
    coil1.add(winding)

grid = pv.UniformGrid(
    dimensions=(41, 41, 41),
    spacing=(2, 2, 2),
    origin=(-40, -40, -40),
)

# compute B-field and add as data to grid
grid['B'] = coil1.getB(grid.points)

# compute field lines
seed = pv.Disc(inner=1, outer=5.2, r_res=3, c_res=12)
strl = grid.streamlines_from_source(
    seed,
    vectors='B',
    max_time=180,
    initial_step_length=0.01,
    integration_direction='both',
)

# create plotting scene
pl = pv.Plotter()

# add field lines and legend to scene
legend_args = {
    'title': 'B (mT)',
    'title_font_size': 20,
    'color': 'black',
    'position_y': 0.25,
    'vertical': True,
}

# draw coils
magpy.show(coil1, canvas=pl, backend='pyvista')

# add streamlines
pl.add_mesh(
    strl.tube(radius=.2),
    cmap="bwr",
    scalar_bar_args=legend_args,
)
# display scene
pl.camera.position=(160, 10, -10)
pl.set_background("white")
pl.show()
``` -->
