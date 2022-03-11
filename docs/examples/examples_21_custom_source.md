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

(examples-custom-source-objects)=
# Custom source objects

The custom source class `CustomSource` allows users to integrate their own custom-objects into the Magpylib interface. The `field_B_lambda` and `field_H_lambda` arguments can be provided with function that are called with `getB` and `getH`.

These custom field functions are treated like core functions. they must accept position inputs (array_like, shape (n,3)) and return the respective field with a similar shape. A fundamental example how to create a custom source object is:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define field function
def custom_field(position):
    """ user defined custom field
    position input: array_like, shape (n,3)
    returns: ndarray, shape (n,3)
    """
    return np.array(position)*2

# custom source
source = magpy.misc.CustomSource(field_B_lambda=custom_field)

# compute field with 2-pixel sensor
sensor = magpy.Sensor(pixel=((1,1,1), (2,2,2)))

B = magpy.getB(source, sensor)
print(B)
```

Custom sources behave like native Magpylib objects. They can make full use of the geometry interface, have style properties, and can be part of collections and field computation together with all other source types. A custom 3D representation for display in `show` can be provided via the `style.model3d` attribute, see {ref}`examples-own-3d-models`.

In the following example we show this functionality by constructing a quadrupole collection from custom source monopoles:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define monopole field
def monopole_field(charge, position):
    """ monopole field"""
    position = np.array(position)
    dist = np.linalg.norm(position, axis=1)
    return charge*(position.T/dist**3).T

# prepare a custom 3D model
trace_pole = magpy.graphics.model3d.make_Ellipsoid(
    backend='matplotlib',
    dimension=(.3,.3,.3),
)

# combine four monopole custom sources into a quadrupole collection
def create_pole(charge):
    """ create a monopole object"""
    field = lambda x: monopole_field( charge, x)
    monopole = magpy.misc.CustomSource(
        field_B_lambda=field,
        style_model3d_showdefault=False,
    )
    monopole.style.model3d.add_trace(trace_pole)
    return monopole

quadrupole = magpy.Collection([create_pole(q) for q in [1,1,-1,-1]])

# move and color the pole objects
pole_pos = np.array([(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)])
for pole, pos, col in zip(quadrupole, pole_pos, 'rrbb'):
    pole.position = pos
    pole.style.color = col

# Matplotlib figure
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d", azim=-80, elev=15,)
ax2 = fig.add_subplot(122,)

# show 3D model in ax1
magpy.show(*quadrupole, canvas=ax1)

# compute B-field on xz-grid and display in ax2
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])
B = quadrupole.getB(grid)

scale = np.linalg.norm(B, axis=2)
cp = ax2.contourf(grid[:,:,0], grid[:,:,2], np.log(scale), levels=100, cmap='rainbow')
ax2.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2], density=2,
    color='k', linewidth=scale**0.3)

# display pole positions in ax2
ax2.plot(pole_pos[:,0], pole_pos[:,2], marker='o', ms=10, mfc='k', mec='w', ls='')

# plot styling
ax1.set(
    title='3D model',
    xlabel='x-position [mm]',
    ylabel='y-position [mm]',
    zlabel='z-position [mm]',
)
ax2.set(
    title='Quadrupole field',
    xlabel='x-position [mm]',
    ylabel='z-position [mm]',
    aspect=1,
)
fig.colorbar(cp, label='[$charge/mm^2$]', ax=ax2)

plt.tight_layout()
plt.show()
```