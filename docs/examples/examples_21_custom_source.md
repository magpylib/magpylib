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

The custom source class `magpylib.misc.CustomSource` allows users to integrate their own custom-objects into the Magpylib interface. The `field_B_lambda` and `field_H_lambda` arguments can be provided with functions that are called with `getB` and `getH`. For this purpose the provided function must accept position inputs (array_like, shape (n,3)) and return the respective field with a similar shape.

In this example a monopole-custom object is created and the field is computed on a grid

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define B-field function for custom source
def monopole_field(obs):
    """ magnetic monopole field
    obs: position, array_like, shape (n,3)
    """
    obs = np.array(obs)
    dist = np.linalg.norm(obs, axis=1)
    return (obs.T/dist**3).T

# custom monopole source
src = magpy.misc.CustomSource(field_B_lambda=monopole_field)

# compute field on grid and display
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])
B = src.getB(grid)

plt.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2], density=2,
    color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='jet')

plt.gca().set(aspect=1)
plt.tight_layout()
plt.show()
```

Custom sources behave like native Magpylib objects. They can make full use of the geometry interface (`position`, `orientation`, `move`, `rotate`) and be part of collections and field computation together with all other source types. A custom 3D representation for display in `show` can be provided via the `style.model3d` attribute, see {ref}`examples-own-3d-models`. It is even possible to make the 3D model dynamic, as is shown in {ref}`examples-own-dynamic-3d-model`.

In the following example we outline this functionality by constructing a quadrupole collection from custom source monopoles:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define B-field function for custom source
def monopole_field(Q, obs):
    """ magnetic monopole field
    obs: position, array_like, shape (n,3)
    Q: charge, number
    """
    obs = np.array(obs)
    dist = np.linalg.norm(obs, axis=1)
    return Q*(obs.T/dist**3).T

# prepare custom source model3d trace
trace_plotly = magpy.display.plotly.make_BaseEllipsoid((.3,.3,.3))
x, y, z, i, j, k = [trace_plotly[k] for k in "xyzijk"]
trace = dict(
    type="plot_trisurf",
    args=(x, y, z),
    triangles=np.array([i, j, k]).T
)

# combine four monopole-Custom Sources into a Quadrupole Collection
field_p = lambda x: monopole_field( 1, x)
field_m = lambda x: monopole_field(-1, x)
pole_pos = np.array([(1,0,0), (0,0,1), (-1,0,0), (0,0,-1)])

coll = magpy.Collection()
for pos, fld, col in zip(pole_pos, [field_p, field_m, field_p, field_m], 'rbrb'):
    src = magpy.misc.CustomSource(
        field_B_lambda=fld,
        position=pos,
        style_model3d_showdefault=False,
        style_color=col,
        )
    src.style.model3d.add_trace(
        backend="matplotlib",
        trace=trace,
        coordsargs={"x": "args[0]", "y": "args[1]", "z": "args[2]"},
    )
    coll.add(src)

# Matplotlib figure
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d", azim=-80, elev=15,)
ax2 = fig.add_subplot(122,)

# show 3D model in ax1
magpy.show(*coll, canvas=ax1)

# compute B-field on xz-grid and display in ax2
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])
B = coll.getB(grid)

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