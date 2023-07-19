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

(gallery-ext-custom-quadrupole)=

# Customization

## Customization

**User-defined 3D models** (traces) for any object that will be displayed by `show`, can be stored in `style.model3d.data`. A trace itself is a dictionary that contains all information necessary for plotting, and can be added with the method `style.model3d.data.add_trace`. In the example gallery {ref}`examples-3d-models` it is explained how to create custom traces with standard plotting backends such as `scatter3d` or `mesh3d` in Plotly, or `plot`, `plot_surface` and `plot_trisurf` in Matplotlib. Some pre-defined models are also provided for easy parts visualization.

**User-defined source objects** are easily realized with the `CustomSource` class. Such a custom source object is provided with a user-defined field computation function, that is stored in the attribute `field_func` and is used when `getB` and `getH` are called. Similar to core functions, `field_func` must have the two positional arguments `field` (can be `'B'` or `'H'`) and `observers` (must accept ndarrays of shape (n,3)), and return the respective fields in units of mT and kA/m in the same shape. Details on working with custom sources are given in {ref}`examples-custom-source-objects`.

While each of these features can be used individually, the combination of the two (own source class with own 3D representation) enables a high level of customization in Magpylib. Such user-defined objects will feel like native Magpylib objects and can be used in combination with all other features, which is demonstrated in the following example:

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# define B/H field function for custom source
def easter_field(field, observers):
    """ points in z-direction and decays with 1/r^3"""
    dist = np.linalg.norm(observers, axis=1)
    dist+=1e-16 # avoid zero division
    return np.c_[np.zeros((len(observers),2)), 1/dist**3]

# create custom source
egg = magpy.misc.CustomSource(
    field_func=easter_field,
    style=dict(color='orange', model3d_showdefault=False, label='The Egg'),
)

# add a custom 3D model
trace = magpy.graphics.model3d.make_Ellipsoid(
    dimension=(1,1,1.4),
)
egg.style.model3d.add_trace(trace)

# move the egg
ts = np.linspace(-2*np.pi, 2*np.pi, 70)
ts = ts + 0.75*np.sin(ts-np.pi/8)**2
egg.position = [(t/3, 0, -0.2*np.sin(t)**2) for t in ts]
egg.rotate_from_euler(ts, 'y', start=0, degrees=False)

# add sensor and compute field on path
sensor = magpy.Sensor(position=(0,0,1.5), style_size=2)
B = sensor.getB(egg)

# animate path and plot field
magpy.show(egg, sensor, backend='plotly', animation=True, style_path_show=False)

fig = go.Figure()
for i,lab in enumerate(['Bx', 'By', 'Bz']):
    fig.add_trace(go.Scatter(x=ts/2*3, y=B[:,i], name=lab))
fig.update_layout(title='Field at sensor', xaxis_title='animation time (s)')
fig.show()
```



(examples-custom-source-objects)=
# CustomSource class

The class `CustomSource` allows users to integrate their own custom-object field computation into the Magpylib interface. For this, the argument `field_func` must be provided with a function that is then automatically called with `getB` and `getH`. This custom field function is treated like a core function. It must have the positional arguments `field` (with values `'B'` or `'H'`), and `observers` (must accept array_like, shape (n,3)) and return the B-field in units of mT and the H-field in units of kA/m with a similar shape. A fundamental example how to create a custom source object is shown below:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define field function
def custom_field(field, observers):
    """ user defined custom field
    position input: array_like, shape (n,3)
    returns: B-field, ndarray, shape (n,3)
    """
    if field=='B':
        return np.array(observers)*2
    return np.array(observers)

# custom source
source = magpy.misc.CustomSource(field_func=custom_field)

# compute field with 2-pixel sensor
sensor = magpy.Sensor(pixel=((1,1,1), (2,2,2)))

B = magpy.getB(source, sensor)
print(B)
H = magpy.getH(source, sensor)
print(H)
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
    field = lambda field, observers: monopole_field(charge, observers)
    monopole = magpy.misc.CustomSource(
        field_func=field,
        style_model3d_showdefault=False,
    )
    monopole.style.model3d.add_trace(trace_pole)
    return monopole

quadrupole = magpy.Collection(*[create_pole(q) for q in [1,1,-1,-1]])

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
    xlabel='x-position (mm)',
    ylabel='y-position (mm)',
    zlabel='z-position (mm)',
)
ax2.set(
    title='Quadrupole field',
    xlabel='x-position (mm)',
    ylabel='z-position (mm)',
    aspect=1,
)
fig.colorbar(cp, label='[$charge/mm^2$]', ax=ax2)

plt.tight_layout()
plt.show()
```
