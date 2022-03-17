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

(examples-adding-CAD-model)=

# Adding a CAD model

As shown in {ref}`examples-own-3d-models`, it is possible to attach custom 3D model representations to any Magpylib object. In the example below we show how a standard CAD model can be transformed into a Magpylib graphic trace, and displayed by both `matplotlib` and `plotly` backends.

```{note}
The code below requires installation of the `numpy-stl` package.
```

```{code-cell} ipython3
import os
import tempfile
import requests
import numpy as np
from stl import mesh  # requires installation of numpy-stl
import magpylib as magpy


def get_stl_color(x):
    """ transform stl_mesh attr to plotly color"""
    sb = f"{x:015b}"[::-1]
    r = int(255 / 31 * int(sb[:5], base=2))
    g = int(255 / 31 * int(sb[5:10], base=2))
    b = int(255 / 31 * int(sb[10:15], base=2))
    return f"rgb({r},{g},{b})"


def trace_from_stl(stl_file, backend='matplotlib'):
    """
    Generates a Magpylib 3D model trace dictionary from an *.stl file.
    backend: 'matplotlib' or 'plotly'
    """
    # load stl file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # extract vertices and triangulation
    p, q, r = stl_mesh.vectors.shape
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    x, y, z = vertices.T

    # generate and return Magpylib traces
    if backend == 'matplotlib':
        triangles = np.array([i, j, k]).T
        trace = {
            'backend': 'matplotlib',
            'constructor': 'plot_trisurf',
            'args': (x, y, z),
            'kwargs': {'triangles': triangles},
        }
    elif backend == 'plotly':
        colors = stl_mesh.attr.flatten()
        facecolor = np.array([get_stl_color(c) for c in colors]).T
        trace = {
            'backend': 'plotly',
            'constructor': 'Mesh3d',
            'kwargs': dict(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor),
        }
    else:
        raise ValueError("Backend must be one of ['matplotlib', 'plotly'].")

    return trace


# load stl file from online resource
url = "https://raw.githubusercontent.com/magpylib/magpylib-files/main/PG-SSO-3-2.stl"
file = url.split("/")[-1]
with tempfile.TemporaryDirectory() as temp:
    fn = os.path.join(temp, file)
    with open(fn, "wb") as f:
        response = requests.get(url)
        f.write(response.content)

    # create traces for both backends
    trace_mpl = trace_from_stl(fn, backend='matplotlib')
    trace_ply = trace_from_stl(fn, backend='plotly')

# create sensor and add CAD model
sensor = magpy.Sensor(style_label='PG-SSO-3 package')
sensor.style.model3d.add_trace(trace_mpl)
sensor.style.model3d.add_trace(trace_ply)

# create magnet and sensor path
magnet = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(15,20))
sensor.position = np.linspace((-15,0,8), (-15,0,-4), 21)
sensor.rotate_from_angax(np.linspace(0, 200, 21), 'z', anchor=0, start=0)

# display with both backends
magpy.show(sensor, magnet, style_path_frames=5, style_magnetization_show=False)
magpy.show(sensor, magnet, style_path_frames=5, backend="plotly")
```
