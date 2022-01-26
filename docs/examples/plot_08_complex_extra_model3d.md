---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Complex extra 3d-model

With the `model3d.extra` style property, it is possible to attach an extra 3d-model representation for any `magpylib` object, as long as it is supported by the chosen plotting backend. The `plotly` backend supports `mesh3d` objects and with the `numpy-stl` package, it becomes possible to import a STL CAD file and transform it to a `mesh3d` object with a little helper function.

```{note}
In order to use this functionality the `numpy-stl` package needs to be installed.
```

```{code-cell} ipython3
import os
import tempfile

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
import requests
from stl import mesh  # needs numpy-stl installed


def get_stl_color(x, return_rgb_string=True):
    sb = f"{x:015b}"[::-1]
    r = int(255 / 31 * int(sb[:5], base=2))
    g = int(255 / 31 * int(sb[5:10], base=2))
    b = int(255 / 31 * int(sb[10:15], base=2))
    if return_rgb_string:
        color = f"rgb({r},{g},{b})"
    else:
        color = (r, g, b)
    return color


# define stl to mesh3d function
def stl2mesh3d(stl_file, recenter=False, backend="matplotlib"):
    """
    an array of faces/triangles is read by numpy-stl from a stl file;  
    this function extracts the unique vertices and the triangulation values and
    returns depending on the backend the corresponding dictionary for further
    magpylib use as extra 3d-model.
    The array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of
    the same vertex; only unique vertices are extracted from all mesh triangles
    """
    stl_mesh = mesh.Mesh.from_file(stl_file)
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    vertices, ixr = np.unique(
        stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
    )
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    if recenter:
        vertices = vertices - 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    colors = stl_mesh.attr.flatten()
    x, y, z = vertices.T
    model3d = {"backend": backend}
    if backend == "matplotlib":
        triangles = np.array([i, j, k]).T
        model3d.update(
            trace=dict(type="plot_trisurf", args=(x, y, z), triangles=triangles),
            coordsargs={"x": "args[0]", "y": "args[1]", "z": "args[2]"},
        )
    elif backend == "plotly":
        facecolor = np.array(
            [get_stl_color(x, return_rgb_string=True) for x in colors]
        ).T
        model3d.update(
            trace=dict(
                type="mesh3d",
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                facecolor=facecolor
                # coordsargs is by default: {"x": "x", "y": "y", "z": "z"}
            ),
        )
    else:
        raise ValueError(
            """Backend type not understood, must be one of ['matplotlib', 'plotly']"""
        )
    return model3d


# import stl file
url = "https://raw.githubusercontent.com/magpylib/magpylib-files/main/PG-SSO-3-2.stl"
file = url.split("/")[-1]
with tempfile.TemporaryDirectory() as tmpdirname:
    fn = os.path.join(tmpdirname, file)
    with open(fn, "wb") as f:
        response = requests.get(url)
        f.write(response.content)

    # create mesh3d
    model3d_matplotlib = stl2mesh3d(fn, backend="matplotlib")
    model3d_plotly = stl2mesh3d(fn, backend="plotly")
# create sensor, add extra 3d-model, create path
sensor = magpy.Sensor(position=(-15, 0, -6))
sensor.style = dict(model3d_extra=[model3d_matplotlib, model3d_plotly])
sensor.rotate_from_angax(np.linspace(0, 150, 33), "z", anchor=0, start=0)
sensor.move(np.linspace((0, 0, 0), (0, 0, 15), 33), start=0)
# create source, and Collection
cuboid = magpy.magnet.Cylinder(magnetization=(0, 0, 1000), dimension=(20, 30))
collection = sensor + cuboid

# display animated system with matplotlib backend
magpy.display(*collection, style_path_show=8, style_magnetization_size=0.4, backend="matplotlib")

# display animated system with plotly backend
magpy.display(*collection, style_path_show=8, backend="plotly")
```


