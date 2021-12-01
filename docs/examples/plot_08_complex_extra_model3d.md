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
import requests
import tempfile
import numpy as np
import magpylib as magpy
import plotly.graph_objects as go
from stl import mesh # needs numpy-stl installed

# define stl to mesh3d function
def stl2mesh3d(stl_file, recenter=False):
    def get_stl_color(x):
        sb = f'{x:015b}'[::-1]
        r = int(255/31*int(sb[:5],base=2))
        g = int(255/31*int(sb[5:10],base=2))
        b = int(255/31*int(sb[10:15],base=2))
        color = f'rgb({r},{g},{b})'
        return color
    stl_mesh = mesh.Mesh.from_file(stl_file)
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    i = np.take(ixr, [3*k for k in range(p)])
    j = np.take(ixr, [3*k+1 for k in range(p)])
    k = np.take(ixr, [3*k+2 for k in range(p)])
    facecolor = np.vectorize(get_stl_color)(stl_mesh.attr.flatten())
    if recenter:
        vertices = vertices-0.5*(vertices.max(axis=0)+vertices.min(axis=0))
    x,y,z = vertices.T
    return dict(type='mesh3d', x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor)

# import stl file
url = 'https://raw.githubusercontent.com/magpylib/magpylib-files/main/PG-SSO-3-2.stl'
file = url.split('/')[-1]
with tempfile.TemporaryDirectory() as tmpdirname:
    fn = os.path.join(tmpdirname,file)
    with open(fn, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)
        
    # create mesh3d
    mesh3d = stl2mesh3d(fn)
    mesh3d['opacity'] = 0.5

# create sensor, add extra 3d-model, create path
sensor = magpy.Sensor(position=(-15,0,-6))
sensor.style = dict(model3d_extra=dict(backend='plotly', trace=mesh3d))
sensor.rotate_from_angax([5]*30, 'z', increment=True, anchor=(0,0,0))
sensor.move([[0,0,0.5]]*30, increment=True, start=0)

# create source, and Collection
cuboid = magpy.magnet.Cylinder(magnetization=(0,1000,0), dimension=(20,30))
collection =  sensor + cuboid

# display animated system with plotly backend
fig = go.Figure()
magpy.display(collection, canvas=fig, path='animate', backend='plotly')
fig.update_layout(height=600)
fig.show()
```


