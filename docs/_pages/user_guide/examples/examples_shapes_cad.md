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

(examples-shapes-cad)=

# Magnets from CAD

The easiest way to create complex magnet shapes from CAD files is through Pyvista using the [TriangularMesh class](docu-magpylib-api-trimesh). Pyvista supports *.stl files, and any open CAD file format is easily transformed to stl.

```{warning}
CAD files might include many Triangles, especially when dealing with round sides and edges, that do not significantly contribute to the field and will slow down the Magpylib computation.
```

```{code-cell} ipython3
import pyvista as pv
from magpylib.magnet import TriangularMesh

# Import *.stl file with Pyvista
mesh = pv.read("logo.stl")

# Transform into Magpylib magnet
magnet = TriangularMesh.from_pyvista(
    polydata=mesh,
    polarization=(1,-1,0),
    check_disconnected=False,
)
magnet.show(backend="plotly")
```

```{hint}
A quick way to work with cad files, especially transforming 2D *.svg to 3D *.stl, is provided by [Tinkercad](https://www.tinkercad.com).
```
