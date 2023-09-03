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

(gallery-shapes-pyvista)=

# Pyvista Bodies

[Pyvista](https://docs.pyvista.org/version/stable/) is a powerful open-source tool for the creation and visualization of meshes. Pyvista `PolyData` objects can be directly transformed into Magpylib `TriangularMesh` magnets via the classmethod `from_pyvista`.

```{note}
The Pyvista library used in the following examples is not automatically installed with Magpylib. A Pyvista installation guide is found [here](https://docs.pyvista.org/getting-started/installation.html).
```

## Dodecahedron Magnet

In this example a Magpylib magnet is generated directly from a Pyvista body.

```{code-cell} ipython3
import numpy as np
import pyvista as pv
import magpylib as magpy

# Create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=dodec_mesh,
)

# Add a sensor with path
sens = magpy.Sensor(position=np.linspace((-2,0,1), (2,0,1), 100))

# Show system and field
with magpy.show_context(dodec, sens, backend='plotly') as s:
    s.show(col=1)
    s.show(col=2, output=['Bx', 'Bz'])

```

## Boolean operations with Pyvista

With Pyvista it is possible to build complex shapes with boolean geometric operations. However, such operations often result in open and disconnected meshes that require some refinement to produce solid magnets. The following example demonstrates the problem, how to analyze and fix it.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# Create a complex pyvista PolyData object using a boolean operation
sphere = pv.Sphere(radius=0.6)
cube = pv.Cube().triangulate()
obj = cube.boolean_difference(sphere)

# Construct magnet from PolyData object and ignore check results
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    check_disconnected="ignore",
    check_open="ignore",
    reorient_faces="ignore",
    style_label="magnet",
)

print(f'mesh status open: {magnet.status_open}')
print(f'mesh status disconnected: {magnet.status_disconnected}')
print(f"mesh status self-intersecting: {magnet.status_selfintersecting}")
print(f'mesh status reoriented: {magnet.status_reoriented}')

magnet.show(
    backend="plotly",
    style_mesh_open_show=True,
    style_mesh_disconnected_show=True,
)
```

The result cannot be used for magnetic field computation. Even if all faces were present, the reorient-faces algorithm would fail when these faces are disconnected. Such problems can be fixed by

1. giving Pyvista a finer mesh to work with from the start
2. Pyvista mesh cleaning (merge duplicate points, remove unused points, remove degenerate faces)

The following code produces a clean magnet .

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# Create a complex pyvista PolyData object using a boolean operation. Start with
# finer mesh and clean after operation
sphere = pv.Sphere(radius=0.6)
cube = pv.Cube().triangulate().subdivide(2)
obj = cube.boolean_difference(sphere)
obj = obj.clean()

# Construct magnet from PolyData object
magnet = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=obj,
    style_label="magnet",
)

print(f'mesh status open: {magnet.status_open}')
print(f'mesh status disconnected: {magnet.status_disconnected}')
print(f"mesh status self-intersecting: {magnet.status_selfintersecting}")
print(f'mesh status reoriented: {magnet.status_reoriented}')

magnet.show(backend="plotly")
```
