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

(gallery-ext-complex-shapes-pv)=

# Complex shapes with Pyvista

## Example - Boolean operations with Pyvista

It is noteworthy that with [Pyvista](https://docs.pyvista.org/) it is possible to build complex shapes with boolean geometric operations. However, such operations often result in open and disconnected meshes that require some refinement to produce solid magnets. The following example demonstrates the problem, and how to view it with `show`.

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.6)
cube = pv.Cube().triangulate()
obj = cube.boolean_difference(sphere)

# use the `from_pyvista` classmethod to construct our magnet
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

Such problems can typically be avoided by
1. Subdivision of the initial triangulation (give Pyvista a finer mesh to work with from the start)
2. Cleaning (merge duplicate points, remove unused points, remove degenerate faces)

The following code solves these problems and produces a clean magnet.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# create a complex pyvista PolyData object with a boolean operation
sphere = pv.Sphere(radius=0.6)
dodec = pv.Cube().triangulate().subdivide(2)
obj = dodec.boolean_difference(sphere)
obj = obj.clean()

# use the `from_pyvista` classmethod to construct our magnet
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

## Example - Dodecahedron magnet from pyvista

`TriangularMesh` magnets can be directly created from Pyvista `PolyData` objects via the classmethod `from_pyvista`.

```{note}
The Pyvista library used in the following examples is not automatically installed with Magpylib. A Pyvista installation guide is found [here](https://docs.pyvista.org/getting-started/installation.html).
```

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(0, 0, 100),
    polydata=dodec_mesh,
)

dodec.show()
```

We can now add a sensor and plot the B-field value along the sensor path.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import magpylib as magpy

# create a simple pyvista PolyData object
dodec_mesh = pv.Dodecahedron()

# create TriangularMesh object
tmesh_dodec = magpy.magnet.TriangularMesh.from_pyvista(
    magnetization=(1000, 0, 0),
    polydata=dodec_mesh,
)

# add sensor and rotate around source
sens = magpy.Sensor(position=(2, 0, 0))
sens.rotate_from_angax(angle=np.linspace(0, 320, 50), axis="z", start=0, anchor=0)

# define matplotlib figure and axes
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(
    121,
    projection="3d",
    azim=-80,
    elev=15,
)
ax2 = fig.add_subplot(
    122,
)

# plot 3D model and B-field
magpy.show(tmesh_dodec, sens, canvas=ax1)
B = tmesh_dodec.getB(sens)
ax2.plot(B, label=["Bx", "By", "Bz"])

# plot styling
ax1.set(
    title="3D model",
    aspect="equal",
)
ax2.set(
    title="Dodecahedron field",
    xlabel="path index ()",
    ylabel="B-field (mT)",
)

plt.gca().grid(color=".9")
plt.gca().legend()
plt.tight_layout()
plt.show()
```