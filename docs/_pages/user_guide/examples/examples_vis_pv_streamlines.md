---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
orphan: true
---

(examples-vis-pv-streamlines)=

# Pyvista 3D field lines

Pyvista offers field-line computation and visualization in 3D. In addition to the field computation, Magpylib offers magnet visualization that seamlessly integrates into a Pyvista plotting scene.

```{code-cell} ipython3
import magpylib as magpy
import pyvista as pv

# Create a magnet with Magpylib
magnet = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(0.010, 0.004))

# Create a 3D grid with Pyvista
grid = pv.ImageData(
    dimensions=(41, 41, 41),
    spacing=(0.001, 0.001, 0.001),
    origin=(-0.02, -0.02, -0.02),
)

# Compute B-field and add as data to grid
grid["B"] = magnet.getB(grid.points) * 1000  # T -> mT

# Compute the field lines
seed = pv.Disc(inner=0.001, outer=0.003, r_res=1, c_res=9)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_step_length=0.1,
    max_time=.02,
    integration_direction="both",
)

# Create a Pyvista plotting scene
pl = pv.Plotter()

# Add magnet to scene - streamlines units are assumed to be meters
magpy.show(magnet, canvas=pl, units_length="m", backend="pyvista")

# Prepare legend parameters
legend_args = {
    "title": "B (mT)",
    "title_font_size": 20,
    "color": "black",
    "position_y": 0.25,
    "vertical": True,
}

# Add streamlines and legend to scene
pl.add_mesh(
    strl.tube(radius=0.0002),
    cmap="bwr",
    scalar_bar_args=legend_args,
)

# Prepare and show scene
pl.camera.position = (0.03, 0.03, 0.03)
pl.show()
```