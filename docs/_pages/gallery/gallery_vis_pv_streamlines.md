---
orphan: true
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

(gallery-vis-pv-streamlines)=

# Field lines with Pyvista streamlines

Pyvista offers field-line computation and visualization in 3D. In addition to the field computation, Magpylib offers magnet visualization that seamlessly integrates into a Pyvista plotting scene.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy

# This line is only needed for pyvista rendering in a jupyter notebook
pv.set_jupyter_backend("panel")

# Create a magnet with Magpylib
magnet = magpy.magnet.Cylinder((0, 0, 1000), (10, 4))

# Create a 3D grid with Pyvista
grid = pv.UniformGrid(
    dimensions=(41, 41, 41),
    spacing=(1, 1, 1),
    origin=(-20, -20, -20),
)

# Compute B-field and add as data to grid
grid["B"] = magnet.getB(grid.points)

# Compute the field lines
seed = pv.Disc(inner=1, outer=3, r_res=1, c_res=6)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_time=180,
    initial_step_length=0.01,
    integration_direction="both",
)

# Create a Pyvista plotting scene
pl = pv.Plotter()

# Add magnet to scene
magpy.show(magnet, canvas=pl, backend="pyvista")

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
    strl.tube(radius=0.2),
    cmap="bwr",
    scalar_bar_args=legend_args,
)

# Prepare and show scene
pl.camera.position = (30, 30, 20)
pl.show()
```