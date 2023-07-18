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

(gallery_pv_streamlines)=

# Streamlines

something something

```{code-cell} ipython3
import numpy as np
import pyvista as pv

import magpylib as magpy

# create a magnet with Magpylib
magnet = magpy.magnet.Cylinder((0, 0, 1000), (11, 4))

# create a grid with Pyvista
grid = pv.UniformGrid(
    dimensions=(21, 21, 21),
    spacing=(2, 2, 2),
    origin=(-20, -20, -20),
)

# compute B-field and add as data to grid
grid["B"] = magnet.getB(grid.points)

# compute field lines
seed = pv.Disc(inner=1, outer=4, r_res=2, c_res=6)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_time=180,
    initial_step_length=0.01,
    integration_direction="both",
)

# create plotting scene
pl = pv.Plotter()

# add legend to scene
legend_args = {
    "title": "B (mT)",
    "title_font_size": 20,
    "color": "black",
    "position_y": 0.25,
    "vertical": True,
}

# add magnet to scene
magpy.show(magnet, canvas=pl, backend="pyvista")

# add streamlines to scene
pl.add_mesh(
    strl.tube(radius=0.2),
    cmap="bwr",
    scalar_bar_args=legend_args,
)

# display scene
pl.camera.position = (30, 30, 20)
pl.set_background("white")
pl.show()
```