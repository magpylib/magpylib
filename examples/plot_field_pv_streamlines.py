"""
Magnetic field with streamtubes
===============================

Display the magnetic field with Pyvista streamtubes
"""
# %%
# something something
import numpy as np
import pyvista as pv

import magpylib as magpy

# create an aircoil with Magpylib
coil = magpy.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpy.current.Loop(
        current=100,
        diameter=10,
        position=(0, 0, z),
    )
    coil.add(winding)

# create a grid with Pyvista
grid = pv.UniformGrid(
    dimensions=(41, 41, 41),
    spacing=(4, 4, 4),
    origin=(-40, -40, -40),
)

# compute B-field and add as data to grid
grid["B"] = coil.getB(grid.points)

# compute field lines
seed = pv.Disc(inner=1, outer=5.2, r_res=3, c_res=12)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_time=180,
    initial_step_length=0.01,
    integration_direction="both",
)

# create plotting scene
pl = pv.Plotter()

# add field lines and legend to scene
legend_args = {
    "title": "B (mT)",
    "title_font_size": 20,
    "color": "black",
    "position_y": 0.25,
    "vertical": True,
}

# draw coils
magpy.show(coil, canvas=pl, backend="pyvista", style_color="r")

# add streamlines
pl.add_mesh(
    strl.tube(radius=0.2),
    cmap="bwr",
    scalar_bar_args=legend_args,
)
# display scene
pl.camera.position = (50, 50, 50)
pl.set_background("white")
pl.show()
