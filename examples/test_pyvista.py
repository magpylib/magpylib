"""
Streamtube field
=========

Display the magnetic field with Pyvista streamtubes
"""
# %%
# [Pyvista](https://docs.pyvista.org/) is an incredible VTK based tool for 3D plotting and mesh analysis.
# The following example shows how to compute and display 3D field lines of `coil1` with Pyvista. To run this example, the user must install Pyvista (`pip install pyvista`). By removing the command `jupyter_backend='static'` in `show`, the 3D figure becomes interactive.
import numpy as np
import pyvista as pv

import magpylib as magpy

coil1 = magpy.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpy.current.Loop(
        current=100,
        diameter=10,
        position=(0, 0, z),
    )
    coil1.add(winding)

grid = pv.UniformGrid(
    dimensions=(41, 41, 41),
    spacing=(2, 2, 2),
    origin=(-40, -40, -40),
)

# compute B-field and add as data to grid
grid["B"] = coil1.getB(grid.points)

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
magpy.show(coil1, canvas=pl, backend="pyvista")

# add streamlines
pl.add_mesh(
    strl.tube(radius=0.2),
    cmap="bwr",
    scalar_bar_args=legend_args,
)
# display scene
pl.camera.position = (160, 10, -10)
pl.set_background("white")
pl.show()
