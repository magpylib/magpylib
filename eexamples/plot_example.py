"""
===========================================
Examples of the use of the display function
===========================================

The `magpylib` library integrates a `display` function attached to every object, which allows its
graphical representation. To date the library includes two possible backends:
- matplotlib (by default)
- plotly
"""

#%%
# Display multiple objects, object paths, markers in 3D using Matplotlib:

import magpylib as magpy

src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move([(0.1, 0, 0)] * 50, increment=True)
src.rotate_from_angax(angle=[10] * 50, axis="z", anchor=0, start=0, increment=True)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
magpy.display(src, sens)

#%%
# The same objects can also be displayed using the `plotly` plotting backend
import plotly.graph_objects as go

fig = go.Figure()
magpy.display(src, sens, canvas=fig, path="animate", backend="plotly")
fig

#%%
# Display figure on your own canvas (here Matplotlib 3D axis):

import matplotlib.pyplot as plt
import magpylib as magpy

my_axis = plt.axes(projection="3d")
magnet = magpy.magnet.Cuboid(magnetization=(1, 1, 1), dimension=(1, 2, 3))
sens = magpy.Sensor(position=(0, 0, 3))
magpy.display(magnet, sens, canvas=my_axis, zoom=1)
plt.show()

#%%
# Use sophisticated figure styling options accessible from defaults, as individual
# object styles or as global style arguments in display.

import magpylib as magpy

src1 = magpy.magnet.Sphere((1, 1, 1), 1)
src2 = magpy.magnet.Sphere((1, 1, 1), 1, (1, 0, 0))
magpy.defaults.display.style.magnet.magnetization.size = 2
src1.style.magnetization.size = 1
magpy.display(src1, src2, style_color="r", zoom=3)
