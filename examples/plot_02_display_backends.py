"""
=================
Display backends
=================
"""

#%%
# The ``magpylib`` package is shipped with a display function which provides a graphical representation
# of every magnet, current and sensor of the library. To date the library includes two possible backends:
#
# - matplotlib (by default)
# - plotly

# %%
# Display multiple objects, object paths, markers in 3D using Matplotlib:

import magpylib as magpy
magnet = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 2, 3))
sens = magpy.Sensor(position=(0, 0, 3))
magpy.display(magnet, sens, zoom=1)

#%%
# Display figure on your own canvas (here Matplotlib 3D axis):

import matplotlib.pyplot as plt
import magpylib as magpy

my_axis = plt.axes(projection="3d")
src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move([(0.1, 0, 0)] * 50, increment=True)
src.rotate_from_angax(angle=[10] * 50, axis="z", anchor=0, start=0, increment=True)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
magpy.display(src, sens, canvas=my_axis)
plt.show()

#%%
# The same objects can also be displayed using the ``plotly`` plotting backend
import plotly.graph_objects as go
import magpylib as magpy

fig = go.Figure()
src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move([(0.1, 0, 0)] * 50, increment=True)
src.rotate_from_angax(angle=[10] * 50, axis="z", anchor=0, start=0, increment=True)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
magpy.display(src, sens, canvas=fig, backend="plotly")
fig

#%%
# The display function is also available as a class method and can be called for every object separately.
import plotly.graph_objects as go
import magpylib as magpy

fig = go.Figure()
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
sens.display(canvas=fig, backend="plotly", zoom=1, style_size=5)
fig

# %%
# .. note::
#   Displaying figures with the ``plotly`` backend does not necessarily need to provide a canvas. It is
#   shown here only as an example.