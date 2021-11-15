"""
=================
Display styling
=================
"""

#%%
# Displaying objects may not yield by default the visual representation the user wants. For this cases, the library includes
# a variety of styling options that can be applied at multiple levels in the user's code. 
#  
# .. warning::
#   Even if both backends can display all object of the library, there is no 100% feature parity between
#   them. Some of the differences include (non-exhaustive list):
#   
#   - ``magnetization.size`` -> ``matplotlib`` only
#   - ``magnetization.color`` -> ``plotly`` only


# %%
# List of available styles
import magpylib as magpy
from magpylib._src.default_utils import linearize_dict, get_defaults_dict
style = get_defaults_dict('display.style')
lin_style = linearize_dict(style)
print('\n'.join(f"{k!r}: {v!r}" for k,v in lin_style.items()))

#%%
# Display figure on your own canvas (here Matplotlib 3D axis):

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