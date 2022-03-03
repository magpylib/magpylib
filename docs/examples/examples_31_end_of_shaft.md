---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# End of shaft

In this example we consider a typical end-of-shaft system where a diametral cylindrical magnet is mounted at the end of a rotating shaft. A magnetic field sensor is mounted centrally below with the purpose of detecting the angular position of the shaft.

The purpose of this example is to
1. compute and understand the fields at the sensor.
1. include the influence of a magnet positioning error in the form of a small displacement from the central axis.
2. show how complex systems and motions can be set up and demonstrated easily. For this 

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# create magnet
src = magpy.magnet.Cylinder(
    magnetization=(1000,0,0),
    dimension=(6,2),
    position=(0,0,1.5),
    style_label='Magnet',
    style_color='.7',
)

# add model3d to magnet for shaft visualization
shaft_trace = magpy.display.plotly.make_BasePrism(
    base_vertices=20,
    diameter=10,
    height=10,
)
shaft = magpy.misc.CustomSource(
    position=(0,0,7),
    style_color='.7',
)
shaft.style.model3d.add_trace(shaft_trace, backend='plotly')

# magnet wobbles when shaft rotates
displacement = 1
angles = np.linspace(0, 360, 72)
coll = src + shaft
src.move((displacement, 0, 0))
coll.rotate_from_angax(angles, 'z', anchor=0, start=0)

# sensor
gap = 3
sens = magpy.Sensor(
    position=(0,0,-gap),
    pixel=[(1,0,0), (-1,0,0)],
    style_pixel_size=0.5,
    style_size=1.5,
)

# show 3D animation with wobble motion
fig1 = go.Figure()
magpy.show(src, sens, shaft, animation=True, backend='plotly', canvas=fig1)
fig1.update_layout(scene_camera_eye_z=-1.1)
fig1.show()

# show sensor output in plotly
fig2 = go.Figure()
B = sens.getB(src)
for px,dash in zip([0,1], ['solid', 'dash']):
    for i,xy,col in zip([0,1], ['x','y'], ['red', 'green']):
        fig2.add_trace(go.Scatter(x=angles, y=B[:,px,i],
            name=f"pixel{px}-B{xy}",
            line=dict(color=col, dash=dash),
        ))
fig2.update_layout(
    xaxis=dict(title='angle [deg]'),
    yaxis=dict(title='field [mT]')
)
fig2.show()
```
