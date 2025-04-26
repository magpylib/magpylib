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

(examples-app-end-of-shaft)=

# Magnetic Angle Sensor

End of shaft angle sensing is a classic example for a magnetic position system. The goal is to determine the angular position of a rotating shaft. A magnet, typically a diametrically magnetized cylinder, is mounted at the end of the shaft. A 2D sensor is mounted below. When the shaft rotates the two sensor outputs will be $s_1=B_0 sin(\varphi)$ and $s_2=B_0 cos(\varphi)$, so that the angle is uniquely given by $\varphi = arctan_2(s_1,s_2)$.

In the example below we show such a typical end-of-shaft system with a 2-pixel sensor, that is commonly used to eliminate external stray fields. In addition, we assume that the magnet is not perfectly mounted at the end of the shaft, but slightly displaced to the side, which results in a wobble motion. Such tolerances are easily implemented with Magpylib, they can be visualized and their influence on the sensor output signal can be tested quickly.

```{code-cell} ipython3
import numpy as np
import plotly.express as px
import magpylib as magpy
import plotly.graph_objects as go

# Create magnet
magnet = magpy.magnet.Cylinder(
    polarization=(1, 0, 0),
    dimension=(.06, .02),
    position=(0, 0, .015),
    style_label="Magnet",
    style_color=".7",
)

# Create shaft dummy with 3D model
shaft = magpy.misc.CustomSource(
    position=(0, 0, .07),
    style_color=".7",
    style_model3d_showdefault=False,
    style_label="Shaft",
)
shaft_trace = magpy.graphics.model3d.make_Prism(
    base=20,
    diameter=.1,
    height=.1,
    opacity=0.3,
)
shaft.style.model3d.add_trace(shaft_trace)

# Shaft rotation / magnet wobble motion
displacement = .01
angles = np.linspace(0, 360, 72)
coll = magnet + shaft
magnet.move((displacement, 0, 0))
coll.rotate_from_angax(angles, "z", anchor=0, start=0)

# Create sensor
gap = .03
sens = magpy.Sensor(
    position=(0, 0, -gap),
    pixel=[(.01, 0, 0), (-.01, 0, 0)],
    style_pixel_size=0.5,
    style_size=1.5,
)

# Show 3D animation of wobble motion
fig1 = go.Figure()
magpy.show(magnet, sens, shaft, animation=True, backend="plotly", canvas=fig1)
fig1.update_layout(scene_camera_eye_z=-1.1)
fig1.show()

# Show sensor output in plotly
fig2 = go.Figure()
df = sens.getB(magnet, output="dataframe")
df["angle (deg)"] = angles[df["path"]]

fig2 = px.line(
    df,
    x="angle (deg)",
    y=["Bx", "By"],
    line_dash="pixel",
    labels={"value": "Field (T)"},
)
fig2.show()
```

```{code-cell} ipython3

```
