---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

(examples-vis-vectorfield)=

# Pixel Vector Field (Quiver Plot)

:::{versionadded} 5.2
Pixel Vector Field
:::


The `Sensor` object with its `pixel` can be conveniently used for visualizing the vector fields `"B"`, `"H"`, `"M"`, or `"J"` in the form of quiver plots. A detailed documentaion of this functionlity is found in the [documentartion](styles-pixel-vectorfield). This notebook provides examples of how to use these features effectively, along with explanations of the relevant parameters.

## Example 1:

Simple example using pixel field functionality combined with magnet transparency displaying the B field on a surface that passes through the magnet.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import magpylib as magpy

# Define a magnet with opacity
magnet = magpy.magnet.Cuboid(
    polarization=(1, 1, 0),
    dimension=(4e-3, 4e-3, 2e-3),
    style_opacity=.5,
)

# Create a grid of pixel positions in the xy-plane
xy_grid = np.mgrid[-4e-3:4e-3:15j, -4e-3:4e-3:15j, 0:0:1j].T[0]

# Create a sensor with pixel array and pixel field style
sens = magpy.Sensor(
    pixel=xy_grid,
    style_pixel_field_vectorsource="B",
    style_pixel_field_sizemode="log",
)

# Display the sensor and magnet using the Plotly backend
magpy.show([sens, magnet], backend='plotly')
```

## Example 2: 

Sensor pixels are not restricted to any specific grid structure and can be positioned freely to represent curved surfaces, lines, or individual points of interest. 

The example below demonstrates the visualization of the magnetic field of a magnetic pole wheel evaluated along curved surfaces and lines.

```{code-cell} ipython3
:tags: [hide-input]

from numpy import pi, sin, cos, linspace
import magpylib as magpy

# Create a pole wheel magnet composed of 12 alternating cylinder segments
pole_wheel = magpy.Collection()
for i in range(12):
    zone = magpy.magnet.CylinderSegment(
        dimension=(1.8, 2, 1, -15, 15),
        polarization=((-1)**i, 0, 0),
    ).rotate_from_angax(30*i, axis='z')
    pole_wheel.add(zone)

# Sensor 1: Pixel line along a circle in the xz-plane
ang1 = linspace(0, 2*pi, endpoint=False)
pixel_line = [(cos(a), 0, sin(a)) for a in ang1]

sensor1 = magpy.Sensor(
    pixel=pixel_line,
    style_pixel_field_vectorsource="H",
)

# Sensor 2: Curved surface (vertical cylinder segment)
z_values = linspace(-1, 1, 10)
ang2 = linspace(-9*pi/8, -2*pi/8, 30)
pixel_grid2 = [[(3.5*cos(a), 3.5*sin(a), z) for a in ang2] for z in z_values]

sensor2 = magpy.Sensor(
    pixel=pixel_grid2,
    style_pixel_field = {
        "vectorsource":"H",
        "sizemode" : "constant",
        "colorscale" : "Blues",
        "symbol" : "arrow3d",
    }
)

# Sensor 3: Curved surface (horizontal annular sector)
r_values = linspace(3, 4, 5)
ang3 = linspace(-pi/8, 6*pi/8, 30)
pixel_grid3 = [[(r*cos(a), r*sin(a), 0) for a in ang3] for r in r_values]

sensor3 = magpy.Sensor(
    pixel=pixel_grid3,
    style_pixel_field = {
        "vectorsource":"H",
        "sizemode" : "log",
        "colorscale" : "Plasma",
        "symbol" : "arrow3d",
    }
)

# Display sensors and magnets using Plotly backend
magpy.show(
    [sensor1, sensor2, sensor3, pole_wheel],
    backend='plotly',
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

## Quiver Plot Animation

animation















```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 51
pix_per_dim = 12

c1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1), style_opacity=0.2)
c1.rotate_from_angax(np.linspace(0, 180, path_len), "z", start=0)
c1.rotate_from_angax(np.linspace(0, 180, path_len), "x", start=0)

ls = np.linspace(-1, 1, pix_per_dim)
s1 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))

magpy.show(
    c1,
    s1,
    animation=True,
    style_pixel_field_symbol="arrow3d",
    style_pixel_field_vectorsource="B",
)
```

### Display B, H, J, or M Field

```{note}
Null or NaN field values are not displayed via a directional symbol but are visible by default.
```

```{code-cell} ipython3
# Only 10 interactive 3d plots (Webgl contexts) can be displayed at time
# The following will display subsequent plots as non-interative png images
import plotly.io as pio
pio.renderers.default = 'png'
pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.update(
    width=1400,  # Set default width
    height=600  # Set default height
)
pio.templates.default = "custom"
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 10

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 0] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
for i, vectorsource in enumerate("BHJM"):
    s1 = s0.copy(
        style_pixel_field_vectorsource=vectorsource,
    )
    s1.style.label = f"{vectorsource}-field"
    objects.append({"objects": (c1, s1), "col": i + 1})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

### Display Field Magnitude via Coloring

```{note}
Field coloring can be set independently of the field vector source. If not specified, it refers to the vector source magnitude. If set to `False`, no coloring is used, and symbols are displayed in black.
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 10

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 0] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
for i, colorsource in enumerate(("H", "Jxy", "Bz", False)):
    s1 = s0.copy(
        style_pixel_field_vectorsource="B",
        style_pixel_field_colorsource=colorsource,
    )
    s1.style.label = f"B-field, color: {colorsource}"
    objects.append({"objects": (c1, s1), "col": i + 1})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

### Use Different Directional Symbols

```{note}
Default is `"cone"` (can be set globally like any other style).
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 10

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 0] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
for i, symbol in enumerate(("cone", "arrow3d", "arrow")):
    s1 = s0.copy(
        style_pixel_field_vectorsource="B",
        style_pixel_field_symbol=symbol,
    )
    s1.style.label = f"B-field, symbol: {symbol}"
    objects.append({"objects": (c1, s1), "col": i + 1})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

### Set the Sizing Modes of Directional Symbols

```{note}
Default is `"constant"` (can be set globally like any other style).
Like for coloring, sizing is normalized for the min-max values of the field values over the whole sensor path, but for each sensor individually.
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 10

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 0] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
for i, sizemode in enumerate(("constant", "log", "linear")):
    s1 = s0.copy(
        style_pixel_field_vectorsource="B",
        style_pixel_field_sizemode=sizemode,
    )
    s1.style.label = f"B-field, sizemode: {sizemode}"
    objects.append({"objects": (c1, s1), "col": i + 1})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

### Edge Cases: Hide `Null` or `NaN` Values

```{note}
Null and NaN values are treated the same. These pixels can be hidden if desired.
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 5

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 0] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
col = 0
for vectorsource in "BJ":
    for shownull in (True, False):
        col += 1
        s1 = s0.copy(
            style_pixel_field_vectorsource=vectorsource,
            style_pixel_field_shownull=shownull,
        )
        s1.style.label = f"{vectorsource}-field, shownull: {shownull}"
        objects.append({"objects": (c1, s1), "col": col})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```

### Color Scales

```{note}
Other color scales are available (a curated list of common sequential colors between Matplotlib and Plotly).
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import magpylib as magpy

path_len = 1
pix_per_dim = 10

c1 = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_opacity=0.2,
)
ls = np.linspace(-1, 1, pix_per_dim)
s0 = magpy.Sensor(
    pixel=[[x, y, 1] for x in ls for y in ls],
    position=(0, 0, 0),
)
objects = []
for i, colorscale in enumerate(("Viridis", "Inferno", "Oranges", "RdPu")):
    s1 = s0.copy(
        style_pixel_field_vectorsource="B",
        style_pixel_field_colorscale=colorscale,
    )
    s1.style.label = f"B-field, colorscale: {colorscale}"
    objects.append({"objects": (c1, s1), "col": i + 1})

magpy.show(
    *objects,
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
```
