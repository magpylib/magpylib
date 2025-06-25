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

# Pixel Field (Quiver Plot)

:::{versionadded} 5.2
Pixel Vector Field
:::

The `Sensor` object with its `pixel` can be conveniently used for visualizing the vector fields `"B"`, `"H"`, `"M"`, or `"J"` in the form of quiver plots. A detailed documentaion of this functionlity is found in the [documentartion](styles-pixel-vectorfield). This notebook provides examples of how to use these features effectively, along with explanations of the relevant parameters.

## Example 1: Transparent Magnet

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

## Example 2: Complex Pixel Grids

Sensor pixels are not restricted to any specific grid structure and can be positioned freely to represent curved surfaces, lines, or individual points of interest.

The example below demonstrates the visualization of the magnetic field of a magnetic pole wheel evaluated along curved surfaces and lines using different color maps and arrow shapes.

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

## Example 3: Pixel Field Animation

The pixel field can be combined with animation to create spectacular visualizations, such as displaying the magnetic field of rotating magnets.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import magpylib as magpy

# Create a cuboid magnet with vertical polarization
magnet = magpy.magnet.Cuboid(
    polarization=(0, 0, 1),
    dimension=(1, 3, 1)
)

# Apply a rotation to the Cuboid that generates a path with 51 steps
magnet.rotate_from_angax(
    angle=np.linspace(0, 360, 51),
    axis="y",
    start=0
)

# Create a sensor with pixel grid in the xy-plane at z=1
pixel_grid = np.mgrid[-2:2:12j, -2:2:12j, 1:1:1j].T[0]
sensor = magpy.Sensor(pixel=pixel_grid)

# Display as animation in the plotly backend
magpy.show(
    magnet,
    sensor,
    animation=True,
    style_pixel_field_symbol="arrow3d",
    style_pixel_field_vectorsource="B",
    backend="plotly",
)
