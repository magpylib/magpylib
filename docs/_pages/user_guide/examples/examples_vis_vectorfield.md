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

# Pixel Vector Field


:::{versionadded} 5.2
Pixel Vector Field
:::

The `Sensor` object with its `pixel` can be conveniently used for visualizing vector fields!
With this new capability, you can display the vector field for `"B"`, `"H"`, `"M"`, or `"J"` and overlay a magnitude (e.g., `"Bxy"`, `"Hz"`, etc.).

This notebook provides examples of how to use these features effectively, along with explanations of the relevant parameters.


## Parameter Overview

### `style.pixel.field` Parameters

- **`vectorsource`** (default=`None`): Defines the field source for vector visualization:
  - `None`: No field direction is shown.
  - `"B"`, `"H"`, `"J"`, `"M"`: Corresponding field vectors are displayed.

- **`colorsource`** (default=`None`): Defines the field source for the color scale:
  - `None`: Field magnitude coloring is taken from the `vectorsource` input and shown via the magnitude of individual pixels.
  - `"B"`, `"Hxy"`, `"Jxyz"`, etc.: Corresponding field magnitude is shown via magnitude coloring of individual pixels.
  - `False`: No magnitude coloring is applied. Symbols are displayed with `pixel.color` if defined, otherwise `"black"`.

- **`symbol`** (default=`"cone"`): Defines the symbol used to represent valid and non-null vectors:
  - `"cone"`: 3D cone representation.
  - `"arrow3d"`: 3D arrow representation.
  - `"arrow"`: 3D line representation.
  - Null or invalid vectors are displayed as:
    - A point (when `field.symbol` is `"arrow"`).
    - A cuboid pixel (when `field.symbol` is `"cone"` or `"arrow3d"`).

- **`shownull`** (default=`True`): Toggles the visibility of null vectors:
  - `True`: Null vectors are displayed.
  - `False`: Null vectors are hidden.

- **`sizemode`** (default=`"constant"`): Defines how arrows are sized relative to the `vectorsource` magnitude:
  - `"constant"`: Uniform size.
  - `"linear"`: Proportional to the magnitude.
  - `"log"`: Proportional to the normalized logarithm of the magnitude.


### Additional Features

- **Color Scales**: Supports predefined color scales (e.g., `"Viridis"`, `"Inferno"`, `"Jet"`) or which are common to both `Plotly` and `Matplotlib`.
- **Normalization**: Field vectors and magnitude normalization are applied per sensor path for each sensor individually.
- **`style.pixel.symbol`** (default=`None`): Accepts valid scatter symbols (e.g., `"."`). Only applies if `pixel.field.vectorsource` is `None` or for invalid/null vectors.
- **Pixel Hulls**: Hulls over pixels are shown only if no field values are provided.
- **Backend Compatibility**: Works seamlessly with all supported backends:
  - Matplotlib
  - Plotly
  - PyVista

+++

## Examples

### Animated B-field

```{note}
Default is `"cone"` (can be set globally like any other style).
```

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
