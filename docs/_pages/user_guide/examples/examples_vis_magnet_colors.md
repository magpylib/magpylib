---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-vis-magnet-colors)=
# Magnet Colors

The polarization direction of a permanent magnet is often graphically displayed with the help of colors. However, there is no unified color scheme that everyone agrees on.

```{figure} ../../../_static/images/examples_vis_magnet_colors.png
:width: 100%
:align: center
:alt: Different magnets in different color schemes.

Some magnet coloring examples taken from the web.
```

Magpylib uses the DIN Specification 91411 (soon [INNOMAG Guideline](examples-app-scales-IGL)) standard as default setting. The tri-color scheme has the advantage that in multi-pole elements it becomes clear which north is "connected" to which south.

```{hint}
The color schemes often seem to represent homogeneous magnetic polarizations, also called "ideal typical" magnets. However, they often just represent general "pole patterns" that are not necessarily the result of homogeneous polarizations, see also {ref}`examples-misc-inhom`, and {ref}`examples-tutorial-modeling-magnets`.
```

As described in the [style documentation](guide-graphic-styles) in detail, users can easily tune the magnet color schemes. The `style` options are:
- `style_magnetization_color_mode` with options `'tricolor'` and `'bicolor'`
- `style_magnetization_color_north`, `middle`, and `south` with color inputs
- `style_magnetization_color_transition` with float input setting the color-transition

```{code-cell} ipython
:tags: [hide-input]

import magpylib as magpy

print('APPLYING DIFFERENT MAGNET COLOR SCHEMES')

# Create a magnetization style dictionary
mstyle = dict(
    mode="color+arrow",
    color=dict(north="magenta", middle="white", south="turquoise"),
    arrow=dict(width=2, color="k")
)

# Create magnet and apply style
sphere = magpy.magnet.Sphere(
    polarization=(1, 1, 1),
    diameter=1,
    position=(-1, 1, 0),
    style_magnetization=mstyle,
)

# Create a second magnet with different style
cube = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),
    dimension=(1, .2, .2),
    position=(1, 1, 0),
    style_magnetization_color_mode="bicolor",
    style_magnetization_color_north="r",
    style_magnetization_color_south="g",
    style_magnetization_color_transition=0,
)

# Create a third magnet with different style
cyl = magpy.magnet.CylinderSegment(
    polarization=(1, 0, 0),
    dimension=(1.7, 2, .3, -145, -35),
)
cyl.style.magnetization.color.north = "cornflowerblue"
cyl.style.magnetization.color.south = "orange"

# Show all three
magpy.show(sphere, cube, cyl, backend='plotly', style_legend_show=False)
```
