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
# Magnet colors

The polarization direction of a permanent magnet is often graphically displayed with the help of colors. However, there is no unified color scheme that everyone agrees on. The following image shows some color-examples from the web.

![](../../../_static/images/examples_vis_magnet_colors.png)

Magpylib uses the DIN Specification 91411 (soon 91479) standard as default setting. The tri-color scheme has the advantage that for multi-pole elements it becomes clear which north is "connected" to which south.

```{hint}
The color schemes often seem to represent homogeneous polarizations, referred to as "ideal typical" magnets in DIN Specification 91479. However, they often just represent general "pole patterns", i.e. rough sketches where the field goes in and where it comes out, that are not the result of homogeneous polarizations. On this topic review also the examples example {ref}`examples-misc-inhom`, and the tutorial {ref}`examples-tutorial-modelling-magnets`.
```

With Magpylib users can easily tune the magnet color schemes. The `style` options are `tricolor` with north, middle and south colors, and `bicolor` with north and south colors.

```{code-cell} ipython
import magpylib as magpy

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

More information about styles and how to apply them is given in the user-guide [style section](guide-graphic-styles).
