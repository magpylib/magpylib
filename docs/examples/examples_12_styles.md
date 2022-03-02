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

(examples-graphic-styles)=
# Styles

The graphic styles define how Magpylib objects are displayed visually when calling `show`. They can be fine-tuned and individualized in many ways.

```{warning}
Users should be aware that specifying style attributes massively increases object initializing time (from <50 to 100-500 $\mu$s).
While this may not be noticeable for a small number of objects, it can become an issue when initializating a lot of objects or when repeatedly creating objects in a loop, e.g. in an optimization algorithm.
If possible, avoid setting styles until it is plotting time.
```

There are multiple hierarchy levels that descide about the final graphical representation of the objects:

1. When no input is given, the **default style** will be applied.
2. **Individual styles** of objects will take precedence over the default values.
3. Collections will override the color property of all children.
4. Setting a **global style** in `show()` will take precedence over all other settings.

## Setting the default style

The default style is stored in the magpylib defaults in the form of nested attributes of the `magpylib.defaults.display.style` class. The top style level, they are separated into the following families,

- *base*: common properties for all families
- *magnet*: `Cuboid`, `Cylinder`, `Sphere`, `CylinderSegment`
- *current*: `Line`, `Loop`
- *sensor*: `Sensor`
- *markers*: markers in the `show` function

and can be set with **dot notation**,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.magnetization.show = True
magpy.defaults.display.style.magnet.magnetization.color.middle = "grey"
magpy.defaults.display.style.magnet.magnetization.color.mode = "bicolor"
```

by assigning a **style dictionary** with equivalent keys,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    "magnetization": {"show": True, "color": {"middle": "grey", "mode": "tricolor"}}
}
```

or by making use of the **`update` method**,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.magnetization.update(
    "show": True,
    "color": {"middle"="grey", mode="tricolor",}
)
```

Once set, the library default can always be restored with the `magpylib.style.reset()` method. The following practical example demonstrates how to create and set a user defined magnetization style as default,

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
src2 = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
src3 = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

print('Display before setting style defaults')
magpy.show(src1, src2, src3, backend="plotly")

user_defined_style = {
    "show": True,
    "color": {
        "transition": 1,
        "mode": "tricolor",
        "middle": "white",
        "north": "magenta",
        "south": "turquoise",
    },
}
magpy.defaults.display.style.magnet.magnetization = user_defined_style

print('Display after setting style defaults')
magpy.show(src1, src2, src3, backend="plotly")
```


## Magic underscore notation
<!-- +++ {"tags": [], "jp-MarkdownHeadingCollapsed": true} -->

To facilitate working with deeply nested properties, all style constructors and object style methods support the magic underscore notation. It enables referencing nested properties by joining together multiple property names with underscores. This feature mainly helps reduce the code verbosity and is heavily inspired by the `plotly` implementation (see [plotly underscore notation](https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation)).

With magic underscore notation, the previous examples can be written as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    "magnetization_show": True,
    "magnetization_color_middle": "grey",
    "magnetization_color_mode": "tricolor",
}
```

or directly as kwargs in the `update` method as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.update(
    magnetization_show=True,
    magnetization_color_middle="grey",
    magnetization_color_mode="tricolor",
)
```

## Setting individual styles

Any Magpylib object can have its own individual style that will take precedence over the default values when `show` is called.

```{note}
When setting individual styles, the object family specifier such as `magnet` or `current` which is required for the defaults settings, but is implicitly defined by the object type, can be ommited.
```

The following example shows the difference between setting an individual and a default current arrow size,

```python
import magpylib as magpy

# set default value
magpy.defaults.display.style.current.arrow.size = 2

# set individual object style
loop = magpy.current.Loop(current=10, diameter=10)
loop.style.arrow.size = 2
```

The following example demonstrates the application of individual styles for magnetization representation in Plotly

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset()  # omit when running as a standalone script

src1 = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
src2 = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
src3 = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

#setting individual styles
src1.style.update(magnetization_color_mode="tricycle")
src2.style.magnetization.color = dict(mode="tricolor")
src3.style.magnetization = {"color": {"mode":"bicolor"}}

magpy.show(src1, src2, src3, backend="plotly")
```

## Setting style via collections

When displaying collections, the collection object `color` property will be automatically assigned to all its children and override default and individual styles. In addition, it is possible to modify the style of all children at the same time with the `set_children_styles` method. In this case, non-matching properties are simply ignored.

In the following example we show how the french magnetization style is applied to all children in a collection,

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset()  # omit when running as a standalone script

src1 = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
src2 = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
src3 = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

col = src1 + src2

col.set_children_styles(magnetization_color_south="blue")

magpy.show(col, src3, backend="plotly")
```

## Global style override

It is also possible to hand style input to the `show` function directly and globally override the given properties for this specific `show` output. Default or individual style attributes will not be modified. Such inputs must start with the `style` prefix and object family specifier must be omitted. Naturally underscore magic is supported.

The following example demonstrates the global style override

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset()  # omit when running as a standalone script

src1 = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
src2 = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
src3 = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

# use global style override
magpy.show(src1, src2, src3, backend="plotly", style_magnetization_show=False)

# back to default styles
magpy.show(src1, src2, src3, backend="plotly")
```

(examples-list-of-styles)=

## List of styles

```{code-cell} ipython3
magpy.defaults.display.style.as_dict(flatten=True, separator='.')
```
