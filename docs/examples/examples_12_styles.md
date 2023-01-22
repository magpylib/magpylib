---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(examples-graphic-styles)=
# Graphic - Styles

The graphic styles define how Magpylib objects are displayed visually when calling `show`. They can be fine-tuned and individualized in many ways.

There are multiple hierarchy levels that decide about the final graphical representation of the objects:

1. When no input is given, the **default style** will be applied.
2. Collections will override the color property of all children with their own color.
3. Object **individual styles** will take precedence over these values.
4. Setting a **local style** in `show()` will take precedence over all other settings.

## Setting the default style

The default style is stored in `magpylib.defaults.display.style`. Default styles can be set as properties,

```python
magpy.defaults.display.style.magnet.magnetization.show = True
magpy.defaults.display.style.magnet.magnetization.color.middle = 'grey'
magpy.defaults.display.style.magnet.magnetization.color.mode = 'bicolor'
```

by assigning a style dictionary with equivalent keys,

```python
magpy.defaults.display.style.magnet = {
    'magnetization': {'show': True, 'color': {'middle': 'grey', 'mode': 'tricolor'}}
}
```

or by making use of the `update` method:

```python
magpy.defaults.display.style.magnet.magnetization.update(
    'show': True,
    'color': {'middle'='grey', mode='tricolor',}
)
```

All three examples result in the same default style.

Once modified, the library default can always be restored with the `magpylib.style.reset()` method. The following practical example demonstrates how to create and set a user defined magnetization style as default,

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

print('Default magnetization style')
magpy.show(cube, cylinder, sphere, backend="plotly")

user_defined_style = {
    'show': True,
    'color': {
        'transition': 1,
        'mode': 'tricolor',
        'middle': 'white',
        'north': 'magenta',
        'south': 'turquoise',
    },
}
magpy.defaults.display.style.magnet.magnetization = user_defined_style

print('Custom magnetization style')
magpy.show(cube, cylinder, sphere, backend="plotly")
```

## Magic underscore notation
<!-- +++ {"tags": [], "jp-MarkdownHeadingCollapsed": true} -->

To facilitate working with deeply nested properties, all style constructors and object style methods support the magic underscore notation. It enables referencing nested properties by joining together multiple property names with underscores. This feature mainly helps reduce the code verbosity and is heavily inspired by the `plotly` implementation (see [plotly underscore notation](https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation)).

With magic underscore notation, the previous examples can be written as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    'magnetization_show': True,
    'magnetization_color_middle': 'grey',
    'magnetization_color_mode': 'tricolor',
}
```

or directly as kwargs in the `update` method as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.update(
    magnetization_show=True,
    magnetization_color_middle='grey',
    magnetization_color_mode='tricolor',
)
```

## Setting individual styles

Any Magpylib object can have its own individual style that will take precedence over the default values when `show` is called. When setting individual styles, the object family specifier such as `magnet` or `current` which is required for the defaults settings, but is implicitly defined by the object type, can be omitted.

```{warning}
Users should be aware that specifying individual style attributes massively increases object initializing time (from <50 to 100-500 $\mu$s).
While this may not be noticeable for a small number of objects, it is best to avoid setting styles until it is plotting time.
```

In the following example the individual style of `cube` is set at initialization, the style of `cylinder` is the default one, and the individual style of `sphere` is set using the object style properties.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

magpy.defaults.reset() # reset defaults defined in previous example

cube = Cuboid(
    magnetization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_magnetization_color_mode='tricycle',
)
cylinder = Cylinder(
    magnetization=(0, 1, 0),
    dimension=(1, 1), position=(2,0,0),
)
sphere = Sphere(
    magnetization=(0, 1, 1),
    diameter=1,
    position=(4,0,0),
)

sphere.style.magnetization.color.mode='bicolor'

magpy.show(cube, cylinder, sphere, backend="plotly")
```

## Setting style via collections

When displaying collections, the collection object `color` property will be automatically assigned to all its children and override the default style. An example that demonstrates this is {ref}`examples-union-operation`. In addition, it is possible to modify the individual style properties of all children with the `set_children_styles` method. Non-matching properties are simply ignored.

In the following example we show how the french magnetization style is applied to all children in a collection,

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

magpy.defaults.reset() # reset defaults defined in previous example

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

coll = cube + cylinder

coll.set_children_styles(magnetization_color_south="blue")

magpy.show(coll, sphere, backend="plotly")
```

## Local style override

Finally it is possible to hand style input to the `show` function directly and locally override the given properties for this specific `show` output. Default or individual style attributes will not be modified. Such inputs must start with the `style` prefix and the object family specifier must be omitted. Naturally underscore magic is supported.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

# use local style override
magpy.show(cube, cylinder, sphere, backend="plotly", style_magnetization_show=False)
```

(examples-list-of-styles)=

## List of styles

```{code-cell} ipython3
magpy.defaults.display.style.as_dict(flatten=True, separator='.')
```
