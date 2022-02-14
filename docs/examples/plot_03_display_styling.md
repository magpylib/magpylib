---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(display_styling_example)=
# Display styling

+++

The default displaying style may not yield by default the visual representation the user wants. For these cases, the library includes a variety of styling options that can be applied at multiple levels in the user's code.

```{warning}
Users should be aware that specifying style attributes increases initializing time. While this may not be noticeable for a small number objects, this can become an issue when initializating a lot of objects or repeatedly creating objects in a loop, for example in an optimization algorithm. In this cases you may want to specify objects styles not until plotting time.
```
+++ {"tags": []}

## Hierarchy of arguments

The styling options can be set at the library level, for an individual object directly or via a `Collection` and as an explicit argument in the `show` function. These settings, are ordered from **lowest** to **highest** precedence as follows:

- library `defaults`
- individual object `style` or at `Collection` level
- in the `show` function or method

+++ {"tags": []}

## Setting library defaults

+++ {"tags": []}

### Defaults style structure

General Magpylib defaults can be set from the top library level by settings `magpylib.defaults` properties and default display styling properties can be accessed with `magpylib.defaults.display.style`. The default styles are separated into the following object families:

- **base** : common properties for all families
- **magnet**: `Cuboid, Cylinder, Sphere, CylinderSegment`
- **current**: `Line, Loop`
- **sensor**: `Sensor`
- **markers**: markers in the `show` function

+++ {"tags": [], "jp-MarkdownHeadingCollapsed": true}

### Accessing nested properties

Nested properties can be set by accessing style attributes with the dot notation:
```python
import magpylib as magpy
magpy.defaults.display.style.magnet.magnetization.show = True
magpy.defaults.display.style.magnet.magnetization.color.middle = "grey"
magpy.defaults.display.style.magnet.magnetization.color.mode = "bicolor"
```

or by assigning a dictionary with equivalent keys:

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    "magnetization": {"show": True, "color": {"middle": "grey", "mode": "tricolor"}}
}
```

+++ {"tags": [], "jp-MarkdownHeadingCollapsed": true}

### Using the _magic underscore notation_
To make it easier to work with nested properties, style constructors and object style method support magic underscore notation. This allows you to reference nested properties by joining together multiple nested property names with underscores. This feature mainly helps reduce the code verbosity and is heavily inspired by the `plotly` implementation (see [plotly underscore notation](https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation)). The previous examples can therefore, also be written as:

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    "magnetization_show": True,
    "magnetization_color_middle": "grey",
    "magnetization_color_mode": "tricolor",
}
```
Additionally, instead of overriding the whole magnet style constructor, style properties can be updated. The underscore notation is also supported here:

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.update(
    magnetization_show=True,
    magnetization_color_middle="grey",
    magnetization_color_mode="tricolor",
)
```

+++

### Practical example

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset() # can be omitted if runing as a standalone script

cuboid = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1))
cylinder.move((2, 0, 0))
sphere = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1)
sphere.move((4, 0, 0))

coll = magpy.Collection(cuboid, cylinder, sphere)

print('Display before setting style defaults')
magpy.show(*coll, backend="plotly")

my_default_magnetization_style = {
    "show": True,
    "color": {
        "transition": 1,
        "mode": "tricolor",
        "middle": "white",
        "north": "magenta",
        "south": "turquoise",
    },
}
magpy.defaults.display.style.magnet.magnetization = my_default_magnetization_style

print('Display after setting style defaults')
magpy.show(*coll, backend="plotly")
```

```{note}
The defaults are only reset at the first magpylib library import call. By reruning the same cell in a jupyter notebook the `defaults.reset()` call may be need as the import will not trigger a reset on its own the second time.
```

+++

## Setting style via `Collection`

+++

The `Collection` object only holds base style properties and only has priority on the children when assigning a color via `style.color`. On the other hand, the `set_children_styles` helper method modifies the style of all its children directly, as long as the set argument(s) match existing child style properties. Non-matching properties are just ignored.

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset() # can be omitted if runing as a standalone script

cuboid = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1))
cylinder.move((2, 0, 0))
sphere = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1)
sphere.move((4, 0, 0))

coll = cuboid + cylinder + sphere

coll.set_children_styles(magnetization_color_south="blue")

magpy.show(coll, backend="plotly")
```

## Setting individual styles

+++

Style properties can also be set for any Magpylib object instance. Note that the object family which is required for the default styles is not present when setting invidual styles.

For example setting the default magnetization north color is set a the family level as:
```python
import magpylib as magpy
magpy.defaults.display.style.current.arrow.size = 2
```

whereas at the individual style level it becomes:

```python
import magpylib as magpy
loop = magpy.current.Loop(current=10, diameter=10)
loop.style.arrow.size = 2
```

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset() # can be omitted if runing as a standalone script

cuboid = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1))
cylinder.move((2, 0, 0))
sphere = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1)
sphere.move((4, 0, 0))

coll = cuboid + cylinder + sphere

#setting individual styles
cylinder.style.update(magnetization_color_mode="bicolor")
cuboid.style.magnetization.color = dict(mode="tricycle")
sphere.style.magnetization = {"color": {"mode":"tricolor"}}


magpy.show(*coll, backend="plotly")
```

+++ {"tags": []}

## Overriding style at display time

+++

The provided styling properties as function arguments will temporarily override the style properties set by any of the aforementioned methods. All styling properties need to start with `style` and underscore magic is supported. The object family must be omitted since the style properties set at display time will apply across object families. Only matching properties to a specific object
will be applied.

```{code-cell} ipython3
import magpylib as magpy

magpy.defaults.reset() # can be omitted if runing as a standalone script

cuboid = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=(1, 1))
cylinder.move((2, 0, 0))

coll = cuboid + cylinder

magpy.show(*coll, backend="plotly", style_magnetization_show=False)
```

```{note}
Setting style arguments in the `show` function does not change the default styles nor does it modify individual object styles,  only the current representation to be displayed is affected.
```

+++

In the following example, both `sensor` and `dipole` have a `size` object level style property that has be set explicitly at object creation.

```{code-cell} ipython3
import magpylib as magpy
import plotly.graph_objects as go

magpy.defaults.reset()

cuboid = magpy.magnet.Cuboid(
    magnetization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 0, 0)
)
sensor = magpy.Sensor(position=((-2, 0, 0)), style_size=1)
sensor2 = magpy.Sensor(position=((2, 0, 0)), style=dict(size=10))
dipole = magpy.misc.Dipole(moment=(1, 1, 1), position=(0, 0, 2), style=dict(size=2))

magpy.show(cuboid, dipole, sensor, sensor2, zoom=1)
```

```{note}
Note that in the previous example, the individual sensor style is set at object creation level, which also supports the magic underscore notation.
```

+++

The size property can be overridden at display time with `style_size=5`

```{code-cell} ipython3
import magpylib as magpy
import plotly.graph_objects as go

magpy.defaults.reset()

cuboid = magpy.magnet.Cuboid(
    magnetization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 0, 0)
)
sensor = magpy.Sensor(position=((-2, 0, 0)), style=dict(size=1))
sensor2 = magpy.Sensor(position=((2, 0, 0)), style=dict(size=10))
dipole = magpy.misc.Dipole(moment=(1, 1, 1), position=(0, 0, 2), style=dict(size=2))

magpy.show(cuboid, dipole, sensor, sensor2, style_size=1, zoom=0)
```

## List of available styles

```{code-cell} ipython3
magpy.defaults.display.style.as_dict(flatten=True)
```

```{warning}
Even if both `matplotlib` and `plotly` backends can display all object of the library, there
is no 100% feature parity between them. Some of the differences include (non-exhaustive list):

- `magnetization.size` -> `matplotlib` only
- `magnetization.color` -> `plotly` only
```
