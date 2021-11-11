---
jupytext:
  formats: ipynb,md:myst
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

+++ {"tags": []}

# This is a test example with magpylib.display

```{code-cell} ipython3
import magpylib as magpy


cuboid = magpy.magnet.Cuboid(magnetization=(1,0,0), dimension=(8, 4 ,6), position=(0,0,0))
cylinder = magpy.magnet.CylinderSegment(dimension=(6,10,4,0,90), position=(15,0,15), magnetization=(1,0,0))\
    .rotate_from_angax(axis=(0,0,1), angle= 45),

col = magpy.Collection(cuboid, cylinder)
magpy.defaults.reset()
magpy.defaults.display.backend = 'matplotlib'
#magpy.defaults.display.style.magnet.magnetization.show = False
cuboid.style.magnetization.show = True
col.set_styles(
    magnetization_show=True,
    magnetization_size=1,
)
magpy.display(
    col,
    #style_magnetization_show=True,
    #style_magnetization_size=1,
    backend='plotly',

)
```

```{note}
MyST markdown is a mixture of two flavors of markdown
```

## Write your first markdown document

Now that you've enabled the `myst-parser` in Sphinx, you can write MyST markdown in a file that ends with `.md` extension for your pages.

It supports all the syntax of **[CommonMark Markdown](https://commonmark.org/)** at its
base. This is a community standard flavor of markdown used across many projects.

In addition, it includes **[several extensions](../syntax/syntax.md) to CommonMark**.
These add extra syntax features for technical writing, such as the roles and directives used by Sphinx.
