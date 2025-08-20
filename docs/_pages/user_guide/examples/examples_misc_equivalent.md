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

(examples-misc-equivalent)=
# Equivalent Charge and Current Method

The magnetic field of a prism-shaped permanent magnet with uniform magnetization $\vec{M}$ oriented normal to its top and bottom surfaces can be computed using **replacement models** instead of directly integrating over the magnet’s volume.

Two commonly used approaches are:

- **Equivalent Surface Charge Method**:
  This method models the magnet as a pair of fictitious magnetic surface charges located on its top and bottom faces. By analogy with electrostatics, a constant magnetization $\vec{M}$ gives rise to a uniform surface magnetic charge density of $\sigma = \vec{M}\cdot \vec{n}$, where $\vec{n}$ is the outward-pointing normal vector of each side face. The resulting magnetic field is computed similarly to the electric field of charged surfaces.

- **Equivalent Current Method**:
  This method replaces the magnetization with equivalent bound surface currents flowing around the prism’s side faces. The surface current density is given by $\vec{j} = \vec{M} \times \vec{n}$. This is analogous to using the Biot–Savart law for current distributions.

We demonstrate and compare these modeling techniques by computing the magnetic field of a cube magnet using three representations: First we use the `Cuboid` solution offered by Magpylib with a magnetization of 10 kA/m.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import magpylib as magpy

# Create a cube magnet
cube = magpy.magnet.Cuboid(
    dimension=(2, 2, 1),
    magnetization=(0, 0, 1e4),
    style_legend_text='Magnet',
)

# Sensor for observing the field
sensor = magpy.Sensor(
    position=np.linspace((-2,0,2), (2,0,2), 100),
)

# Display magnetic field and objects
magpy.show(
    {"objects": [sensor, cube],   "col": 1},
    {"objects": [sensor, cube],   "output": "Hz", "col": 2},
    backend="plotly",
)
```

Then we construct a cube with the equivalent charge method by combining four `Triangle` surfaces to make up charged top and bottom of a cube. The surface charge density is 10 kA/m (Magpylib automatically projects the magnetization vector of `Triangle` onto the triangle plane).

```{code-cell} ipython3
:tags: [hide-input]

# Vertices of a Cuboid
p1 = (-1, -1, -.5)
p2 = (-1, -1, .5)
p3 = (1, -1, -.5)
p4 = (1, -1, .5)
p5 = (1, 1, -.5)
p6 = (1, 1, .5)
p7 = (-1, 1, -.5)
p8 = (-1, 1, .5)

# Create charged top and bottom faces of cube
charge = magpy.Collection(style_legend_text='Charge')
charge.add(
    magpy.misc.Triangle(vertices=(p1,p3,p5), magnetization=(0,0,-1e4)),
    magpy.misc.Triangle(vertices=(p1,p5,p7), magnetization=(0,0,-1e4)),
    magpy.misc.Triangle(vertices=(p2,p4,p6), magnetization=(0,0,1e4)),
    magpy.misc.Triangle(vertices=(p2,p6,p8), magnetization=(0,0,1e4)),
)

# Display objects and field
magpy.show(
    {"objects": [sensor, charge], "col": 1},
    {"objects": [sensor, charge], "output": "Hz", "col": 2},
    backend="plotly",
)
```

Finally we realize the equivalent current method creating a `TriangleStrip` current that flows along the edges of the cube.

```{code-cell} ipython3
:tags: [hide-input]
# Create a current strip where the current flows along the cube edges
strip = magpy.current.TriangleStrip(
    vertices=[p1, p2, p3, p4, p5, p6, p7, p8, p1, p2],
    current=1e4,
    style_legend_text='Current',
)

# Display objects and field
magpy.show(
    {"objects": [sensor, strip],  "col": 1},
    {"objects": [sensor, strip],  "output": "Hz", "col": 2},
    backend="plotly",
)
```

Notice that all three representations yield the same magnetic field, confirming the physical equivalence of the models.

It is also worth noting that many of the analytical expressions found in the literature—and those used in Magpylib, including the one for `Cuboid`—were originally derived using these replacement pictures.
