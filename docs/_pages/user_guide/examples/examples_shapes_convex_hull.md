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

(examples-shapes-convex-hull)=

# Convex Hull

In geometry the convex hull of a point cloud is the smallest convex shape that contains all points, see [Wikipedia](https://en.wikipedia.org/wiki/Convex_hull).

Magpylib offers construction of convex hull magnets by combining the `magpylib.magnets.TriangularMesh` and the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) classes via the class method `from_ConvexHull`. Note, that the Scipy method does not guarantee correct face orientations if `reorient_faces` is disabled.

## Pyramid magnet

This is the fastest way to construct a pyramid magnet.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Create pyramid magnet
points = np.array([(-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0), (0, 0, 3)]) / 100
tmesh_pyramid = magpy.magnet.TriangularMesh.from_ConvexHull(
    polarization=(0, 0, 1),
    points=points,
    style_label="Pyramid Magnet",
)

# Display graphically
tmesh_pyramid.show(backend="plotly")
```

```{code-cell} ipython3

```
