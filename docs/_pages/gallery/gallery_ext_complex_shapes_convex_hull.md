---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(gallery-ext-complex-shapes-convex-hull)=

# Complex shapes from Convex Hull

## Pyramid magnet from ConvexHull

`TriangularMesh` objects are easily constructed from the convex hull of a given point cloud using the classmethod `from_ConvexHull`. This classmethod  makes use of the [scipy.spatial.ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) class. Note that the Scipy method does not guarantee correct face orientations if `reorient_faces` is disabled.

```{code-cell} ipython3
import magpylib as magpy

# create Pyramid
points = [[-2,-2, 0], [-2,2,0], [2,-2,0], [2,2,0], [0,0,3]]
tmesh_pyramid = magpy.magnet.TriangularMesh.from_ConvexHull(
    magnetization=(0, 0, 1000),
    points=points,
)

#display graphically
tmesh_pyramid.show(
    style_opacity=0.5,
    style_orientation_show=True,
    style_orientation_size=1.5,
)
```