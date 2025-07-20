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

(examples-misc-image-method)=

# Method of Images

The method of images is a classical technique from electrostatics that can also be used in magnetostatics to obtain analytical solutions in the presence of highly permeable materials. In this example we show how it can be applied with Magpylib.

## Textbook Fundamentals

For the correct application of the method of images several requirements must be satisfied:

1. **Linearity of the Governing Equation:** The scalar potential $ \phi $ must satisfy a linear partial differential equation such as the Poisson equation:

   $$\Delta \phi(\vec{r}) = -\frac{\rho(\vec{r})}{\varepsilon}$$

   where $ \Delta = \nabla \cdot \nabla $ is the Laplace operator, $ \varepsilon $ is a constant permittivity, and $ \rho(\vec{r}) $ is the source distribution (=charges) that generates the field.

2. **Homogeneity in the Region of Interest:** The method of images is only applicable in charge-free regions, where the governing equation reduces to Laplace’s equation:
   
   $$\Delta \phi(\vec{r}) = 0$$

3. **Simple Geometry and Boundary Conditions:** The boundary must be geometrically simple -- typically a plane, sphere, or infinite cylinder. On the surface a simple boundary condition, like constant potential, must hold, and natural boundaries must be satisfied at infinity. This implies that the resulting gradient field $ \vec{F} = -\nabla \phi $ is normal to the boundary surface, and decays at infinity.

4. **Applicability of the Uniqueness Theorem:** The method relies on the uniqueness theorem: if a solution satisfies both the differential equation and the boundary conditions, it is the only physically valid solution in that region.

When all of these criteria are satisfied, the potential field in the region of interest can be constructed as the superposition of the original source distribution $ \rho(\vec{r}) $ and an image source $ \rho^*(\vec{r}) $, which is a mirror reflection of the original source across the boundary.

## Method of Images in Magnetostatics

In magnetostatics, the method of images can be applied in the presence of highly permeable ($\mu_r \gg 1$) soft magnetic materials. In such materials, the magnetic moments can rotate freely and align in a way that completely suppresses the $ \vec{H} $-field inside the body. As a result, the field is forced to be normal to the material’s surface, making simple planar boundaries ideal for image constructions and satisfying condition (3).


Furthermore:

- In magnetostatics, the magnetic field $ \vec{H} $ can be expressed as the gradient of a magnetic scalar potential: $\vec{H} = -\nabla \phi_m $ satisfying condition (1), provided we are in regions without free currents.

- Uniformly magnetized bodies, like the permanent magnets implemented in Magpylib, can be modeled using [magnetic surface charges](examples-misc-equivalent). This means that condition (2) is fulfilled everywhere except on these surfaces.

- The [uniqueness theorem](https://en.wikipedia.org/wiki/Electromagnetism_uniqueness_theorem) applies to all well-posed magnetostatic problems, satisfying condition (4).

---

This means that magnets and currents can be mirrored across a highly permeable surface to compute the magnetic field outside the material. For magnetization, the normal component is preserved and the tangential components are inverted. For currents, the tangential component is preserved, while the normal component is inverted, as illustrated in the image below.

```{figure} ../../../_static/images/examples_misc_mirror.png
:width: 100%
:align: center
:alt: Sketch of image method in magnetostatics.

Sketch of how the image method works in magnetostatics
```

**Note**: Do not confuse this with the case of [superconductors](https://www.imp.kiev.ua/~kord/wiki/method_of_images.html), where the method of images is also applicable — but for entirely different physical reasons (*perfect diamagnetism* rather than high permeability).

## Example Applications

The method of images does not require an idealized, infinitely permeable half-space. In practice, even thin (not too thin though), finite soft-magnetic layers—such as the soft-magnetic backs of inductors or magnetic shielding plates—can provide sufficient contrast for the image method. In addition, even relatively low permeabilities $\mu_r> 10$ or $\mu_r>100$ result in good agreement between experiment and theory.

One practical use case is shown in the example on [magnetic scale structures](examples-app-scales), such as pole wheels, where a soft-magnetic backing enhances the field of out-of-plane magnetized patterns.

Another application is the [computation of holding forces](examples-force-holding-force) between permanent magnets and soft magnetic surfaces.

## Example of the Magnetic Image Method

In the following example, a permanent magnet is placed above a planar magnetic shield located in the xy-plane. The mirror image is modeled as a second magnet, placed below the plane according to the method of images. We then compute the magnetic field in the upper half-space.

For field visualization, we use [PyVista streamlines](examples-vis-pv-streamlines). Notice how the field lines are perpendicular to the mirror surface at the interface — consistent with the boundary conditions imposed by a highly permeable material.

```{code-cell} ipython3
:tags: [hide-input]

import pyvista as pv
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

# Magnet Parameters
height = 1
angle = 30

# Create a cuboid magnet above the mirror plane
magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1),
    dimension=(1, 1),
    position=(0, 0, height),
    orientation=R.from_euler('y', angle, degrees=True)
)

# Create mirror image of the magnet
mirror_magnet = magnet.copy(
    position=(0, 0, -1),  # Mirror position
    orientation=R.from_euler('y',-angle, degrees=True)  # Mirror orientation
)

# Create a 3D grid with Pyvista
grid = pv.ImageData(
    dimensions=(61, 61, 61),
    spacing=(0.1, 0.1, 0.1),
    origin=(-3, -3, 0),
)

# Compute B-field and add as data to grid
col = magnet + mirror_magnet
grid["B"] = col.getB(grid.points)*1000 # Convert to mT

# Compute the field lines
seed = pv.Disc(center=(0,0,0), inner=2, outer=3, r_res=1, c_res=12)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_step_length=0.02,
    integration_direction="backward",
)

# Create a Pyvista plotting scene
pl = pv.Plotter()

# Add magnet to scene - streamlines units are assumed to be meters
magpy.show(magnet, canvas=pl, units_length="m", backend="pyvista")

# Prepare legend parameters
legend_args = {
    "title": "B (mT)",
    "title_font_size": 20,
    "color": "black",
    "position_y": 0.25,
    "vertical": True,
}

# Add streamlines and legend to scene
pl.add_mesh(
    strl.tube(radius=0.02),
    cmap="bwr",
    scalar_bar_args=legend_args,
)

# Add mirror to the scene
disc = pv.Disc(center=(0, 0, 0), inner=0.0, outer=4, normal=(0, 0, 1), r_res=1, c_res=100)
pl.add_mesh(disc, color='lightblue', show_edges=False)

# Prepare and show scene
pl.camera.position = (5,15, 10)
pl.show()
```






