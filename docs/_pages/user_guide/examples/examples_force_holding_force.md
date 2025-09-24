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

(examples-force-holding-force)=
# Magnetic Holding Force

This example demonstrates how to compute the magnetic holding force experienced by a permanent magnet attached to a soft-ferromagnetic plate. The holding force represents the minimum pull force required to detach the magnet from the surface—a critical parameter for engineering applications involving magnetic fasteners, sensors, and actuators.

```{figure} ../../../_static/images/examples_force_holding_force.png
:width: 40%
:align: center
:alt: Sketch of holding force.

Sketch of holding force F that must be overcome to detach the magnet from a soft-magnetic plate.
```

**Computational approach:**

We calculate the holding force using the [method of images](examples-misc-image-method). The method states that the magnetic field of a dipole in front of a soft-magnetic surface is equivalent to the field produced by two dipoles: the original one and its mirror image, which mirrors both position and charge across the surface. For permanent magnets this means that the normal component of the magnetization remains the same, while the tangential component is flipped in the mirror image

The following code shows how to compute the holding force for a cubical magnet attached to a permeable surface.

```{code-cell} ipython3
:tags: [hide-input]

import magpylib as magpy

print('HOLDING FORCE COMPUTATION\n')

# Create magnet and mirror image
cube = magpy.magnet.Cuboid(
    dimension=(5e-3, 2.5e-3, 1e-3),
    polarization=(0, 0, 1.33),
    meshing = 100
)
mirror_image = cube.copy(position=(0, 0, 1e-3))

# Compute force
F,_ = magpy.getFT(mirror_image, cube)
print(f"Holding Force: {F[2]*100:.3e} g")
```

**Validation:**
The magnet dimensions and N45 material properties in this example are based on a commercial product from [Supermagnete](https://www.supermagnete.at/quadermagnete-neodym/quadermagnet-5mm-2.5-1.5mm_Q-05-2.5-1.5-HN). N45 neodymium material has a remanence between 1.32 and 1.36 T (see the ["Modeling a real magnet"](examples-tutorial-modeling-magnets) tutorial for details). Our computed value of 349 g matches the manufacturer's specification of approximately 350 g.

```{hint}
**Accuracy of this approach**: The method of images provides excellent results even for moderately permeable materials (μᵣ > 50) and finite plate thicknesses comparable to the magnet dimensions.
```
