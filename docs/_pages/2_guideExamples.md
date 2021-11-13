(examples)=

# Example Codes

This section includes a few code examples that show how the library can be used and what it can be used for. Detailed package, class, method and function documentations are found in the library docstrings {ref}`genindex`.

Technical details are outlined in {ref}`docu`.

## Contents

- {ref}`examples-simple`
- {ref}`examples-basic`
- {ref}`examples-coil`

(examples-simple)=

## Just compute the field

The most fundamental functionality of the library - compute the field (B in \[mT\], H in \[kA/m\]) of a source (here Cylinder magnet) at the observer position (1,2,3).

```python
from magpylib.magnet import Cylinder
src = Cylinder(magnetization=(222,333,444), dimension=(2,2))
B = src.getB((1,2,3))
print(B)
# Output: [-2.74825633  9.77282601 21.43280135]
```

(examples-basic)=

## Basic functionality

In this general example two source objects (magnet and current) are created, moved and rotated. The system geometry before and after move/rotate is displayed together with the magnetic field in the xz-plane. Notice that xz is a symmetry plane where the field has no y-component.

```{eval-rst}
.. plot:: _codes/examples_basic.py
    :include-source:

```

(examples-coil)=

## Modelling a Coil

A coil consists of large number of windings that can be modeled using `Loop` sources. The total coil is then a `Collection` of windings. One must be careful to take the line-current approximation into consideration. This means that the field diverges when approaching the current, while the field is correct outside a hypothetical wire with homogeneous current distribution.

```{eval-rst}
.. plot:: _codes/examples_coil.py
    :include-source:
```
