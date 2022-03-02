% magpylib documentation master file, created by
% sphinx-quickstart on Tue Feb 26 11:58:33 2019.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# What is Magpylib ?

- Python package for calculating 3D static magnetic fields of magnets, currents and other sources.
- The fields are computed using analytical solutions (very fast computations, simple geometries and superpositions thereof).
- The field computation is coupled to a geometry interface (position, orientation, paths) which makes it convenient to model relative positioning between sources and observers.

```{image} _static/images/index/source_fundamentals.png
:alt: source fundamentals
:align: center
```
# Quickstart

**Install Magpylib** with pip or conda:

```bash
pip install magpylib
```
or

```bash
conda install magpylib
```

The following **Example code** calculates the magnetic field of a cylindrical magnet:

```python
import magpylib as magpy
src = magpy.magnet.Cylinder(
    magnetization=(0,0,350),
    dimension=(4,5),
    position=(1,2,3))
obs = (4,4,4)
B = src.getB(obs)
print(B)

# out: [ 10.30092924   6.86728616 -20.96623472]
```

Here, a cylinder shaped permanent magnet with diameter/height of 4/5 millimeters is created in a global coordinate system with cylinder axis parallel to the z-axis and geometric magnet center at position (1,2,3). The magnetization is homogeneous and points in z-direction with an amplitude of 350 millitesla (=$\mu_0\times M$). The B-field is calculated in units of millitesla at the observer position (4,4,4).

```{toctree}
:caption: 'Content:'
:glob: true
:maxdepth: 1

_pages/*
```

```{toctree}
:caption: Example galleries
:glob: true
:maxdepth: 2

examples/01_fundamentals.md
examples/02_graphic_output.md
examples/03_advanced_features.md
examples/04_application_examples.md

```

```{toctree}
:caption: 'Library Docstrings:'
:glob: true
:maxdepth: 1

_autogen/magpylib
```

```{toctree}
:caption: 'Changelog:'
:glob: true
:maxdepth: 2

_changelog.md
```

# Index

- {ref}`genindex`
- {ref}`modindex`

```bash

```
