% magpylib documentation master file, created by
% sphinx-quickstart on Tue Feb 26 11:58:33 2019.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# What is Magpylib ?

Magpylib is a Python package for calculating **3D static magnetic fields** of magnets, line currents and other sources. The computation is based on analytical expressions and therefore **extremely fast**. A **user friendly geometry interface** enables convenient relative positioning between sources and observers.

```{image} _static/images/index/source_fundamentals.png
:alt: source fundamentals
:align: center
```

--------

# Quickstart

Magpylib is on PyPI and conda-forge. **Install using pip** (`pip install magpylib`) or **conda** (`conda install magpylib`) package managers.

The following **Example code** outlines the core functionality:

```python
import magpylib as magpy
source = magpy.magnet.Cylinder(magnetization=(0,0,350), dimension=(4,5), position=(1,2,3))
observer = (4,4,4)
B = source.getB(observer)
print(B)

# out: [ 10.30092924   6.86728616 -20.96623472]
```

Here, a cylinder shaped permanent magnet with (diameter, height) of (4, 5) millimeters is created in a global coordinate system at position (1,2,3). The magnetization is homogeneous and points in z-direction with an amplitude of 350 millitesla (=$\mu_0\times M$). The B-field is computed at the observer position (4,4,4) and returned in units of millitesla.

--------

```{toctree}
:caption: CONTENT
:glob: true
:maxdepth: 1

_pages/*
```

```{toctree}
:caption: EXAMPLE GALLERIES
:glob: true
:maxdepth: 2

examples/01_fundamentals.md
examples/02_graphic_output.md
examples/03_advanced.md
examples/04_application_examples.md
```

```{toctree}
:caption: LIBRARY DOCSTRINGS
:glob: true
:maxdepth: 1

_autogen/magpylib
```

```{toctree}
:caption: CHANGELOG
:glob: true
:maxdepth: 2

_changelog.md
```

# Index

- {ref}`genindex`
- {ref}`modindex`

```bash

```
