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
pip install magpylib .
```
or

```bash
conda install magpylib .
```

This **Example code** calculates the magnetic field of a cylindrical magnet.

```python
import magpylib as magpy
s = magpy.magnet.Cylinder(magnetization=(0,0,350), dimension=(4,5))
observer_pos = (4,4,4)
print(s.getB(observer_pos))

# Output: [ 5.08641867  5.08641867 -0.60532983]
```

A cylinder shaped permanent magnet with diameter and height of 4 and 5 millimeter, respectively, is created in a global coordinate system with cylinder axis parallel to the z-axis and geometric magnet center in the origin. The magnetization is homogeneous and points in z-direction with an amplitude of 350 millitesla. The magnetic field is calculated in units of millitesla at the observer position (4,4,4) in units of millimeter.

```{toctree}
:caption: 'Content:'
:glob: true
:maxdepth: 1

_pages/*
```

```{toctree}
:caption: 'Library Docstrings:'
:glob: true
:maxdepth: 1

_autogen/magpylib
```

```{toctree}
:caption: Example galleries
:glob: true
:maxdepth: 2

examples/gallery.md
```

# Index

- {ref}`genindex`
- {ref}`modindex`
