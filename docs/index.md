% magpylib documentation master file, created by
% sphinx-quickstart on Tue Feb 26 11:58:33 2019.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# What is Magpylib ?

Magpylib is a Python package for calculating **3D static magnetic fields** of magnets, line currents and other sources. The computation is based on analytical expressions and therefore **extremely fast**. A **user friendly geometry API** enables convenient relative positioning between sources and observers.

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
:caption: FUNCTIONALITY
:glob: true
:maxdepth: 2

examples/examples_02_paths.md
examples/examples_05_backend_canvas.md
examples/examples_12_styles.md
examples/examples_15_animation.md
examples/examples_13_3d_models.md
examples/examples_03_collections.md
examples/examples_21_compound.md
examples/examples_01_complex_forms.md
examples/examples_06_triangle.md
examples/examples_04_custom_source.md
```

```{toctree}
:caption: EXAMPLE GALLERIES
:glob: true
:maxdepth: 2

examples/examples_30_coil_field_lines.md
examples/examples_22_field_interpolation.md
examples/examples_31_end_of_shaft.md
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

--------------------------

# Index

- {ref}`genindex`
- {ref}`modindex`
