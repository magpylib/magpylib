
> [!WARNING]
> Version 5 introduces critical breaking changes with, among others, the _move to SI units_. We recommended to pin your dependencies to `magpylib>=4.5<5` until you are ready to migrate to the latest version! ([see details](https://github.com/magpylib/magpylib/discussions/647))

<p align="left"><img align="center" src=docs/_static/images/magpylib_flag.png width=35%>
</p>

---

<div>
<a href="https://opensource.org/licenses/BSD-2-Clause"> <img src="https://img.shields.io/badge/License-BSD_2--Clause-orange.svg">
</a>
<a href="https://github.com/magpylib/magpylib/actions/workflows/python-app.yml"> <img src="https://github.com/magpylib/magpylib/actions/workflows/python-app.yml/badge.svg">
</a>
<a href="https://magpylib.readthedocs.io/en/latest/"> <img src="https://readthedocs.org/projects/magpylib/badge/?version=latest">
</a>
<a href="https://codecov.io/gh/magpylib/magpylib"> <img src="https://codecov.io/gh/magpylib/magpylib/branch/main/graph/badge.svg">
</a>
<a href="https://pypi.org/project/magpylib/"> <img src="https://badge.fury.io/py/magpylib.svg" alt="PyPI version" height="18">
</a>
<a href="https://anaconda.org/conda-forge/magpylib"> <img src="https://anaconda.org/conda-forge/magpylib/badges/version.svg" alt="Conda Cloud" height="18">
</a>
<a href="https://mybinder.org/v2/gh/magpylib/magpylib/5.1.0?filepath=docs%2Fexamples"> <img src="https://mybinder.org/badge_logo.svg" alt="MyBinder link" height="18">
</a>
<a href="https://github.com/psf/black"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black" height="18">
</a>
</div>

Magpylib is an **open-source Python package** for calculating static **magnetic fields** of magnets, currents, and other sources. It uses **analytical expressions**, solutions to macroscopic magnetostatic problems, implemented in **vectorized** form which makes the computation **extremely fast** and leverages the open-source Python ecosystem for spectacular visualizations!

# Installation

Install from PyPI using **pip**
```
pip install magpylib
```
Install from conda forge using **conda**
```
conda install -c conda-forge magpylib
```
Magpylib supports _Python3.10+_ and relies on common scientific computation libraries _NumPy_, _Scipy_, _Matplotlib_ and _Plotly_. Optionally, _Pyvista_ is recommended as graphical backend.

# Resources

 - Check out our **[Documentation](https://magpylib.readthedocs.io/en/stable)** for detailed information about the last stable release, or the **[Dev Docs](https://magpylib.readthedocs.io/en/latest)** to see the unreleased development version features.
 - Please abide by our **[Code of Conduct](https://github.com/magpylib/magpylib/blob/main/CODE_OF_CONDUCT.md)**.
 - Contribute through **[Discussions](https://github.com/magpylib/magpylib/discussions)** and coding by following the **[Contribution Guide](https://github.com/magpylib/magpylib/blob/main/CONTRIBUTING.md)**. The Git project **[Issues](https://github.com/magpylib/magpylib/issues)** give an up-to-date list of potential enhancements and planned milestones. Propose new ones.
 - A **[Youtube video](https://www.youtube.com/watch?v=LeUx6cM1vcs)** introduction to Magpylib v4.0.0 within the **[GSC network](https://www.internationalcollaboration.org/).**
- An **[open-access paper](https://www.sciencedirect.com/science/article/pii/S2352711020300170)** from the year 2020 describes v2 of this library with most basic concepts still intact in later versions.

# Quickstart

Here is an example on how to use Magpylib.

```python
import magpylib as magpy

# Create a Cuboid magnet with sides 1,2 and 3 cm respectively, and a polarization
# of 1000 mT pointing in x-direction.
cube = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),  # in SI Units (T)
    dimension=(0.01, 0.02, 0.03),  # in SI Units (m)
)

# By default, the magnet position is (0,0,0) and its orientation is the unit
# rotation (given by a scipy rotation object), which corresponds to magnet sided
# parallel to global coordinate axes.
print(cube.position)  # --> [0. 0. 0.]
print(cube.orientation.as_rotvec())  # --> [0. 0. 0.]

# Manipulate object position and orientation through the respective attributes,
# or by using the powerful `move` and `rotate` methods.
cube.move((0, 0, -0.02))# in SI Units (m)
cube.rotate_from_angax(angle=45, axis="z")
print(cube.position)  # --> [0. 0. -0.02]
print(cube.orientation.as_rotvec(degrees=True))  # --> [0. 0. 45.]

# Compute the magnetic B-field in units of T at a set of observer positions. Magpylib
# makes use of vectorized computation. Hand over all field computation instances,
# e.g. different observer positions, at one function call. Avoid Python loops !!!
observers = [(0, 0, 0), (0.01, 0, 0), (0.02, 0, 0)]  # in SI Units (m)
B = magpy.getB(cube, observers)
print(B.round(2))  # --> [[-0.09 -0.09  0.  ]
#                         [ 0.   -0.04  0.08]
#                         [ 0.02 -0.01  0.03]]  # in SI Units (T)

# Sensors are observer objects that can have their own position and orientation.
# Compute the H-field in units of A/m.
sensor = magpy.Sensor(position=(0, 0, 0))
sensor.rotate_from_angax(angle=45, axis=(1, 1, 1))
H = magpy.getH(cube, sensor)
print(H.round())  # --> [-94537. -35642. -14085.]  # in SI Units (A/m)

# Position and orientation attributes of Magpylib objects can be vectors of
# multiple positions/orientations referred to as "paths". When computing the
# magnetic field of an object with a path, it is computed at every path index.
cube.position = [(0, 0, -.02), (1, 0, -.02), (2, 0, -.02)]  # in SI Units (m)
B = cube.getB(sensor)
print(B.round(2))  # --> [[-0.12 -0.04 -0.02]
#                         [ 0.   -0.    0.  ]
#                         [ 0.   -0.    0.  ]] # in SI Units (T)

# When several objects are involved and things are getting complex, make use of
# the `show` function to view your system through Matplotlib, Plotly or Pyvista backends.
magpy.show(cube, sensor, backend="pyvista")
```

More details and other important features are described in detail in the **[Documentation](https://magpylib.readthedocs.io/en/stable)**. Key features are:

- **Collections**: Group multiple objects for common manipulation
- **Complex shapes**: Create magnets with arbitrary shapes
- **Graphics**: Styling options, graphic backends, animations, and 3D models
- **CustomSource**: Integrate your own field implementation
- **Direct interface**: Bypass the object oriented interface (max speed)

# How can I cite this library ?

We would be happy if you give us credit for our efforts. A valid bibtex entry for the [2020 open-access paper](https://www.sciencedirect.com/science/article/pii/S2352711020300170) would be

```
@article{ortner2020magpylib,
  title={Magpylib: A free Python package for magnetic field computation},
  author={Ortner, Michael and Bandeira, Lucas Gabriel Coliado},
  journal={SoftwareX},
  volume={11},
  pages={100466},
  year={2020},
  publisher={Elsevier}
}
```

A valid software citation could be

```
@software{magpylib,
    author = {{Michael-Ortner et al.}},
    title = {magpylib},
    url = {https://magpylib.readthedocs.io/en/latest/},
    version = {5.1.0},
    date = {2023-06-25},
}
```
