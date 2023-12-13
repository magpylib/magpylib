
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
<a href="https://mybinder.org/v2/gh/magpylib/magpylib/4.5.0?filepath=docs%2Fexamples"> <img src="https://mybinder.org/badge_logo.svg" alt="MyBinder link" height="18">
</a>
<a href="https://github.com/psf/black"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black" height="18">
</a>
</div>

Magpylib is a Python package for calculating **3D static magnetic fields** of magnets, line currents and other sources. The computation is based on explicit expressions and is therefore **extremely fast**. A **user friendly API** enables convenient positioning of sources and observers.

# Installation

Install from PyPI using **pip**
```
pip install magpylib
```
Install from conda forge using **conda**
```
conda install -c conda-forge magpylib
```
Magpylib supports _Python3.8+_ and relies on common scientific computation libraries _Numpy_, _Scipy_, _Matplotlib_ and _Plotly_. Optionally, _Pyvista_ is recommended as graphical backend.

# Ressources

 - Check out our **[Documentation](https://magpylib.readthedocs.io/en/latest)** for detailed information.
 - Please abide by our **[Code of Conduct](https://github.com/magpylib/magpylib/blob/main/CODE_OF_CONDUCT.md)**.
 - Contribute through **[Discussions](https://github.com/magpylib/magpylib/discussions)** and coding by following the **[Contribution Guide](https://github.com/magpylib/magpylib/blob/main/CONTRIBUTING.md)**. The Git project **[Issues](https://github.com/magpylib/magpylib/issues)** give an up-to-date list of potential enhancements and planned milestones. Propose new ones.
 - A **[Youtube video](https://www.youtube.com/watch?v=LeUx6cM1vcs)** introdution to Magpylib v4.0.0 within the **[GSC network](https://www.internationalcollaboration.org/).**
- An **[open-access paper](https://www.sciencedirect.com/science/article/pii/S2352711020300170)** from the year 2020 describes v2 of this library with most basic concepts still intact in later versions.

# Quickstart

Here is an example how to use Magpylib.

```python3
import magpylib as magpy

# Create a Cuboid magnet with sides 1,2 and 3 mm respectively, and magnetization
# (polarization) of 1000 mT pointing in x-direction.
cube = magpy.magnet.Cuboid(
  magnetization=(1000,0,0),
  dimension=(1,2,3),
)

# By default, the magnet position is (0,0,0) and its orientation is the unit
# rotation (given by a scipy rotation object), which corresponds to magnet sided
# parallel to global coordinate axes.
print(cube.position)                   # --> [0. 0. 0.]
print(cube.orientation.as_rotvec())    # --> [0. 0. 0.]

# Manipulate object position and orientation through the respective attributes,
# or by using the powerful `move` and `rotate` methods.
cube.move((0,0,-2))
cube.rotate_from_angax(angle=45, axis='z')
print(cube.position)                            # --> [0. 0. -2.]
print(cube.orientation.as_rotvec(degrees=True)) # --> [0. 0. 45.]

# Compute the magnetic field in units of mT at a set of observer positions. Magpylib
# makes use of vectorized computation. Hand over all field computation instances,
# e.g. different observer positions, at one funtion call. Avoid Python loops !!!
observers = [(0,0,0), (1,0,0), (2,0,0)]
B = magpy.getB(cube, observers)
print(B.round()) # --> [[-91. -91.   0.]
                 #      [  1. -38.  84.]
                 #      [ 18. -14.  26.]]

# Sensors are observer objects that can have their own position and orientation.
# Compute the H-field in units of kA/m.
sensor = magpy.Sensor(position=(0,0,0))
sensor.rotate_from_angax(angle=45, axis=(1,1,1))
H = magpy.getH(cube, sensor)
print(H.round()) # --> [-95. -36. -14.]

# Position and orientation attributes of Magpylib objects can be vectors of
# multiple positions/orientations refered to as "paths". When computing the
# magnetic field of an object with a path, it is computed at every path index.
cube.position = [(0,0,-2), (1,0,-2), (2,0,-2)]
B = cube.getB(sensor)
print(B.round()) # --> [[-119.  -45.  -18.]
                 #      [   8.  -73.  -55.]
                 #      [  15.  -30.   -8.]]

# When several objects are involved and things are getting complex, make use of
# the `show` function to view your system through Matplotlib, Plotly or Pyvista backends.
magpy.show(cube, sensor, backend='pyvista')
```

More details and other important features are described in detail in the **[Documentation](https://magpylib.readthedocs.io/en/latest)**. Key features are:

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
    version = {4.5.0},
    date = {2023-06-25},
}
```
