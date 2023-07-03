
<p align="left"><img align="center" src=docs/_static/images/magpylib_flag.png width=35%>
</p>

---
<div>
<a href="https://dev.azure.com/magpylib/magpylib/_build/latest?definitionId=1&branchName=main"> <img src="https://dev.azure.com/magpylib/magpylib/_apis/build/status/magpylib.magpylib?branchName=main">
</a>
<a href="https://circleci.com/gh/magpylib/magpylib"> <img src="https://circleci.com/gh/magpylib/magpylib.svg?style=svg">
</a>
<a href="https://magpylib.readthedocs.io/en/latest/"> <img src="https://readthedocs.org/projects/magpylib/badge/?version=latest">
</a>
<a href="https://opensource.org/licenses/BSD-2-Clause"> <img src="https://img.shields.io/badge/License-BSD_2--Clause-orange.svg">
</a>
<a href="https://codecov.io/gh/magpylib/magpylib"> <img src="https://codecov.io/gh/magpylib/magpylib/branch/main/graph/badge.svg">
</a>
<a href="https://pypi.org/project/magpylib/"> <img src="https://badge.fury.io/py/magpylib.svg" alt="PyPI version" height="18">
</a>
<a href="https://anaconda.org/conda-forge/magpylib"> <img src="https://anaconda.org/conda-forge/magpylib/badges/version.svg" alt="Conda Cloud" height="18">
</a>
<a href="https://mybinder.org/v2/gh/magpylib/magpylib/4.3.0?filepath=docs%2Fexamples"> <img src="https://mybinder.org/badge_logo.svg" alt="MyBinder link" height="18">
</a>
</div>

Magpylib is a Python package for calculating **3D static magnetic fields** of magnets, line currents and other sources. The computation is based on explicit expressions and is therefore **extremely fast**. A **user friendly API** enables convenient positioning of sources and observers.

# Installation & Ressources

Install from PyPI using **pip**
```
pip install magpylib
```
of from conda forge using  **conda**
```
conda install -c conda-forge magpylib
```
Magpylib supports _Python3.8+_ and relies on common scientific computation libraries _Numpy_, _Scipy_, _Matplotlib_ and _Plotly_. Optionally, _Pyvista_ is recommended as graphical backend.

Check out our **[Documentation](https://magpylib.readthedocs.io/en/latest)** for detailed information! Please abide by our **[Code of Conduct](https://github.com/magpylib/magpylib/blob/main/CODE_OF_CONDUCT.md)**. Contribute through **[Discussions](https://github.com/magpylib/magpylib/discussions)** and coding by following the **[Contribution Guide]()**. The Git project [Issues](https://github.com/magpylib/magpylib/issues) give an up-to-date list of potential enhancements. Propose new ones.

# Quickstart

Here is an example how to use Magpylib.

```python3
import magpylib as magpy

# Create a Cuboid magnet with sides 1,2 and 3 mm and magnetization (polarization)
# of 1000 mT pointing in x-direction.
cube = magpy.magnet.Cuboid(
  magnetization=(1000,0,0),
  dimension=(1,2,3),
)

# By default, the magnet position is (0,0,0) and its orientation is the unit
# rotation (given by a scipy rotation object), which corresponds to magnet sided
# parallel to global coordinate axes.
print(cube.position)                   # --> [0. 0. 0.]
print(cube.orientation.as_rotvec())    # --> [0. 0. 0.]

# Manipulate object position and rotation using the powerful `move` and `rotate`
# methods.
cube.move((0,0,-2))
cube.rotate_from_angax(angle=45, axis='z')
print(cube.position)                            # --> [0. 0. -2.]
print(cube.orientation.as_rotvec(degrees=True)) # --> [0. 0. 45.]

# Compute the magnetic field at a set of observer positions. Magpylib makes use
# of vectorized computation. This means that the field computation should not be
# used in a loop, but all instances (e.g. different observer positions) should be
# handed over at one function call, if possible.
observers = [(0,0,0), (1,0,0), (2,0,0)]
B = magpy.getB(cube, observers)
print(B.round()) # --> [[-91. -91.   0.]
                 #      [  1. -38.  84.]
                 #      [ 18. -14.  26.]]

# Sensors are observer objects that can have their own position and orientation.
sensor = magpy.Sensor(position=(0,0,-2))
sensor.rotate_from_angax(angle=45, axis=(1,1,1))
H = magpy.getH(cube, sensor)
print(H.round()) # --> [-476. -179.  -71.]

# Position and orientation attributes of magpylib objects can be vectors of
# multiple positions/orientations that we then refer to as paths. When computing
# the magnetic field of an object with a path, it is simultaneously computeted at
# every path index using vectorization.
cube.position = [(-1,0,0), (0,0,0), (1,0,0)]
B = cube.getB(sensor)
print(B.round()) # --> [[   8.  -73.  -55.]
                 #      [-119.  -45.  -18.]
                 #      [ -44.   11.   80.]]

# When several objects are involved and things are getting complex, make use of
# the show function to view your system through Matplotlib, Plotly or Pyvista backends.
magpy.show(cube, sensor, backend='pyvista')
```

Other important features one should be aware of:

- **Collections**: Group multiple objects for common manipulation
- **Complex shapes**: Create magnets with arbitrary shapes
- **Graphics**: Styling options, graphic backends, animations, and 3D models
- **CustomSource**: Integrate your own field implementation
- **Direct interface**: Circumvent the object oriented interface (max speed)

# How can I cite this library ?

An [open-access paper](https://www.sciencedirect.com/science/article/pii/S2352711020300170) from 2020 describes v2 of this library. A valid bibtex entry would be.

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

# Links

The **[Github project](https://github.com/magpylib/magpylib)** where everything comes together.

The **[official documentation](https://magpylib.readthedocs.io/en/latest/)** on read the docs.

A **[Youtube video](https://www.youtube.com/watch?v=LeUx6cM1vcs)** introdution to Magpylib v4.0.0 within the **[GSC network](https://www.internationalcollaboration.org/).**

