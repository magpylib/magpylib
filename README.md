[![CircleCI](https://circleci.com/gh/OrtnerMichael/magPyLib.svg?style=svg)](https://circleci.com/gh/OrtnerMichael/magPyLib) 
[![](https://readthedocs.com/projects/magpylib-magpylib/badge/?version=sphinx-docs)](https://magpylib-magpylib.readthedocs-hosted.com/)

---

# About
*A simple and user friendly magnetic toolbox for Python 3.2+*

## What is magpylib ?
- Python package for calculating magnetic fields of magnets, currents and
  moments (sources).
- It provides convenient methods to generate, geometrically manipulate, group
  and vizualize assemblies of sources.
- The magnetic fields are determined from underlying (semi-analytical)
  solutions which results in fast computation times (sub-millisecond) and
  requires little computation power.

<p align="center">
    <img align='center' src="./docs/_static/images/index/sourceBasics.svg"></center>
</p>
---
### Dependencies: 
_Python3.2+_, _Numpy_, _Matplotlib_

### Installation:

- Clone this repository 
```bash
$ git clone https://github.com/magpylib/magpylib
```
- Create virtual environment:
```bash
$ conda create -n packCondaTest 
```
- Activate:

```bash
$ conda activate packCondaTest
```

- Install the generated library for the environment:
```bash
(packCondaTest) $ cd magpylib/
```

```bash
(packCondaTest) $ pip install .
```

The library is now in the packCondaTest environment.


---
### Example:

- Two permanent magnets with axial magnetization are created and geometrically manipulated. They are grouped in a single collection and the system geometry is displayed using a supplied method.
- The total magnetic field that is generated by the collection is calculated on a grid in the xz-plane and is displayed using matplotlib.

**Program output:**
![](./docs/_static/images/documentation/examplePlot.jpg)

**Code:**
```python
# imports
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
 
# create magnets
magnet1 = magpy.source.magnet.Box(mag=[0,0,600],dim=[3,3,3],pos=[-4,0,3])
magnet2 = magpy.source.magnet.Cylinder(mag=[0,0,500], dim=[3,5], pos=[0,0,0])

# manipulate magnets
magnet1.rotate(45,[0,1,0],anchor=[0,0,0])
magnet2.move([5,0,-4])

# collect magnets
pmc = magpy.Collection(magnet1,magnet2)

# display system geometry
pmc.displaySystem()

# calculate B-fields on a grid
xs = np.linspace(-10,10,20)
zs = np.linspace(-10,10,20)
Bs = np.array([[pmc.getB([x,0,z]) for x in xs] for z in zs])

# display fields using matplotlib
fig, ax = plt.subplots()
X,Y = np.meshgrid(xs,zs)
U,V = Bs[:,:,0], Bs[:,:,2]
ax.streamplot(X, Y, U, V, color=np.log(U**2+V**2), density=1.5)
plt.show() 
```
