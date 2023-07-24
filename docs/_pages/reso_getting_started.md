(getting-started)=

# Getting Started

## Installation and Dependencies

Magpylib supports *Python3.8+* and relies on common scientific computation libraries *Numpy*, *Scipy*, *Matplotlib* and *Plotly*. Optionally, *Pyvista* is recommended as graphical backend.

::::{grid} 2
:::{grid-item-card} Install with pip:
:text-align: center
:shadow: none
```console
pip install magpylib
```
:::
:::{grid-item-card} Install with conda:
:text-align: center
:shadow: none
```console
conda install -c conda-forge magpylib
```
:::
::::

## Magpylib fundamentals

Learn the fundamentals in 5 minutes. This requires a basic understanding of the Python programming language, the [Numpy array class](https://numpy.org/doc/stable/) and the [Scipy Rotation class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html).

### Step 1: Create sources and observers as Python objects

```python
import magpylib as magpy

# Create a Cuboid magnet with magnetization (polarization) of 1000 mT pointing
# in x-direction and sides of 1,2 and 3 mm respectively.

cube = magpy.magnet.Cuboid(magnetization=(1000,0,0), dimension=(1,2,3))

# Create a Sensor that measures the field at its position

sensor = magpy.Sensor()
```

### Step2: Manipulate object position and orientation

```python
# By default, the position of a Magpylib object is (0,0,0) and its orientation is the unit
# rotation, given by a scipy rotation object with unit rotation.

print(cube.position)                   # -> [0. 0. 0.]
print(cube.orientation.as_rotvec())    # -> [0. 0. 0.]

# Manipulate object position and orientation through the respective attributes:

from scipy.spatial.transform import Rotation as R
cube.position = (1,0,0)
cube.orientation = R.from_rotvec((0,0,45), degrees=True)

print(cube.position)                            # -> [1. 0.  0.]
print(cube.orientation.as_rotvec(degrees=True)) # -> [0. 0. 45.]

# Apply relative motion with the powerful `move` and `rotate` methods.

sensor.move((-1,0,0))
sensor.rotate_from_angax(angle=-45, axis='z')

print(sensor.position)                            # -> [-1.  0.  0.]
print(sensor.orientation.as_rotvec(degrees=True)) # -> [ 0.  0. -45.]
```

### Step 3: View your system

```python
# Make use of the `show` function to view your system through Matplotlib, Plotly
# or Pyvista backends.

magpy.show(cube, sensor, backend='plotly')
```

<img src="/_static/images/getting_started_fundamentals1.png" width=50%>

### Step 4: Compute the magnetic field

```python
# Compute the B-field in units of mT at a set of points.

points = [(0,0,-1), (0,0,0), (0,0,1)]
B = magpy.getB(cube, points)
print(B.round()) # -> [[263.  68.  81.]
                 #     [276.  52.   0.]
                 #     [263.  68. -81.]]

# Compute the H-field at a sensor in units of kA/m.

H = magpy.getH(cube, sensor)
print(H.round()) # -> [220.  41.   0.]
```

```{warning}
Magpylib makes use of vectorized computation (massive speedup). This requires that you hand over all field computation instances (multiple observers, multiple observer positions, multiple sources) at the same time. Avoid Python loops !!!
```

Other important features include

* **Paths**: Assign multiple positions/orientations to an object
* **Collections**: Group multiple objects for common manipulation
* **Complex magnet shapes**: Create magnets with arbitrary shapes
* **Graphics**: Styling options, graphic backends, animations, and 3D models
* **CustomSource**: Integrate your own field implementation
* **Direct interface**: Bypass the object oriented interface for maximal performance
