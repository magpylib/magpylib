
(docu-position)=

# Position, Orientation and Paths


::::{grid} 2

:::{grid-item}
:columns: 7
The explicit magnetic field expressions, implemented in Magpylib, are generally described in beneficial coordinates of the sources. It is a common problem to transform the field into an application relevant lab coordinate system. While not technically difficult, such transformations are prone to error.

Here Magpylib helps out. All Magpylib objects lie in a global Cartesian coordinate system. Object position and orientation are defined by the attributes `position` and `orientation`, 😏. Objects can easily be moved around using the `move()` and `rotate()` methods. Eventually, the field is computed in the reference frame of the observers.
:::
:::{grid-item}
:columns: 5
![](../../_static/images/docu_position_sketch.png)
:::
::::

Position and orientation of all Magpylib objects are defind by the two attributes

::::{grid} 2
:::{grid-item-card}
:shadow: none
:columns: 5
<span style="color: orange">**position**</span> - a point $(x,y,z)$ in the global coordinates, or a set of such points $(P_1, P_2, ...)$. By default objects are created with `position=(0,0,0)`.
:::
:::{grid-item-card}
:shadow: none
:columns: 7
<span style="color: orange">**orientation**</span> - a [Scipy Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) which describes the object rotation relative to its default orientation (defined in {ref}`docu-classes`). By default, objects are created with unit rotation `orientation=None`.
:::
::::

The position and orientation attributes can be either **scalar**, i.e. a single position or a single rotation, or **vector**, when they are arrays of such scalars. The two attributes together define the **path** of an object - Magpylib makes sure that they are always of the same length. When the field is computed, it is automatically computed for the whole path.

```{tip}
To enable vectorized field computation, paths should always be used when modeling multiple object positions. Avoid using Python loops at all costs for that purpose! If your path is difficult to realize, consider using the [direct interface](docu-direct-interface) instead.
```

Magpylib offers two powerful methods for object manipulation:

::::{grid} 2
:::{grid-item-card}
:columns: 5
:shadow: none
<span style="color: orange">**move(**</span>`displacement`, `start="auto"`<span style="color: orange">**)**</span> -  move object by `displacment` input. `displacement` is a position vector (scalar input) or a set of position vectors (vector input).
:::
:::{grid-item-card}
:columns: 7
:shadow: none
<span style="color: orange">**rotate(**</span>`rotation`, `anchor=None`, `start="auto"`<span style="color: orange">**)**</span> - rotates the object by the `rotation` input about an anchor point defined by the `anchor` input. `rotation` is a [Scipy Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html), and `anchor` is a position vector. Both can be scalar or vector inputs. With `anchor=None` the object is rotated about its `position`.
:::
::::

- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start="auto"` the index is set to `start=0` and the functionality is **moving objects around**.
- Vector input of length $n$ applies the $n$ individual operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start="auto"` the index is set to `start=len(object path)` and the functionality is **appending paths**.

The practical application of this formalism is best demonstrated by the following program

```python
import magpylib as magpy

sensor = magpy.Sensor()
print(sensor.position)                                    # default value
#   --> [0. 0. 0.]

sensor.move((1,1,1))                                      # scalar input is by default applied
print(sensor.position)                                    # to the whole path
#   --> [1. 1. 1.]

sensor.move([(1,1,1), (2,2,2)])                           # vector input is by default appended
print(sensor.position)                                    # to the existing path
#   --> [[1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]]

sensor.move((1,1,1), start=1)                             # scalar input and start=1 is applied
print(sensor.position)                                    # to whole path starting at index 1
#   --> [[1. 1. 1.]  [3. 3. 3.]  [4. 4. 4.]]

sensor.move([(0,0,10), (0,0,20)], start=1)                # vector input and start=1 merges
print(sensor.position)                                    # the input with the existing path
#   --> [[ 1.  1.  1.]  [ 3.  3. 13.]  [ 4.  4. 24.]]     # starting at index 1.
```

Several extensions of the `rotate` method give a lot of flexibility with object rotation:

:::{dropdown} <span style="color: orange">**rotate_from_angax(**</span>`angle`, `axis`, `anchor=None`, `start="auto"`, `degrees=True` <span style="color: orange">**)**</span>
`angle`: scalar or array_like with shape (n,)
    Angle(s) of rotation in units of deg (by default).

`axis`: str or array_like, shape (3,)
    The direction of the axis of rotation. Input can be a vector of shape (3,) or a string 'x', 'y' or 'z' to denote respective directions.

`anchor`: None, 0 or array_like with shape (3,) or (n,3), default=None
    The axis of rotation passes through the anchor point given in units of mm. By default (anchor=None) the object will rotate about its own center. anchor=0 rotates the object about the origin (0,0,0).

`start`: int or str, default='auto'
    Starting index when applying operations. See 'General move/rotate behavior' above for details.

`degrees`: bool, default=True
    Interpret input in units of deg or rad.
:::

:::{dropdown} <span style="color: orange">**rotate_from_rotvec(**</span>`rotvec`, `anchor=None`, `start="auto"`, `degrees=True` <span style="color: orange">**)**</span>
`rotvec` : array_like, shape (n,3) or (3,)
    Rotation input. Rotation vector direction is the rotation axis, vector length is the rotation angle in units of rad.

`anchor`: None, 0 or array_like with shape (3,) or (n,3), default=None
    The axis of rotation passes through the anchor point given in units of mm. By default (anchor=None) the object will rotate about its own center. anchor=0 rotates the object about the origin (0,0,0).

`start`: int or str, default='auto'
    Starting index when applying operations. See 'General move/rotate behavior' above for details.

`degrees`: bool, default=True
    Interpret input in units of deg or rad.
:::

:::{dropdown} <span style="color: orange">**rotate_from_euler(**</span> `angle`, `seq`, `anchor=None`, `start="auto"`, `degrees=True` <span style="color: orange">**)**</span>
angle: int, float or array_like with shape (n,)
    Angle(s) of rotation in units of deg (by default).
`seq` : string
    Specifies sequence of axes for rotations. Up to 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic rotations cannot be mixed in one function call.

`anchor`: None, 0 or array_like with shape (3,) or (n,3), default=None
    The axis of rotation passes through the anchor point given in units of mm. By default (anchor=None) the object will rotate about its own center. anchor=0 rotates the object about the origin (0,0,0).

`start`: int or str, default='auto'
    Starting index when applying operations. See 'General move/rotate behavior' above for details.

`degrees`: bool, default=True
    Interpret input in units of deg or rad.
:::

:::{dropdown} <span style="color: orange">**rotate_from_quat(**</span>`quat`, `anchor=None`, `start="auto"` <span style="color: orange">**)**</span>
`quat` : array_like, shape (n,4) or (4,)
    Rotation input in quaternion form.

`anchor`: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
    The axis of rotation passes through the anchor point given in units of mm.
    By default (`anchor=None`) the object will rotate about its own center.
    `anchor=0` rotates the object about the origin `(0,0,0)`.

`start`: int or str, default=`'auto'`
    Starting index when applying operations. See 'General move/rotate behavior' above
    for details.
:::

:::{dropdown} <span style="color: orange">**rotate_from_mrp(**</span>`matrix`, `anchor=None`, `start="auto"` <span style="color: orange">**)**</span>
`matrix` : array_like, shape (n,3,3) or (3,3)
    Rotation input. See scipy.spatial.transform.Rotation for details.

`anchor`: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
    The axis of rotation passes through the anchor point given in units of mm.
    By default (`anchor=None`) the object will rotate about its own center.
    `anchor=0` rotates the object about the origin `(0,0,0)`.

`start`: int or str, default=`'auto'`
    Starting index when applying operations. See 'General move/rotate behavior' above
    for details.
:::

:::{dropdown} <span style="color: orange">**rotate_from_mrp(**</span>`mrp`, `anchor=None`, `start="auto"` <span style="color: orange">**)**</span>
`mrp` : array_like, shape (n,3) or (3,)
    Rotation input. See scipy Rotation package for details on Modified Rodrigues
    Parameters (MRPs).

`anchor`: `None`, `0` or array_like with shape (3,) or (n,3), default=`None`
    The axis of rotation passes through the anchor point given in units of mm.
    By default (`anchor=None`) the object will rotate about its own center.
    `anchor=0` rotates the object about the origin `(0,0,0)`.

`start`: int or str, default=`'auto'`
    Starting index when applying operations. See 'General move/rotate behavior' above
    for details.
:::

When objects with different path lengths are combined, e.g. when computing the field, the shorter paths are treated as static beyond their end to make the computation sensible. Internally, Magpylib follows a philosophy of edge-padding and end-slicing when adjusting paths.

::::{grid} 2
:::{grid-item-card}
:columns: 7
:shadow: none
**Edge-padding:** whenever path entries beyond the existing path length are needed the edge-entries of the existing path are returned. This means that the object is considered to be “static” beyond its existing path.
:::
:::{grid-item-card}
:columns: 5
:shadow: none
**End-slicing:** whenever a path is automatically reduced in length, Magpylib will slice such that the ending of the path is kept.
:::
::::

The tutorial {ref}`gallery-tutorial-paths` shows intuative good practice examples of the important functionality described in this section.
