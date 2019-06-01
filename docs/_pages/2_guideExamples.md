# Guide - Examples and How to Use

It is the aim of this section to give a few code examples that show how the library can be used. Detailed information about the library structure can be found in the [documentation](0_documentation.md).

- Content
  - [A Simple Collection and its Field](#a-simple-collection-and-its-field)
  - [Simulating Magnet Motion](#simulating-magnet-motion)
  - [getBsweep]()
  - [Multi Processing]()
  - [Collections and Superposition]()
  - [Creating complex shapes(addition + subtration)]()

### A Simple Collection and its Field

In this first example a simple collection is created from two magnets. The magnets are geometrically manipulated and the system geometry is displayed using the `displaySystem` method. The field is then calculated on a grid and displayed in the xz-plane.

```eval_rst

.. plot:: pyplots/examples/01_SimpleCollection.py
   :include-source:

:download:`01_SimpleCollection.py <../pyplots/examples/01_SimpleCollection.py>`
```

### Orientation and Rotation Basics

In the following examples we show orientations and rotations of source objects. The first example shows a cubical magnet initialized with four different orientations defined by the classical Euler angle rotations about the three cartesian axes.

```eval_rst

.. plot:: pyplots/examples/00a_OrientRot1.py
   :include-source:

:download:`00a_OrientRot1.py <../pyplots/examples/00a_OrientRot1.py>`
```

The second example shows a general form of orientation for different angles about an axis (1,-1,1). The upper three boxes are initilized with different orientations. The lower three boxes are all initialized with INIT ORIENTATION and are then rotated (about themselves) to achive the same result as above.

```eval_rst

.. plot:: pyplots/examples/00b_OrientRot2.py
   :include-source:

:download:`00b_OrientRot2.py <../pyplots/examples/00b_OrientRot2.py>`
```

Finally, the last example shows rotations with desiganted anchor-axis combinations. Here we distinguish between pivot points (the closest point on the rotation axis to the magnet) and anchor points which are simply required to define an axis in 3D space (together with the direction).

```eval_rst

.. plot:: pyplots/examples/00c_OrientRot3.py
   :include-source:

:download:`00c_OrientRot3.py <../pyplots/examples/00c_OrientRot3.py>`
```

### Simulating Magnet Motion

In this example a joystick is simulated. A magnetic joystick is realized by a rod that can tilt freely (two degrees of freedom) about a center of tilt. The upper part of the rod is the joystick handle. At the bottom of the rod a cylindrical magnet (dimension *D/H*) with axial magnetization (amplitude *M0*) is fixed. The magnet lies at a distance *d* below the center of tilt. The system is constructed such that, when the joystick is in the center position a sensor lies at distance *gap* below the magnet and in the origin of a cartesian coordinate system. The magnet thus moves with the joystick above the fixed sensor.

In the following program the magnetic field is calculated for all degrees of freedom. Different tilt angles are set by rotation about the center of tilt by the angle *th* (different colors). Then the tilt direction is varied from 0 to 360 degrees by simulating the magnet 'motion' as rotation about the z-axis.

```eval_rst

.. plot:: pyplots/examples/02_MagnetMotion.py
   :include-source:

:download:`02_MagnetMotion.py <../pyplots/examples/02_MagnetMotion.py>`
```


### Translations and Rotations

All Objects, be it a Source Object or a Collection Object, have a set of methods that allow for Translations and Rotation.

```python
from magpylib import source

neutral = [0,0,0]

b = source.magnet.Box(mag=[1,2,3],
                      dim=[2,2,2],
                      pos=neutral)

b.setPosition([2,0,0]) ## Place object in [2,0,0]
print(b.position)      ## [2,0,0]

b.move([5,0,0])        ## Move 5 units in X
print(b.position)      ## [7,0,0]

b.setPosition(neutral) ## Place object in [0,0,0]
print(b.position)      ## [0,0,0]

b.move([5,0,0])        ## Move 5 units in X
print(b.position)      ## [5,0,0]

```

```eval_rst

.. note::
   Source Objects within a Collection will have their coordinates modified within the Collection frame, Collections do not create copies.
   If you'd like to avoid this, create `a deep copy <https://docs.python.org/3/library/copy.html>`_. of the source object and add the copy to the Collection instead.


Source Objects may be rotated in respect to themselves or an anchored pivot point.

The result of :func:`~magpylib.source.magnet.Cylinder.rotate` is affected relative to the current position of the Source Object.

If you'd like to set a position that is absolute to the Source's frame, use :func:`~magpylib.source.magnet.Cylinder.setPosition` instead as the manipulation is always the same. This is not available for Collection.

.. plot:: pyplots/guide/rotate1.py
   :include-source:

```

Rotations may also be done with an anchored pivot point. The following code adds two Objects to a Collection, and only moves one of them.

```eval_rst

.. plot:: pyplots/guide/rotate2.py
   :include-source:

```

---

Collections may be rotated using the previous logic as well. Keep in mind if an anchor is not provided, all objects will rotate relative to their own center.


```eval_rst

.. plot:: pyplots/guide/rotate3.py
   :include-source:

```

---

Ultimately, Collections can be added to other Collections, and rotated independently.

```eval_rst

.. plot:: pyplots/guide/rotate4.py
   :include-source:

```
### Multipoint Field Calculations

One of the greatest strengths of the analytical approach is that all desired points of a field computation may be done in parallel, reducing computation overhead.

```eval_rst
.. warning::

    Due to how multiprocessing works on **Windows Systems, the following structure for your code is mandatory**:

    .. code::
    
       from multiprocessing import freeze_support

       def your_code():
           ## Your code

       if __name__ == "__main__":
            freeze_support()
            your_code()

    Failure to comply to this will cause your code **to not yield any results.**
```
Here is an example calculating several marked points in sequence.

```eval_rst
.. plot::  pyplots/guide/multiprocessing1.py
   :include-source:

```

#### Displacement Input

The parallel function may also be utilized to calculate samples of several setups in parallel.

Field sample position, Source object orientation and positioning may be adjusted in every setup, like the following structure:
```eval_rst

.. image:: ../_static/images/user_guide/multiprocessing.gif
   :align: center

.. image:: ../_static/images/user_guide/sweep.png
   :align: center
   :scale: 50 % 

```

```python
from magpylib import source
from multiprocessing import freeze_support

def setup_creator(sensorPos,magnetPos,angle):
    # Return a properly defined setup
    axis = (0,0,1) # Z axis
    setup = [sensorPos,    # field sampler position
             magnetPos,    # magnet position
             (angle,axis)] # Rotation arguments
    return setup

def main():
    ## Define information for 8 setups
    sensors = [ [-1,-6,6], [-1,-5,5], 
                [-1,-4,4.5],[-1,-6,3.5], 
                [-1,-5,2.5], [-1,-4,1.5],
                [-1,-5,-0.5], [-1,-4,-1.0] ]

    angles = [  0, 30,
                60, 90,
                120,180,
                210,270 ]

    positions = [ [3,-4,6],[3,-4,5],
                  [3,-4,4],[3,-4,3,],
                  [3,-4,2],[3,-4,1],
                  [3,-4,0],[3,-4,-1] ]

    ## Define magnet
    b = source.magnet.Box([1,2,3],
                        [1,1,1])

    setups = [setup_creator(sensors[i],
                            positions[i],
                            angles[i]) for i in range(0,8)]

    # Calculate results sequentially
    results = b.getBsweep(setups)
    # Calculate results again but in parallel
    results = b.getBsweep(setups,multiprocessing=True)

    print(results)
    ## Result for each of the 8 setups:
    # [array([ 0.0033804 ,  0.00035464, -0.00266834]), 
    #  array([ 0.00151226, -0.00219274, -0.00340392]), 
    #  array([-0.00427052, -0.00226601, -0.00292288]), 
    #  array([-0.00213505, -0.00281333, -0.00213425]), 
    #  array([-0.00567799, -0.00189228, -0.00231176]), 
    #  array([-0.00371514,  0.00242773, -0.00302629]), 
    #  array([-0.00030278,  0.00243991, -0.00334978]),
    #  array([ 0.0049694 ,  0.00124235, -0.00372705])]

if __name__ == "__main__":
    freeze_support()
    main()
```








