*********************
Library Documentation
*********************

The idea behind magpylib is to provide a simple and easy-to-use interface
for computing the magnetic field of magnets, currents and moments. The
computation is based on (semi-)analytical solutions found in the literature
discussed in the `Physics Section`__.

__ _pages/9_Physics/


Contents
########

* `Package Structure`_
* `Units`_
* `IO types`_
* `The Source Class`_

  * `Position and Orientation`_
  * `Geometry / Dimension & Excitation`_
  * `Methods for Geometric Manipulation`_

* `Calculating the Magnetic Field`_

  *`Using magpylib.vector`_



Package Structure
#################

The top level of magpylib contains the sub-packages  and :mod:`~magpylib.source`, :mod:`~magpylib.vector` and :mod:`~magpylib.math`, the classes :class:`magpylib.Collection` and :class:`magpylib.Sensor` as well as the function :meth:`magpylib.displaySystem`.

1. The **source module** includes a set of classes that represent physical sources of the magnetic field (e.g. permanent magnets).

2. The **vector module** includes functions for performance computation of the magnetic field.

3. The **math module** contains practical functions for working with angle-axis rotations and transformation to Euler angles.

4. The **Collection class** is used to group sources and for common manipulation.

5. The **Sensor Class** represents a 3-axis magnetic field point-sensor.

6. The **displaySystem function** is used to create a graphical output of the system geometry.

.. figure:: ../_static/images/documentation/lib_structure.png
    :align: center
    :alt: Library structure fig missing !!!
    :figclass: align-center
    :scale: 60 %

    **Figure:** Outline of library structure.


Units
######

In magpylib all inputs and outputs are made in the physical units of

- **Millimeter** for lengths
- **Degree** for angles
- **Millitesla** for magnetization, magnetic moment and magnetic field,
- **Ampere** for currents.


IO types
##########

The library **input** is constructed so that any

- **scalar input** can be `int`, `float` or of `numpy.float` type
- **vector/matrix input** can be given either in the form of a `list`, as a `tuple` or as a `numpy.array`

unless specifically state otherwise in the docstrings. For example, the `magpylib.vector` functions require `numpy.array` input.

The library **output** and all object attributes are either of `numpy.float64` or `numpy.array64` type.


The Source Class
#################

This is the core class of the library. The idea is that source objects represent physical magnetic sources in Cartesian three-dimensional space. The following source types are currently implemented in magpylib.

.. figure:: ../_static/images/documentation/SourceTypes.JPG
  :align: center
  :scale: 60 %

  **Figure:** Source types currently available in magpylib.

All source objects share various attributes and methods. The attributes characterize the source (e.g. position, orientation, dimension) while the methods can be used for geometric manipulation and for calculating the magnetic field. The figure below gives a graphical overview.

.. figure:: ../_static/images/documentation/sourceVars_Methods.png
  :align: center
  :scale: 60 %

  **Figure:** Illustration of attributes and methods of the source class.


Position and Orientation
------------------------
The most fundamental properties of a source object `s` are position and orientation which are represented through the attributes `s.position` (arr3), `s.angle` (float) and `s.axis`(arr3). At source initialization, if no values are specified, the source object is initialized by default with `position=(0,0,0)`, and **init orientation** defined to be `angle=0` and `axis=(0,0,1)`.

Due to their different nature each source type is characterized by different attributes. However, in general the `position` attribute refers to the position of the geometric center of the source. The **init orientation** generally defines sources standing upright oriented along the Cartesian coordinates axes, see e.g. the following image.

An orientation given by (`angle`,`axis`) refers to a rotation of the source RELATIVE TO the **init orientation** about an axis specified by the `axis` vector anchored at the source `position`. The angle of this rotation is given by the `angle` attribute. Mathematically, every possible orientation can be expressed by such a single angle-axis rotation. For easier use of the angle-axis rotation and transformation to Euler angles the `Math Package`_ provides some useful methods. 

.. figure:: ../_static/images/documentation/source_Orientation.JPG
  :align: center
  :scale: 50 %

  **Figure:** Illustration of the angle-axis system for source orientations.


Geometry / Dimension & Excitation
--------------------

While position and orientation have default values, a source is defined through its geometry (e.g. Cylinder) and excitation (e.g. Magnetization Vector) which must be initialized to provide meaning. The source geometry is generally described by the `dimension` attribute. However, as each source requires different input parameters, the format is always different. Detailed information about the attributes of each specific source type and how to initialize them can be found in the respective class docstrings:
:mod:`~magpylib.source.magnet.Box`, :mod:`~magpylib.source.magnet.Cylinder`,:mod:`~magpylib.source.magnet.Sphere`, :mod:`~magpylib.source.magnet.Facet`, :mod:`~magpylib.source.current.Line`, :mod:`~magpylib.source.current.Circular`, :mod:`~magpylib.source.moment.Dipole` 

The excitation is either the magnetization, the current or the magnetic moment. Magnet sources represent homogeneously magnetized permanent magnets (other types with radial or multipole magnetization are not implemented at this point). The magnetization vector is described by the `magnetization` attribute (arr3). The magnetization vector is always given with respect to the INIT ORIENTATION of the magnet. The current sources represent line currents. They require a scalar `current` input. The moment class represents a magnetic dipole moment which requires a `moment` (arr3) input.

.. note::
  For convenience **magnetization**, **current**, **dimension**, **position** are initialized through the keywords **mag**, **curr**, **dim** and **pos**.

The following code shows how to initialize a source object, a D4H5 permanent magnet cylinder with diagonal magnetization, positioned with the center in the origin, standing upright with axis in z-direction.

.. code-block:: python

  from magpylib.source.magnet import Cylinder

  s = Cylinder( mag = [500,0,500], # The magnetization vector in mT.
                dim = [4,5])       # dimension (diameter,height) in mm.
                
  # no pos, angle, axis specified so default values are used

  print(s.magnetization)  # Output: [500. 0. 500.]
  print(s.dimension)      # Output: [4. 5.]
  print(s.position)       # Output: [0. 0. 0.]
  print(s.angle)          # Output: 0.0
  print(s.axis)           # Output: [0. 0. 1.]

.. figure:: ../_static/images/documentation/Source_Display.JPG
  :align: center
  :scale: 30 %

  **Figure:** Magnet geometry created by above code: A cylinder which stands upright with geometric center at the origin.


Methods for Geometric Manipulation
----------------------------------

In most cases we want to move the magnet to a designated position, orient it in a desired way or change its dimension dynamically. There are several ways to achieve this:

**At initialization:**

When initializing the source we can set all attributes as desired. So instead of *moving one source around* one could create a new one for each set of parameters of interest.

**Manipulation after initialization:**

We initialize the source and manipulate it afterwards as desired by

1. directly setting the source attributes.
2. using provided methods of manipulation.

The latter is often the most practical and intuitive way. To this end the source class provides a set of methods for convenient geometric manipulation. The methods include `setPosition` and `move` for translation of the objects as well as `setOrientation` and `rotate` for rotation operations. Upon application to source objects they will simply modify the object attributes accordingly.

* `s.setPosition(newPos)`: Moves the source to the position given by the argument vector (*newPos*. *s.position -> newPos*)
* `s.move(displacement)`: Moves the source by the argument vector *displacement*. (*s.position -> s.position + displacement*) 
* `s.setOrientation(angle,axis)`: This method sets a new source orientation given by *angle* and *axis*. (*s.angle -> angle, s.axis -> axis*)
* `s.rotate(angle,axis,anchor=self.position)`: Rotates the source object by *angle* about the axis *axis* which passes through a position given by *anchor*. As a result position and orientation attributes are modified. If no value for anchor is specified, the anchor is set to the object position, which means that the object rotates about itself.

The following videos show the application of the four methods for geometric manipulation.

|move| |setPosition|

.. |setPosition| image:: ../_static/images/documentation/setPosition.gif
  :width: 45%

.. |move| image:: ../_static/images/documentation/move.gif
  :width: 45%

|rotate| |setOrientation|

.. |setOrientation| image:: ../_static/images/documentation/setOrientation.gif
   :width: 45%

.. |rotate| image:: ../_static/images/documentation/rotate.gif
   :width: 45%

The following example code shows how geometric operations are applied to source objects.

.. code-block:: python

  from magpylib.source.magnet import Cylinder

  s = Cylinder( mag = [500,0,500], dim = [4,5])

  print(s.position)       # Output: [0. 0. 0.]

  s.move([1,2,3])
  print(s.position)       # Output: [1. 2. 3.]

  s.move([1,2,3])
  print(s.position)       # Output: [2. 4. 6.]


Calculating the Magnetic Field
##############################

To calculate the fields, magpylib uses mostly analytical expressions that can be found in the literature. A detailed analysis of the precision and applicability of these solutions can be found in the `Physics section`__. In a nutshell, the fields of dipole and current are exact for their geometry. For the magnet classes the analytical solutions deal with homogeneous, fixed magnetizations. For typical hard ferromagnets like Ferrite, Neodyme and SmCo the accuracy of the solution easily exceeds 98%.

__ _pages/9_Physics/

There are two possibilities to calculate the magnetic field:

1. Using the `s.getB(pos)` method of source objects.
2. Using the `magpylib.vector` subpackage.

**The first method:** Each source object (or collection) `s` has a method `s.getB(pos)` which returns the magnetic field generated by `s` at the position `pos`.

.. code-block:: python

  from magpylib.source.magnet import Cylinder
  s = Cylinder( mag = [500,0,500], dim = [4,5])
  print(s.getB([4,4,4]))       

  # Output: [ 7.69869084 15.407166    6.40155549]

Using magpylib.vector
---------------------

**The second method:** In most cases one will be interested to determine the field for a set of sensor positions, or for different magnet positions and orientations. While this can manually be achieved by looping `s.getB` this results in slow computation times. For performance computation the `magpylib.vector` subpackge contains the `getBv` functions that offer quick access to vectorized code. A discussion of vectorized code, SIMD and performance is shown in the `Physics & Computation`__ section.

__ _pages/9_Physics/

The core idea of the `magpylib.vector.getBv` functions is that the field is evaluated for `N` different sets of input parameters. The `N` input parameters (e.g. magnetization vectors) are provided as arrays of size *N* (e.g. *Nx3* array for the magnetization input) to the `getBv` functions:

`getBv_magnet(type,MAG,DIM,POSo,POSm,[angs1,angs2,...],[AXIS1,AXIS2,...],[ANCH1,ANCH2,...])`

* `type` is a string that specifies the magnet geometry (e.g. 'box' or 'sphere').
* `MAG` is an *Nx3* array of magnetization vectors.
* `DIM` is an *Nx3* array of magnet dimensions.
* `POSo` is an *Nx3* array of observer positions.
* `POSm` is an *Nx3* array of initial (before rotation) magnet positions.
* The inputs `[angs]`, `[AXIS]`, `[ANCH]` are a lists of size *N*/*Nx3* arrays that correspond to angles, axes and anchors of rotation operations. By providing multiple list entries one can apply subsequent rotation operations. By ommitting the lists it is assumed that no rotation is applied.

As a rule of thumb, `s.getB()` will be faster than `getBv` for ~5 or less field evaluations while the vectorized code will be up to ~100 times faster for 10 or more field evaluations. To achieve this performance it is critical that one follows the vectorized code paradigm when creating the `getBv` inputs.

In the following example the magnetic field at a fixed sensor is calculated for a magnet that moves in x-direction above the sensor.

.. code-block:: python

  import magpylib as magpy
  import numpy as np

  # vector size: we calculate the field N times with different inputs
  N = 1000

  # Constant vectors
  mag  = np.array([0,0,1000],dtype='float64')    # magnet magnetization
  dim  = np.array([2,2,2],dtype='float64')       # magnet dimension
  poso = np.array([0,0,-4],dtype='float64')      # position of observer

  # magnet x-positions
  xMag = np.linspace(-10,10,N)

  # magpylib classic ---------------------------

  Bc = np.zeros((N,3))
  for i,x in enumerate(xMag):
      s = magpy.source.magnet.Box(mag,dim,[x,0,0])
      Bc[i] = s.getB(poso)

  # magpylib vector ---------------------------

  # Vectorizing input using numpy native instead of python loops
  MAG = np.tile(mag,(N,1))        
  DIM = np.tile(dim,(N,1))        
  POSo = np.tile(poso,(N,1))
  POSm = np.c_[xMag,np.zeros((N,2))]

  # N-times evalulation of the field with different inputs
  Bv = magpy.vector.getBv_magnet('box',MAG,DIM,POSo,POSm)


  # result ----------------------------------- 
  # Bc == Bv

More examples of vectorized code can be found in the `Examples`__ section.

__ _pages/2_guideExamples/