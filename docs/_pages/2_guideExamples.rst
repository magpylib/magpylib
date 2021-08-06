.. _examples:

*******************************
Example Codes
*******************************

This section includes a few code examples that show how the library can be used and what it can be used for. Detailed package, class, method and function documentations are found in the library docstrings :ref:`genindex`.

Technical details are outlined in :ref:`docu`.


Contents
########

* :ref:`examples-simple`
* :ref:`examples-basic`
* :ref:`examples-position_orientation`
* :ref:`examples-paths`
* :ref:`examples-field1`
* :ref:`examples-sensor`
* :ref:`examples-collection`
* :ref:`examples-getBHv`
* :ref:`examples-coil`


.. _examples-simple:

Just compute the field
######################

The most fundamental functionality of the library - compute the field (B in [mT], H in [kA/m]) of a source (here Cylinder magnet) at the observer position (1,2,3).

.. code-block:: python

    from magpylib.magnet import Cylinder
    src = Cylinder(magnetization=(222,333,444), dimension=(2,2))
    B = src.getB((1,2,3))
    print(B)
    # Output: [-2.74825633  9.77282601 21.43280135]


.. _examples-basic:

Basic functionality
###################

In this general example two source objects (magnet and current) are created, moved and rotated. The system geometry before and after move/rotate is displayed together with the magnetic field in the xz-plane. Notice that xz is a symmetry plane where the field has no y-component.

.. plot:: _codes/examples_basic.py
    :include-source:


.. _examples-position_orientation:

Object position and orientation
################################

All Magpylib source and sensor objects have ``position`` (ndarray of shape (3,)) and ``orientation`` (scipy Rotation object) attributes that can be manipulated by hand, with ``move`` and ``rotate`` methods. The method ``rotate_from_angax`` provides rotation without refering to scipy Rotation objects.

.. plot:: _codes/examples_position_orientation.py
    :include-source:


.. _examples-paths:

Paths
#####

Position and orientation attributes can also be of shape (n,3) where n denotes multiple steps in a path. Paths can be generated conveniently using the ``move`` and ``rotate`` methods. Paths are automatically shown in ``display()``. Through the kwarg ``show_path=x`` the object can be shown at each x'th path position. Fields are automatically computed for each path position, see :ref:`examples-sensor`.

.. plot:: _codes/examples_paths.py
    :include-source:


.. _examples-field1:

Field computation
#################

Compute B-field in units of [mT] and H-field in units of [kA/m] directly from sources or through top-level functions. For the functions/methods ``getB`` and ``getH`` sources and observers must always be defined. Sources are e.g. magnets, currents or collections. Observers can be arbitrary arrays/lists/tuples of position vectors (shape=(n1,n2,n3,...,3)) or Sensor objects.

.. plot:: _codes/examples_field1.py
    :include-source:


.. _examples-sensor:

Sensors
#######

Sensors are Magpylib objects that can function as observers and simulate typical industrial magnetic field sensors. They can be moved and rotated just like source objects and automatically compute the field in their local coordinate system. Sensors can be defined with multiple internal pixel cells, that correspond to positions inside the sensor where the field is determined. Sensors are represented by a coordinate cross in ``display()`` and pixel positions are indicated by o-markers.

.. plot:: _codes/examples_sensor.py
    :include-source:


.. _examples-collection:

Collections
###########

Multiple Magpylib sources can be grouped into Collection objects for common manipulation. Collection objects do not have their own position and orientation attributes, but have ``move``, ``rotate`` and ``getBH`` methods defined. Geometric operations applied to a Collection will be applied individually to all objects in the Collection. For ``getB`` and ``getH`` the Collection acts as a single source.

.. plot:: _codes/examples_collection.py
    :include-source:


.. _examples-getBHv:

getBHv - Direct access to analytical solutions
##############################################

Magpylib provides direct access to the vectorized analytical formulas through the top level ``getB_dict`` and ``getH_dict`` functions. The input arguments must be shape (n,x) vectors/lists/tuple. Depending on the ``source_type``, different input arguments are expected (see docstring for details). Static inputs e.g. of shape (3,) are automatically tiled up to shape (n,3).

.. plot:: _codes/examples_getBHv.py
    :include-source:


.. _examples-coil:

Modelling a Coil
################

A coil consists of large number of windings that can be modeled using ``Circular`` sources. The total coil is then a ``Collection`` of windings.

.. plot:: _codes/examples_coil.py
    :include-source:
