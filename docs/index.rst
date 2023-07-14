:orphan:

.. title:: Magpylib documentation

##################################
Magpylib |release| documentation
##################################


* Magpylib is a Python package for calculating 3D static magnetic fields of magnets, currents and other sources.
* The computation is based on vectorized explicit expressions and is extremely fast.
* With Magpylib, sources (magnets, currents, ...) and observers (sensors, position grids, ...) are created as Python objects with position and orientation properties.
* These objects can be moved around, grouped, displayed graphically, and the field is computed in the reference frame of the observers.

.. grid:: 1 1 6 6

    .. grid-item::
      .. image:: _static/images/index_icon1.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon2.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon3.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon4.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon5.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon6.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext

`TEMP LINK TO GALLERY`_

.. _TEMP LINK TO GALLERY: auto_examples/index.html

***************************
When can you use Magpylib ?
***************************
The analytical solutions are exact when there is no material response and natural boundary conditions can be assumed. In general, Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnets (Ferrite, NdFeB, SmCo or similar materials). When **magnet** permeabilities are below $\mu_r < 1.1$ the error typically undercuts 1-5 % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. The line **current** solutions give the exact same field as outside of a wire that carries a homogeneous current. For more details check out the :ref:`physComp` section.

*****************************
Installation and Dependencies
*****************************

.. grid:: 1 1 2 2

    .. grid-item::

        Install from PyPI with pip_:

        .. code-block:: bash

            pip install matplotlib

    .. grid-item::

        Install from conda forge with conda_:

        .. code-block:: bash

            conda install -c conda-forge magpylib

Magpylib supports *Python3.8+* and relies on common scientific computation libraries *Numpy*, *Scipy*, *Matplotlib* and *Plotly*. Optionally, *Pyvista* is recommended as graphical backend.

.. _pip: https://pip.pypa.io/en/stable/
.. _conda: https://docs.conda.io/en/latest/

**********
Ressources
**********

* The project is hosted and organized on `GitHub`_.
* We welcome your contribution ! Please follow the :ref:`contributing`.
* Always abide by the :ref:`code_of_conduct`.
* A `Youtube video`_ introdution to Magpylib v4.0.0 within the `GSC network`_.
* An `open-access paper`_ describing version 2 with basic concepts still intact in later versions.

.. _GitHub: https://github.com/magpylib/magpylib
.. _Youtube video: https://www.youtube.com/watch?v=LeUx6cM1vcs
.. _GSC network: https://www.internationalcollaboration.org/
.. _open-access paper: https://www.sciencedirect.com/science/article/pii/S2352711020300170

**********
Quickstart
**********

Here is an example how to use Magpylib.

.. code-block:: python

  import magpylib as magpy

  # Create a Cuboid magnet with magnetization (polarization) of 1000 mT pointing
  # in x-direction and sides of 1,2 and 3 mm respectively.

  cube = magpy.magnet.Cuboid(magnetization=(1000,0,0), dimension=(1,2,3))

  # By default, the magnet position is (0,0,0) and its orientation is the unit
  # rotation (given by a scipy rotation object), which corresponds to magnet sides
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

Other important features include

* **Collections**: Group multiple objects for common manipulation
* **Complex shapes**: Create magnets with arbitrary shapes
* **Graphics**: Styling options, graphic backends, animations, and 3D models
* **CustomSource**: Integrate your own field implementation
* **Direct interface**: Bypass the object oriented interface (max speed)

*****************************
How can I cite this library ?
*****************************

We would be happy if you give us credit for our efforts. A valid bibtex entry for the `2020 open-access paper`_ would be

.. _2020 open-access paper: https://www.sciencedirect.com/science/article/pii/S2352711020300170 

.. code-block:: latex

  @article{ortner2020magpylib,
    title={Magpylib: A free Python package for magnetic field computation},
    author={Ortner, Michael and Bandeira, Lucas Gabriel Coliado},
    journal={SoftwareX},
    volume={11},
    pages={100466},
    year={2020},
    publisher={Elsevier}
  }

*******
Content
*******

.. toctree::
  :maxdepth: 2
  :caption: API DOCUMENTATION
  :glob:

  _pages/docu_classes.md
  _pages/docu_pos_ori.md
  _pages/docu_field_comp.md
  _pages/docu_graphic_styles.md

.. toctree::
  :maxdepth: 2
  :caption: RESSOURCES

  auto_examples/index.rst
  _pages/ressources_physics.md
  _changelog.md
  _contributing.md
  _code_of_conduct.md
  _license.md
