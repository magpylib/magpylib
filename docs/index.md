# The Magpylib Documentation

::::{grid} 2
:gutter: 2

:::{grid-item}
:columns: 7
Magpylib is an **open-source Python package** for calculating static **magnetic fields** of magnets, currents and other sources. The computation is based on vectorized **explicit expressions**, solutions to the macroscopic magnetostatic problems, and is therefore **extremely fast**.
:::
:::{grid-item}
:columns: 5
![](../../_static/images/index_head.png)
:::
::::

## How it works

::::{grid} 1
:gutter: 2

:::{grid-item}
:columns: 12
![](_static/images/index_flowchart.png)
:::

:::{grid-item}
:columns: 12
In Magpylib, **sources** (magnets, currents, ...) and **observers** (sensors, position grids, ...) are created as Python objects with position and orientation attributes. These objects can be **grouped** and **moved** around. The system can be **viewed** graphically trough various backends. The **magnetic field** is computed in the observer reference frame. Magpylib collects all inputs, and vectorizes the computation for maximal performance.
:::
::::

## Ressources

::::{grid} 6
:gutter: 2

:::{grid-item-card}
:link: get-started
:link-type: ref
:link-alt: link to Getting Started
:img-top: ../../_static/images/index_icon_getting_started.png
:text-align: center
**Get Started**
:::

:::{grid-item-card}
:link: gallery
:link-type: ref
:link-alt: link to Gallery
:img-top: ../../_static/images/index_icon_gallery.png
:text-align: center
**Gallery**
:::

:::{grid-item-card}
:link: docu
:link-type: ref
:link-alt: link to Documentation
:img-top: ../../_static/images/index_icon_docu.png
:text-align: center
**Docu**
:::

:::{grid-item-card}
:link: https://github.com/magpylib/magpylib
:link-alt: link to Github
:img-top: ../../_static/images/index_icon_github.png
:text-align: center
**Project**
:::

:::{grid-item-card}
:link: contributing
:link-type: ref
:link-alt: link to Contribution Guide
:img-top: ../../_static/images/index_icon_contributing.png
:text-align: center
**Contribute**
:::

:::{grid-item-card}
:link: https://www.sciencedirect.com/science/article/pii/S2352711020300170
:link-alt: link to Journal
:img-top: ../../_static/images/index_icon_academic.png
:text-align: center
**Paper**
:::

```{toctree}
:maxdepth: 2
:hidden:

_pages/reso_get_started.md
_pages/gallery/gallery_index.md
_pages/docu/docu_index.md
_pages/reso_contributing.md
_pages/reso_code_of_conduct.md
_pages/reso_license.md
_pages/reso_changelog.md
_pages/reso_site_notice.md
```

::::

<!-- ## How can I cite this library ?

Thank you for considering to give credit! A valid bibtex entry for the [2020 open-access paper](https://www.sciencedirect.com/science/article/pii/S2352711020300170) would be

```latex
@article{ortner2020magpylib,
  title={Magpylib: A free Python package for magnetic field computation},
  author={Ortner, Michael and Bandeira, Lucas Gabriel Coliado},
  journal={SoftwareX},
  volume={11},
  pages={100466},
  year={2020},
  publisher={Elsevier}
}
``` -->




<!-- 
:orphan:

.. title:: Magpylib documentation

##################################
Magpylib |release| documentation
##################################


* Magpylib is a Python package for calculating static magnetic fields of magnets, currents and other sources.
* The computation is based on vectorized explicit expressions and is extremely fast.
* In Magpylib, sources (magnets, currents, ...) and observers (sensors, position grids, ...) are created as Python objects with position and orientation attributes.
* These objects can then be moved around, grouped, and displayed graphically.
* Finally, the magnetic field is computed in the reference frame of the observers.

.. grid:: 1 1 6 6

    .. grid-item::
      .. image:: _static/images/index_icon1.png
        :height: 100
        :target: auto_examples/plot_field_mpl_streamplot.html
    .. grid-item::
      .. image:: _static/images/index_icon2.png
        :height: 100
        :target: https://stackoverflow.com/questions/14087784/linked-image-in-restructuredtext
    .. grid-item::
      .. image:: _static/images/index_icon3.png
        :height: 100
        :target: auto_examples/plot_field_pv_streamlines.html
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

***************************
When can you use Magpylib ?
***************************
The expressions used in Magpylib describe perfectly homogeneous magnets, surface charges, and line currents with natural boundary conditions. Magpylib is at its best when dealing with static air-coils (no eddy currents, no soft-magnetic cores) and high grade permanent magnets (Ferrite, NdFeB, SmCo or similar materials). When **magnet** permeabilities are below $\mu_r < 1.1$ the error typically undercuts few % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not included. The line **current** solutions give the exact same field as outside of a wire that carries a homogeneous current. For more details check out the :ref:`physComp` section.

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

* The Magpylib project is hosted and organized on `GitHub`_.
* We welcome your contribution ! Please follow the guide for :ref:`contributing`.
* Always abide by our :ref:`code_of_conduct`.
* There is a `Youtube video`_ introduction to Magpylib v4.0.0 presented within the `GSC network`_.
* An `open-access paper`_ describes version 2 where fundamental concepts are still intact in later versions.

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

  print(cube.position)                   # -> [0. 0. 0.]
  print(cube.orientation.as_rotvec())    # -> [0. 0. 0.]

  # Manipulate object position and orientation through the respective attributes,
  # or by using the powerful `move` and `rotate` methods.

  cube.move((0,0,-2))
  cube.rotate_from_angax(angle=45, axis='z')
  print(cube.position)                            # -> [0. 0. -2.]
  print(cube.orientation.as_rotvec(degrees=True)) # -> [0. 0. 45.]

  # Compute the magnetic field in units of mT at a set of observer positions. Magpylib
  # makes use of vectorized computation. Hand over all field computation instances,
  # e.g. different observer positions, at one funtion call. Avoid Python loops !!!

  observers = [(0,0,0), (1,0,0), (2,0,0)]
  B = magpy.getB(cube, observers)
  print(B.round()) # -> [[-91. -91.   0.]
                  #      [  1. -38.  84.]
                  #      [ 18. -14.  26.]]

  # Sensors are observer objects that can have their own position and orientation.
  # Compute the H-field in units of kA/m.

  sensor = magpy.Sensor(position=(0,0,0))
  sensor.rotate_from_angax(angle=45, axis=(1,1,1))
  H = magpy.getH(cube, sensor)
  print(H.round()) # -> [-95. -36. -14.]

  # Position and orientation attributes of Magpylib objects can be vectors of
  # multiple positions/orientations refered to as "paths". When computing the
  # magnetic field of an object with a path, it is computed at every path index.

  cube.position = [(0,0,-2), (1,0,-2), (2,0,-2)]
  B = cube.getB(sensor)
  print(B.round()) # -> [[-119.  -45.  -18.]
                  #      [   8.  -73.  -55.]
                  #      [  15.  -30.   -8.]]

  # When several objects are involved and things are getting complex, make use of
  # the `show` function to view your system through Matplotlib, Plotly or Pyvista backends.

  magpy.show(cube, sensor, backend='pyvista')

Other important features include

* **Collections**: Group multiple objects for common manipulation
* **Complex magnet shapes**: Create magnets with arbitrary shapes
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
  :caption: DOCUMENTATION
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
  _license.md -->
