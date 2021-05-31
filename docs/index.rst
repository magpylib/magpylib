.. magpylib documentation master file, created by
   sphinx-quickstart on Tue Feb 26 11:58:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is Magpylib ?
##################

- Python package for calculating static magnetic fields of magnets, currents and moments (sources).
- The fields are computed using fully vectorized analytical solutions (very fast but limited geometries, no material response)
- The field computation is coupled to a geometry interface (position, orientation, motion) to determine relative motion between sources and observers.

.. image:: _static/images/index/sourceFundamentals.png
   :align: center

When can you use Magpylib ?
###########################

The analytical solutions are exact when there is no material response. In permanent magnets, when remanent permeabilities (susceptibilities) are below ~1.1 the error is typically below 1% (long magnet shapes are better, large distance from magnet is better).

Magpylib is at its best when dealing with air-coils and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials).

Quickstart
##########

Install magpylib with pip or conda:

``>> pip install magpylib``.
``>> conda install magpylib``.

**Example:**

Run this simple code to calculate the magnetic field of a cylindrical magnet.

.. code-block:: python

    import magpylib as mag3
    s = mag3.magnet.Cylinder(magnetization=(0,0,350), dimension=(4,5))
    observer_pos = (4,4,4)
    print(s.getB(observer_pos))

    # Output: [ 5.08641867  5.08641867 -0.60532983]

In this example, a cylinder shaped permanent magnet with diameter and height of 4 and 5 millimeter respectively is created in a global coordinate system with cylinder axis parallel to the z-axis and geometric magnet center in the origin. The magnetization / remanence field is homogeneous and points in z-direction with an amplitude of 350 millitesla. The magnetic field is calculated in units of millitesla at the observer position (4,4,4) in units of millimeter.


**Ressources**

Examples can be found in the `Examples Section`__.

__ _pages/2_guideExamples/

Technical details can be found in the :ref:`docu` .


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Content:

   _pages/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Library Docstrings:

   _autogen/magpylib
   _autogen/magpylib.source
   _autogen/magpylib.math


Index
################

* :ref:`modindex`
* :ref:`genindex`
.. * :ref:`search`
