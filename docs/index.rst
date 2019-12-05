.. magpylib documentation master file, created by
   sphinx-quickstart on Tue Feb 26 11:58:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is magpylib ?
##################

- Free Python package for calculating magnetic fields of magnets, currents and moments (sources).
- Provides convenient methods to create, geometrically manipulate, group and visualize assemblies of sources.
- The magnetic fields are determined from underlying (semi-analytical) solutions which results in fast computation times and requires little computation power.

.. image:: _static/images/index/sourceFundamentals.png
   :align: center

Quickstart
##########

Install magpylib with pip: ``>> pip install magpylib``.

**Example:**

Run this simple code to calculate the magnetic field of a cylindrical magnet.

.. code-block:: python

    from magpylib.source.magnet import Cylinder
    s = Cylinder( mag = [0,0,350], dim = [4,5])
    print(s.getB([4,4,4]))       

    # Output: [ 5.08641867  5.08641867 -0.60532983]

In this example the cylinder axis is parallel to the z-axis. The diameter and height of the magnet are 4 millimeter and 5 millimeter respectively and the magnet position (=geometric center) is in the
origin. The magnetization / remanence field is homogeneous and points in z-direction with an amplitude of 350 millitesla.  Finally, the magnetic field **B** is calculated in units of millitesla at
the positition *[4,4,4]* millimeter.

**Example:**

The following code calculates the combined field of two magnets. They are geometrically manipulated, the system geometry is displayed together with the field in the xz-plane.

.. plot:: pyplots/examples/01_SimpleCollection.py
   :include-source:


More examples can be found in the `Examples Section`__.

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


Index and tables
################

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`
