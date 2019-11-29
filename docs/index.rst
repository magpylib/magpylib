.. magpylib documentation master file, created by
   sphinx-quickstart on Tue Feb 26 11:58:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is magpylib ?
~~~~~~~~~~~~~~~~~~
- Python package for calculating magnetic fields of magnets, currents and
  moments (sources).
- Provides convenient methods to generate, geometrically manipulate, group
  and vizualize assemblies of sources.
- The magnetic fields are determined from underlying (semi-analytical)
  solutions which results in fast computation times and requires little
  computation power.

.. image:: _static/images/index/sourceFundamentals.png
   :align: center


Quickstart
~~~~~~~~~~
Install magpylib with pip (``>> pip install magpylib``).

Run this simple code to calculate the magnetic field of a cylindrical magnet:

.. highlight:: python
    :linenothreshold: 1

    from magpylib.source.magnet import Cylinder
    s = Cylinder( mag = [0,0,350], dim = [4,5])
    print(s.getB([4,4,4]))       

    # Output: [ 5.08641867  5.08641867 -0.60532983]


.. code-block:: python

    from magpylib.source.magnet import Cylinder
    s = Cylinder( mag = [0,0,350], dim = [4,5])
    print(s.getB([4,4,4]))       

    # Output: [ 5.08641867  5.08641867 -0.60532983]



``
In [1]: from magpylib.source.magnet import Cylinder

In [2]: s = Cylinder( mag = [0,0,350], dim = [4,5])

In [3]: s.getB([4,4,4])
Out[3]: array([ 5.08641867  5.08641867 -0.60532983])
``




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
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`
