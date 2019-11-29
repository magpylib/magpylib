.. magpylib documentation master file, created by
   sphinx-quickstart on Tue Feb 26 11:58:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to magpylib's documentation!
====================================

.. rst-class:: center

   *Powered by:*

.. image:: _static/images/index/sal.svg
   :align: center
   :target: https://silicon-austria-labs.com/en/

|

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

.. toctree::
   :glob:
   :maxdepth: 2
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
* :ref:`search`
