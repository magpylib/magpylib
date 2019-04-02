.. MagPyLib documentation master file, created by
   sphinx-quickstart on Tue Feb 26 11:58:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MagPyLib's documentation!
====================================

About
~~~~~
*A simple and user friendly magnetic toolbox for Python 3.2+*


What is Magpylib ?
~~~~~~~~~~~~~~~~~~
 - Magpylib is a Python library for calculating magnetic fields from permanent magnets and current distributions. 
 - It provides an intuitive magnetic source class to quickly generate, group, geometrically manipulate and visualize assemblies of magnets and currents and the fields they generate.
 - The magnetic fields are determined from underlying (semi-) analytical solutions that are found in the literature.

Why Magpylib ?
~~~~~~~~~~~~~~
- Provide a fast and convenient tool to quickly calculate magnetic fields directly in a local Python work-environment for immediate analysis, test and manipulation.
- This is made possible by the fast computation times of the underlying analytical solutions which can be in the sub millisecond range and require no computation power as opposed to heavy numerical simulation like FEM.
- Practical tool for magnetic system design
    - Efficient for multivariate global optimization problems (for geometric shape variation).
    - Quick visualization of fields of complex assemblies (for finding a magnetic map concept).


Table of Contents
-----------------

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Pages:

   _pages/*

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Guide:

   _pages/_guide/*


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Library Docstrings:

   _autogen/magpylib
   _autogen/magpylib.source
   _autogen/magpylib.math




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
