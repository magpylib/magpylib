.. _docu:

***********************************
Documentation v4.0.0
***********************************

Brief overview and some critical information.


Contents
########

* :ref:`docu-idea`
* :ref:`docu-when_to_use`
* :ref:`docu-layout_functionality`
* :ref:`docu-performance`
* :ref:`docu-units_scaling`
* :ref:`docu-close_to_surface`
* :ref:`docu-superposition_complex`


.. _docu-idea:

The idea behind Magpylib
########################

Magpylib provides fully tested and properly implemented analytical solutions to permanent magnet and current problems, which gives quick access to extremely fast magnetic field computation. Details on how these solutions are obtained can be found in the :ref:`physComp` section.

The core API is an object oriented interface that allows a user to create sensor and source objects with position and orientation in a global coordinate system. Magpylib objects can be easily created, manipulated, displayed, grouped and used for user-friendly field computation.

The central idea behind Magpylib is to provide convenient, high-performance magnetic field computations for everyone. Contrary to typical numerical algorithms (Finite Elements, Finite Differences, ...), the analytical solutions do not require meshing and boundary conditions (natural boundaries are assumed in the derivations). The solutions are exact, when there is no material response. There is no numerical error-propagation. The object oriented interface automatically vectorizes the code.


.. _docu-when_to_use:

When can you use Magpylib ?
###########################

The analytical solutions are exact when there is no material response. In permanent magnets, when (remanent) permeabilities are below :math:`\mu_r < 1.1` the error is typically below 5% (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. With these factors the precision can typically be incresed to below 1% error. Error estimation as a result of the materal response is evaluated in more detail in the appendix of `Malagò2020 <https://www.mdpi.com/1424-8220/20/23/6873>`_.

The line-current solutions give the exact same field as outside of a wire which carries a homogenous current.

Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials). For more details check out the :ref:`physComp` section.


.. _docu-layout_functionality:

Package layout and functionality overview
#########################################

In Magpylib everything revolves about the source and sensor objects. Magnet sources can be found in the sub-packages ``magpylib.magnet`` and include ``Cuboid``, ``Cylinder`` and ``Sphere``. Current sources are in the sub-package ``magpylib.current`` and include ``Line`` and ``Circular``. The Dipole source can be accessed as ``magpylib.misc.Dipole``. The sensor class lies at the top level ``magpylib.Sensor``

Magpylib source and sensor objects have a ``position`` `(ndarray, shape (m,3))` and ``orientation`` `(scipy Rotation object, shape (m,3))` attribute that describe how they are located in a global coordinate system. When `m>1`, position and orientation describe a path of the respective object. The ``move``, ``rotate`` and ``rotate_from_angax`` methods allow the user to geometrically manipulate the Magpylib objects and to conveniently generate complex paths.

The top level class ``magpylib.Collection`` allows a user to group sources for common manipulation. A Collection functions like a list of source objects extended by Magpylib source methods: all operations applied to a Collection are applied to each source individually. Specific sources in the Collection can still be accessed and manipulated individually.

When all source and sensor objects are created and all paths are defined the ``display`` function (top level) or method (all Magpylib objects) provides a convenient way to graphically view the geometric arrangement through Matplotlib.

The functions (top level) or methods (all Magpylib objects) ``getB`` and ``getH`` are used for magnetic field computation. This always requires ``sources`` and ``observers`` inputs. Sources are single Magpylib source objects, Collections or lists thereof.  Observers are arbitrary tensors of position vectors `(shape (n1,n2,n3,...,3))`, sensors or lists thereof. The output of the most general field computation through the top level function ``magpylib.getB(sources, observers)`` is an ndarray of shape `(l,m,k,n1,n2,n3,...,3)` where `l` is the number of input sources, `m` the pathlength, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or shape of position vector and `3` the three magnetic field components `(x,y,z)`. The B-field is computed in [mT], the H-field in [kA/m].

Finally the ``magpylib.getB_dict`` and ``magpylib.getHv`` functions give direct access to the analytical formulas implemented in Magpylib.

The functionality of Magpylib is demonstrated with several intuitive examples in the :ref:`examples` section. Details can be found in the library docstrings :ref:`genindex`.


.. _docu-performance:

Performance
###########

The analytical solutions provide extreme performance. Single field evaluations take of the order of `100 µs`. For large input arrays (e.g. many observer positions or many similar magnets) the computation time drops below `1 µs` on single state-of-the-art x86 mobile cores (tested on `Intel Core i5-8365U @ 1.60GHz`), depending on the source type.

The fastest way to compute fields is through the direct access to the top-level ``getB_dict`` and ``getHv`` functions. However, this requires the user to vectorize the input properly. The object-oriented interface automatically vectorizes the computation for the user (similar source types of multiple input-objects are grouped). The additional overhead makes the object-oriented interface slightly slower (by a factor of 1.1-2), specfically, when only single field evaluations are made (overhead gets in the way) or when a large number of source objects are handed to ``getB`` or ``getH``.


.. _docu-units_scaling:

Units and scaling property
##########################

Magpylib uses the following physical units:

- [mT]: for the B-field and the magnetization (µ0*M).
- [kA/m]: for the H-field.
- [mm]: for position and length inputs.
- [deg]: for angle inputs by default.
- [A]: for current inputs.

However, the analytical solutions scale in such a way that the magnetic field is the same when the system scales in size. This means that a 1-meter sized magnet in a distance of 1-meter produces the same field as a 1-millimeter sized magnet in a distance of 1-millimeter. The choice of position/length input dimension is therefore not relevant - the Magpylib choice of [mm] is a result of history and practical considerations (we mostly work with mm-sized systems :) ).

In addition, ``getB`` returns the unit of the input magnetization. The Magpylib choice of [mT] (theoretical physicists will point out that it is µ0*M) is historical and convenient. When the magnetization is given in [mT], then ``getH`` returns [kA/m] which is simply related by factor of `10/4pi`. Of course, ``getB`` also adds the magnet magnetization when computing the field inside the magnet, while ``getH`` does not.


.. _docu-close_to_surface:

Close to surfaces, edges and corners
####################################

Evaluation of analytical solutions are often limited by numerical precision when approaching singularities or indeterminate forms on magnet surfaces, edges or corners. 64-bit precision limits evaluation to 16 significant digits, but unfortunately many solutions include higher powers of the distances so that the precision limit is quickly approached.

As a result, Magpylib automatically sets solution that lie closer than ``magpylib.Config.EDGESIZE`` to problematic surfaces, edges or corners to 0. The default value is `1e-8`. The user can adjust this value simply with the command ``magpylib.Config.EDGESIZE=x``.


.. _docu-superposition_complex:

Superposition and complex shapes
################################

Magpylib only provides solutions for simple forms. However, in Magnetostatics the superposition principle holds: the total magnetic field is given by the (vector-)sum of all the fields of all sources. For magnets this means that complex magnet shapes can be constructed from simple forms. Specifically, it is possible to cut-out a part of a magnet simply by placing a second magnet with opposite magnetization inside the first magnet.
