.. _physComp:

***************************
[WIP] Physics & Computation
***************************

The analytical solutions
########################

Magnetic field computations in Magpylib are based on known analytical solutions (formulas) to permanent magnet and current problems that can be found in the literature. The magnet solutions are mostly derived via the magnetic scalar potential, the current solutions through the law of Biot-Savart.

In magnetostatics (no currents) the magnetic field becomes conservative (Maxwell: :math:`\nabla \times {\bf H} = 0`) and can thus be expressed through the magnetic scalar potential :math:`\Phi_m`:

.. math::

    {\bf H} = -\nabla\cdot\Phi_m

The solution to this equation can be expressed by an integral over the magnetization distribution :math:`{\bf M}` as

.. math::

    \Phi_m({\bf r}) = \frac{1}{4\pi}\int_{V'}\frac{\nabla'\cdot {\bf M}}{|{\bf r}-{\bf r}'|}dV'+\frac{1}{4\pi}\oint_{S'}\frac{\n'\cdot {\bf M}}{|{\bf r}-{\bf r}'|}dS'

where :math:`{\bf r}` denotes the position, :math:`V` is the magnetized volume with surface :math:`S` and normal vector :math:`{\bf n}` onto the surface. This solution is derived in detail e.g. in [Jackson]. In some special cases this integral can be worked out directly to give a simple analytical formula. These cases refer to homogeneous magnetizations and simple magnet shapes.

For magpylib we have used the following references:

* Field of cuboid magnets: [1999Johansen, 2013Camacho]
* Field of cylindrical magnets: [1994Furlani, 2009Derby]
* Field of facet bodies: [2009Janssen, 2013Rubeck]
* all others simply derived by hand

Hard and soft magnetic materials
--------------------------------
We cannot do soft, theres a lot of reasons we we dont need to

Demagnetization
---------------

Solution accuracy, analytical modeling of demagnetization and interaction

multiple sources, no interaction

Limits and Convergence
######################

Convergence of diametral Cylinder
No. iterations

Computation
###########

SIMD

vectorized code

performance tests

parallelization

