.. _physComp:

***************************
[WIP] Physics & Computation
***************************

The analytical solutions
########################

Magnetic field computations in Magpylib are based on known analytical solutions (formulas) to permanent magnet and current problems that can be found in the literature. For magpylib we have used the following references:

* Field of cuboid magnets: [1999Johansen, 2013Camacho]
* Field of cylindrical magnets: [1994Furlani, 2009Derby]
* Field of facet bodies: [2009Janssen, 2013Rubeck]
* all others simply derived by hand

Subsequently we show a short reflection on how these formulas are achieved. The magnet solutions are mostly derived via the magnetic scalar potential. In magnetostatics (no currents) the magnetic field becomes conservative (Maxwell: :math:`\nabla \times {\bf H} = 0`) and can thus be expressed through the magnetic scalar potential :math:`\Phi_m`:

.. math::

    {\bf H} = -\nabla\cdot\Phi_m

The solution to this equation can be expressed by an integral over the magnetization distribution :math:`{\bf M}` as

.. math::

    \Phi_m({\bf r}) = \frac{1}{4\pi}\int_{V'}\frac{\nabla'\cdot {\bf M}}{|{\bf r}-{\bf r}'|}dV'+\frac{1}{4\pi}\oint_{S'}\frac{{\bf n}'\cdot {\bf M}}{|{\bf r}-{\bf r}'|}dS'

where :math:`{\bf r}` denotes the position, :math:`V` is the magnetized volume with surface :math:`S` and normal vector :math:`{\bf n}` onto the surface. This solution is derived in detail e.g. in [1999Jackson].

The current solutions are directly derived using the law of Biot-Savart:

.. math::

    {\bf B}({\bf r}) = \frac{\mu_0}{4\pi}\int_{V'} {\bf J}({\bf r}')\times \frac{{\bf r}-{\bf r}'}{|{\bf r}-{\bf r}'|^3} dV'

In some special cases (simple shapes and homogeneous magnetizations and current distributions) these integrals can be worked out directly to give analytical formulas (or simple, fast converging series). The derivations can be found in the respective references.

References:
* [1999Johansen] T. H. Johansen et al., "Potential and force between a magnet and a bulk Y1Ba2Cu3O7-? superconductor studied by a mechanical pendulum", Superconductor Science and Technology 3(12):591, 1999
* [2013 Camacho] J.M. Camacho and V. Sosa, "Alternative method to calculate the magnetic field of permanent magnets with azimuthal symmetry", Revista Mexicana de Fisica E 59 8–17, 2013
* [1994Furlani] E. P. Furlani, S. Reanik and W. Janson, "A Three-Dimensional Field Solution for Bipolar Cylinders", IEEE Transaction on Magnetics, VOL. 30, NO. 5, 1994
* [2009Derby] N. Derby, "Cylindrical Magnets and Ideal Solenoids", arXiv:0909.3880v1, 2009
* [2009Janssen] J.L.G. Janssen, J.J.H. Paulides and E.A. Lomonova, "3D ANALYTICAL FIELD CALCULATION USING TRIANGULAR MAGNET SEGMENTS APPLIED TO A SKEWED LINEAR PERMANENT MAGNET ACTUATOR", ISEF 2009 - XIV International Symposium on Electromagnetic Fields in Mechatronics, Electrical and Electronic Engineering Arras, France, September 10-12, 2009
* [2013Rubeck] C. Rubeck et al., "Analytical Calculation of Magnet Systems: Magnetic Field Created by Charged Triangles and Polyhedra", IEEE Transactions on Magnetics, VOL. 49, NO. 1, 2013
* [1999Jackson] J. D. Jackson, "Classical Electrodynamics", 1999 Wiley, New York


Accuracy of the Solutions and Demagnetization
#############################################

Solution accuracy, analytical modeling of demagnetization and interaction

multiple sources, no interaction

Convergence of diametral Cylinder
No. iterations

Computation
###########

SIMD

vectorized code

performance tests

parallelization

