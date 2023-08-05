---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(docu-physics)=

# Physics

## When can you use Magpylib ?

The expressions used in Magpylib describe perfectly homogeneous magnets, surface charges, and line currents with natural boundary conditions. Magpylib is at its best when dealing with static air-coils (no eddy currents, no soft-magnetic cores) and high grade permanent magnets (Ferrite, NdFeB, SmCo or similar materials). When **magnet** permeabilities are below $\mu_r < 1.1$ the error typically undercuts few %. Demagnetization factors are not included. The line **current** solutions give the exact same field as outside of a wire that carries a homogeneous current.

## The analytical solutions

Magnetic field computations in Magpylib are based on known analytical solutions (closed form expressions) to permanent magnet and current problems. The Magpylib implementations are based on the following literature references:

- Field of cuboid magnets: \[Yang1999, Engel-Herbert2005, Camacho2013, Cichon2019\]
- Field of cylindrical magnets: \[Furlani1994, Derby2009, Caciagli2018, Slanovc2021\]
- Field of triangular surfaces: \[Guptasarma1999, Janssen2009, Rubeck2013\]
- Field of the current loop: \[Ortner2022\]
- all others derived by hand

A short reflection on how these formulas can be achieved: In magnetostatics the magnetic field becomes conservative (Maxwell: $\nabla \times {\bf H} = 0$) and can thus be expressed through the magnetic scalar potential $\Phi_m$:

$$
{\bf H} = -\nabla\cdot\Phi_m
$$

The solution to this equation can be expressed by an integral over the magnetization distribution ${\bf M}({\bf r})$ as

$$
\Phi_m({\bf r}) = \frac{1}{4\pi}\int_{V'}\frac{\nabla'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dV'+\frac{1}{4\pi}\oint_{S'}\frac{{\bf n}'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dS'
$$

where ${\bf r}$ denotes the position, $V$ is the magnetized volume with surface $S$ and normal vector ${\bf n}$ onto the surface. This solution is derived in detail e.g. in \[Jackson1999\].

The fields of currents are directly derived using the law of Biot-Savart with the current distribution ${\bf J}({\bf r})$:

$$
{\bf B}({\bf r}) = \frac{\mu_0}{4\pi}\int_{V'} {\bf J}({\bf r}')\times \frac{{\bf r}-{\bf r}'}{|{\bf r}-{\bf r}'|^3} dV'
$$

In some special cases (simple shapes, homogeneous magnetizations and current distributions) the above integrals can be worked out directly to give analytical formulas (or simple, fast converging series). The derivations can be found in the respective references. A noteworthy comparison between the Coulombian approach and the Amperian current model is given in \[Ravaud2009\].

## Accuracy of the Solutions and Demagnetization

### Line currents:

The magnetic field of a wire carrying a homogeneous current density is (on the outside) similar to the field of a line current positioned in the center of the wire, which carries the same total current. Current distributions become inhomogeneous at bends of the wire or when eddy currents (finite frequencies) are involved.

### Magnets and Demagnetization

The analytical solutions are exact when bodies have a homogeneous magnetization. However, real materials always have a material response which results in an inhomogeneous magnetization even when the initial magnetization is perfectly homogeneous. There is a lot of literature on such [demagnetization effects](https://en.wikipedia.org/wiki/Demagnetizing_field).

Modern high grade permanent magnets (NdFeB, SmCo, Ferrite) have a very weak material responses (local slope of the magnetization curve, remanent permeability) of the order of $\mu_r \approx 1.05$. In this case the analytical solutions provide an excellent approximation with less than 1% error even at close distance from the magnet surface. A detailed error analysis and discussion is presented in the appendix of \[Malago2020\].

Error estimation as a result of the material response is evaluated in more detail in the appendix of [Malagò2020](https://www.mdpi.com/1424-8220/20/23/6873).

Demagnetization factors can be used to compensate a large part of the demagnetization effect. Analytical expressions for the demagnetization factors of cuboids can be found at [magpar.net](http://www.magpar.net/static/magpar/doc/html/demagcalc.html).

(phys-remanence)=
### Modelling a datasheet magnet

The material remanence, often found in data sheets, simply corresponds to the material magnetization/polarization when not under the influence of external fields. This can never happen, as the material itself generates a magnetic field. Such self-interactions result in self-demagnetization that can be approximated using the demagnetization factors and the material permeability (or susceptibility).

For example, a cube with 1 mm sides has a demagnetization factor is 0.333, see [magpar.net](http://www.magpar.net/static/magpar/doc/html/demagcalc.html). When the remanence field of this cube is 1 T, and its susceptibility is 0.1, the magnetization resulting from self-interaction is reduced to 1 T - 0.3333*0.1 T = 0.9667 T, assuming linear material laws.

It must be understood that the change in magnetization resulting from self-interaction has a homogeneous contribution which is approximated by the demagnetization factor, and an inhomogeneous contribution which cannot be modeled easily with analytical solutions. The inhomogeneous part, however, is typically an order of magnitude lower than the homogenous part. You can use the Magpylib extension [Magpylib material response](https://github.com/magpylib/magpylib-material-response) to model the self-interactions.

### Soft-Magnetic Materials

Soft-magnetic materials like iron or steel with large permeabilities $\mu_r \sim 1000$ and low remanence fields are dominated by the material response. It is not possible to describe such bodies with analytical solutions. However, recent developments showed that the Magnetostatic Method of Moments can be a powerful tool in combination with Magpylib to compute such a material response. An integration into Magpylib is planned for the future.


(docu-performance)=
## Computation and Performance

Magpylib code is fully [vectorized](https://en.wikipedia.org/wiki/Array_programming), written almost completely in numpy native. Magpylib automatically vectorizes the computation with complex inputs (many sources, many observers, paths) and never falls back on using loops.

```{note}
Maximal performance is achieved when `.getB(sources, observers)` is called only a single time in your program. Try not to use loops.
```

The object oriented interface comes with an overhead. If you want to achieve maximal performance this overhead can be avoided with {ref}`docu-direct-interface`.

The analytical solutions provide extreme performance. Single field evaluations take of the order of `100 µs`. For large input arrays (e.g. many observer positions or many similar magnets) the computation time drops below `1 µs` per evaluation point on single state-of-the-art x86 mobile cores (tested on `Intel Core i5-8365U @ 1.60GHz`), depending on the source type.

## Numerical stability

Many expressions provided in the literature have very questionable numerical stability. Many of these problems are fixed in Magpylib, but one should be aware that accuracy can be a problem very close to objects, close the z-axis in cylindrical symmetries, at edge extensions, and at large distances. We are working on fixing these problems. Some details can be found in \[Ortner2022\].

**References**

- \[Yang1999\] Z. J. Yang et al., "Potential and force between a magnet and a bulk Y1Ba2Cu3O7-d superconductor studied by a mechanical pendulum", Superconductor Science and Technology 3(12):591, 1999
- \[Engel-Herbert2005\] R. Engel-Herbert et al., Journal of Applied Physics 97(7):074504 - 074504-4 (2005)
- \[Camacho2013\] J.M. Camacho and V. Sosa, "Alternative method to calculate the magnetic field of permanent magnets with azimuthal symmetry", Revista Mexicana de Fisica E 59 8–17, 2013
- \[Cichon2019\] D. Cichon, R. Psiuk and H. Brauer, "A Hall-Sensor-Based Localization Method With Six Degrees of Freedom Using Unscented Kalman Filter", IEEE Sensors Journal, Vol. 19, No. 7, April 1, 2019.
- \[Furlani1994\] E. P. Furlani, S. Reanik and W. Janson, "A Three-Dimensional Field Solution for Bipolar Cylinders", IEEE Transaction on Magnetics, VOL. 30, NO. 5, 1994
- \[Derby2009\] N. Derby, "Cylindrical Magnets and Ideal Solenoids", arXiv:0909.3880v1, 2009
- \[Caciagli2018\] A. Caciagli, R. J. Baars, A. P. Philipse and B. W. M. Kuipers, "Exact expression for the magnetic field of a finite cylinder with arbitrary uniform magnetization", Journal of Magnetism and Magnetic Materials 456 (2018) 423–432.
- \[Slanovc2022\] F. Slanovc, M. Ortner, M. Moridi, C. Abert and D. Suess, "Full analytical solution for the magnetic field of uniformly magnetized cylinder tiles", submitted to Journal of Magnetism and Magnetic Materials.
- \[Ortner2022\] M. Ortner, S. Slanovc and P. Leitner, "Numerically Stable and Computationally Efficient Expression for the Magnetic Field of a Current Loop", MDPI Magnetism, 3(1), 11-31, 2022.
- \[Guptasarma1911\] D. Guptasarma and B. Singh, "New scheme for computing the magnetic field resulting from a uniformly magnetized arbitrary polyhedron", Geophysics (1999), 64(1):70.
- \[Janssen2009\] J.L.G. Janssen, J.J.H. Paulides and E.A. Lomonova, "3D ANALYTICAL FIELD CALCULATION USING TRIANGULAR MAGNET SEGMENTS APPLIED TO A SKEWED LINEAR PERMANENT MAGNET ACTUATOR", ISEF 2009 - XIV International Symposium on Electromagnetic Fields in Mechatronics, Electrical and Electronic Engineering Arras, France, September 10-12, 2009
- \[Rubeck2013\] C. Rubeck et al., "Analytical Calculation of Magnet Systems: Magnetic Field Created by Charged Triangles and Polyhedra", IEEE Transactions on Magnetics, VOL. 49, NO. 1, 2013
- \[Jackson1999\] J. D. Jackson, "Classical Electrodynamics", 1999 Wiley, New York
- \[Ravaud2009\] R. Ravaud and G. Lamarquand, "Comparison of the coulombian and amperian current models for calculating the magnetic field produced by radially magnetized arc-shaped permanent magnets", HAL Id: hal-00412346
- \[Malago2020\] P. Malagò et al., Magnetic Position System Design Method Applied to Three-Axis Joystick Motion Tracking. Sensors, 2020, 20. Jg., Nr. 23, S. 6873.
