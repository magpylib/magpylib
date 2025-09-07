(guide-ressources-physics)=
# Physics and Computation

## Analytical Solutions

Magnetic field computations in Magpylib are based on known analytical solutions (closed form expressions) to permanent magnet and current problems. The Magpylib implementations are based on the following literature references:

- Field of cuboid magnets [^1] [^2] [^3] [^4]
- Field of cylindrical magnets: [^5] [^6] [^7] [^8]
- Field of triangular surface charges: [^9] [^10] [^11]
- Field of the current loop: [^12]
- all others derived by hand

A short reflection on how these formulas can be achieved: In magnetostatics the magnetic field becomes conservative (Maxwell: $\nabla \times {\bf H} = 0$) and can thus be expressed through the magnetic scalar potential $\Phi_m$:

$$
{\bf H} = -\nabla\cdot\Phi_m
$$

The solution to this equation can be expressed by an integral over the magnetization distribution ${\bf M}({\bf r})$ as

$$
\Phi_m({\bf r}) = \frac{1}{4\pi}\int_{V'}\frac{\nabla'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dV'+\frac{1}{4\pi}\oint_{S'}\frac{{\bf n}'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dS'
$$

where ${\bf r}$ denotes the position, $V$ is the magnetized volume with surface $S$ and normal vector ${\bf n}$ onto the surface. This solution is derived in detail e.g. in [^13].

The fields of currents are directly derived using the law of Biot-Savart with the current distribution ${\bf J}({\bf r})$:

$$
{\bf B}({\bf r}) = \frac{\mu_0}{4\pi}\int_{V'} {\bf J}({\bf r}')\times \frac{{\bf r}-{\bf r}'}{|{\bf r}-{\bf r}'|^3} dV'
$$

In some special cases (simple shapes, homogeneous magnetisations, and current distributions) the above integrals can be worked out directly to give analytical formulas (or simple, fast converging series). The derivations can be found in the respective references. A noteworthy comparison between the Coulombian approach and the Amperian current model is given in [^14].

(guide-physics-demag)=

------------------------------------

## Accuracy of the Solutions

While Magpylib's analytical solutions are exact, the idealized assumptions of perfectly uniform current and magnetization distributions rarely match real-world conditions. Understanding these limitations is crucial for interpreting computational results and assessing their practical applicability.

**Currents**

The magnetic field of a wire carrying homogeneous current density is (on the outside) equivalent to that of a line current positioned at the wire's center carrying the same total current. However, real current distributions deviate from this ideal in several scenarios:

- Wire bends and corners result in non-uniform current densities.
- AC effects like the [Skin effect](https://en.wikipedia.org/wiki/Skin_effect) and the [Proximity effect](https://en.wikipedia.org/wiki/Proximity_effect_(electromagnetism)) redistribute currents at finite frequencies.
- Temperature and material parameter gradients result in local conductivity variations causing localized current density changes.

**Magnets**

Real permanent magnets exhibit several deviations from the ideal uniform magnetization assumption:

- Manufacturing variations of off-the-shelf permanent magnets typically show variations of a few percent in magnetization amplitude and several degrees in magnetization direction, resulting from imperfect magnetization processes and material property variations.

- Even with perfect initial magnetization, [material response](https://en.wikipedia.org/wiki/Demagnetizing_field) creates inhomogeneous magnetization distributions. However, modern high-grade permanent magnets (NdFeB, SmCo, Ferrite) have very weak material responses (remanent permeability) as low as $\mu_r \approx 1.05$. A detailed demagnetization error analysis for cuboids is presented in the appendix of [^15].

**Advanced modeling techniques**:

- Demagnetization factors can be used to correct for self-interaction effects. Analytical expressions for cuboid demagnetization factors are available at [magpar.net](http://www.magpar.net/static/magpar/doc/html/demagcalc.html).

- The [Magpylib material response](https://github.com/magpylib/magpylib-material-response) extension enables modeling of self-interactions and inhomogeneous magnetization distributions. It also supports **soft-magnetic** materials with large permeabilities ($\mu_r \sim 1000$) and low remanence.

- The [Method of Images](examples-misc-image-method) can be used to model magnets near soft-magnetic plates, demonstrated impressively in the [Holding Force Example](examples-force-holding-force).

- Consult the [Tutorial on Modeling Datasheet Magnets](examples-tutorial-modeling-magnets) for practical guidance on handling demagnetization effects.

------------------------------------

(docu-performance)=
## Computation Performance

Magpylib code is fully [vectorized](https://en.wikipedia.org/wiki/Array_programming), written almost completely in numpy native. Magpylib automatically vectorizes computations with complex inputs (many sources, many observers, paths) and never falls back on using loops.

```{note}
Maximal performance is achieved when `.getB(sources, observers)` is called only a single time in your program. Try not to use loops --- unless you run out of memory.
```

The object-oriented interface comes with an overhead. If you want to achieve maximal performance this overhead can be avoided with the {ref}`docs-field-functional`.

The analytical solutions provide extreme performance. Single field evaluations take of the order of `100 µs`. For large input arrays (e.g. many observer positions or many similar magnets) the computation time can drop below `1 µs` per evaluation point on single state-of-the-art x86 mobile cores (tested on `Intel Core i5-8365U @ 1.60GHz`), depending on the expression complexity (e.g. Dipole is fast, CylinderSegment is slow).

------------------------------------

## Numerical Stability

Many expressions provided in the literature have very questionable numerical stability, meaning that naive, straight-forward implementation of the provided expression results in a low number of significant digits about singular evaluation points [^12]. Many of these problems are fixed in Magpylib, but one should be aware that accuracy can be problematic
- very close to objects, specifically edges and corners
- close the z-axis in cylindrical symmetries
- at edge extensions
- at very large distances (>100 x object size)

We are working on fixing these problems.

------------------------------------

(guide-physics-force-computation)=
## Force Computation

### Force Equations

The force $\vec{F}_\text{tot}$ and torque $\vec{T}_\text{tot}$ acting on macroscopic magnetic and current-carrying objects in external magnetic fields $\vec{B}(\vec{r})$ are governed by fundamental electromagnetic principles.

For **magnets** defined by a magnetization distribution $\vec{M}(\vec{r})$ we have:

$$\vec{F}_\text{tot} = \int \nabla (\vec{M}(\vec{r})\cdot\vec{B}(\vec{r})) \ d^3 r.$$

$$\vec{T}_\text{tot} = \int \vec{M}(\vec{r}) \times \vec{B}(\vec{r}) \ d^3r + \int (\vec{r} - \vec{r}_\text{piv}) \times \vec{F}(\vec{r}) \ d^3r.$$

For **current-carrying objects** defined by a current distribution $\vec{j}(\vec{r})$ we have:

$$\vec{F}_\text{tot} = \int \vec{j}(\vec{r})\times \vec{B}(\vec{r}) \ d^3r.$$
$$\vec{T}_\text{tot} = \int (\vec{r} - \vec{r}_\text{piv}) \times \vec{F}(\vec{r}) \ d^3r$$

Here $\vec{r}_\text{piv}$ denotes a pivot point about which the object rotates. This is the center of mass when the object is floating freely.

### Computational Approach

In contrast to magnetic field computation the idea behind Magpylib force computation is working out the above integrals by numerical discretization. For this purpose the target bodies are split up into small cells and force and torque calculations are performed simultaneously for all cells using vectorized operations. For magnetized objects, the required magnetic field gradient $\nabla\vec{B}$ is computed using a finite difference scheme. This approach prioritizes computational speed through vectorization at the cost of higher memory usage.

**Computation scheme:**

1. **Mesh preparation:** Generate mesh points (OBS) and compute cell properties
   - Current vectors (CVEC) for current-carrying objects
   - Magnetic moments (MOM) for magnet and `Dipole` objects

2. **B-field evaluation:** The B-field (B) is evaluated at all mesh points in a single vectorized operation, including the 6 additional points needed for finite difference gradient calculation (DB) for magnets.

3. **Force computation:**
   - Magnets: F = DB.MOM (dipole force)
   - Currents: F = CVEC x B (Lorentz force)

4. **Torque computation:**
   - Magnets: T = MOM x B (intrinsic magnetic torque)
   - Both: T += (OBS-PIV) x F (force moment)

5. **Integration/Reduction:** We now have computed all forces and torques on each mesh cell. The final step is summation of contributions from all mesh cells of each target, and all targets of each collection.

The computation is most accurate when the mesh is uniform with cell aspect ratios of 1 (the ideal cell is a sphere), which is what the meshing algorithms are trying to achieve.

------------------------------------

## References

[^1]: Z. J. Yang et al., "Potential and force between a magnet and a bulk Y1Ba2Cu3O7-d superconductor studied by a mechanical pendulum", Superconductor Science and Technology 3(12):591, 1990

[^2]: R. Engel-Herbert et al., Journal of Applied Physics 97(7):074504 - 074504-4 (2005)

[^3]: J.M. Camacho and V. Sosa, "Alternative method to calculate the magnetic field of permanent magnets with azimuthal symmetry", Revista Mexicana de Fisica E 59 8–17, 2013

[^4]: D. Cichon, R. Psiuk and H. Brauer, "A Hall-Sensor-Based Localization Method With Six Degrees of Freedom Using Unscented Kalman Filter", IEEE Sensors Journal, Vol. 19, No. 7, April 1, 2019.

[^5]: E. P. Furlani, S. Reanik and W. Janson, "A Three-Dimensional Field Solution for Bipolar Cylinders", IEEE Transaction on Magnetics, VOL. 30, NO. 5, 1994

[^6]: N. Derby, "Cylindrical Magnets and Ideal Solenoids", arXiv:0909.3880v1, 2009

[^7]: A. Caciagli, R. J. Baars, A. P. Philipse and B. W. M. Kuipers, "Exact expression for the magnetic field of a finite cylinder with arbitrary uniform magnetization", Journal of Magnetism and Magnetic Materials 456 (2018) 423–432.

[^8]: F. Slanovc, M. Ortner, M. Moridi, C. Abert and D. Suess, "Full analytical solution for the magnetic field of uniformly magnetized cylinder tiles", submitted to Journal of Magnetism and Magnetic Materials.

[^9]: D. Guptasarma and B. Singh, "New scheme for computing the magnetic field resulting from a uniformly magnetized arbitrary polyhedron", Geophysics (1999), 64(1):70.

[^10]: J.L.G. Janssen, J.J.H. Paulides and E.A. Lomonova, "3D ANALYTICAL FIELD CALCULATION USING TRIANGULAR MAGNET SEGMENTS APPLIED TO A SKEWED LINEAR PERMANENT MAGNET ACTUATOR", ISEF 2009 - XIV International Symposium on Electromagnetic Fields in Mechatronics, Electrical and Electronic Engineering Arras, France, September 10-12, 2009

[^11]: C. Rubeck et al., "Analytical Calculation of Magnet Systems: Magnetic Field Created by Charged Triangles and Polyhedra", IEEE Transactions on Magnetics, VOL. 49, NO. 1, 2013

[^12]: M. Ortner, S. Slanovc and P. Leitner, "Numerically Stable and Computationally Efficient Expression for the Magnetic Field of a Current Loop", MDPI Magnetism, 3(1), 11-31, 2022.

[^13]: J. D. Jackson, "Classical Electrodynamics", 1999 Wiley, New York

[^14]: R. Ravaud and G. Lamarquand, "Comparison of the coulombian and amperian current models for calculating the magnetic field produced by radially magnetized arc-shaped permanent magnets", HAL Id: hal-00412346

[^15]: P. Malagò et al., Magnetic Position System Design Method Applied to Three-Axis Joystick Motion Tracking. Sensors, 2020, 20. Jg., Nr. 23, S. 6873.
