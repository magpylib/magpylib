(guide-ressources-physics)=
# Physics and Computation

## Analytical Solutions

Magnetic field computations in Magpylib are based on known analytical solutions (closed form expressions) to permanent magnet and current problems. The Magpylib implementations are based on the following literature references:

- Field of cuboid magnets {cite}`yangt1990potential, engel2005calculation, camacho2013alternative, cichon2019hall`
- Field of cylindrical magnets: {cite}`furlani2002three, derby2010cylindrical, caciagli2018exact, slanovc2022full`
- Field of triangular surface charges: {cite}`guptasarma1999new, janssen20103d, rubeck2012analytical`
- Field of the current loop: {cite}`ortner2022numerically`
- all others derived by hand

A short reflection on how these formulas can be achieved: In magnetostatics the magnetic field becomes conservative (Maxwell: $\nabla \times {\bf H} = 0$) and can thus be expressed through the magnetic scalar potential $\Phi_m$:

$$
{\bf H} = -\nabla\cdot\Phi_m
$$

The solution to this equation can be expressed by an integral over the magnetization distribution ${\bf M}({\bf r})$ as

$$
\Phi_m({\bf r}) = \frac{1}{4\pi}\int_{V'}\frac{\nabla'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dV'+\frac{1}{4\pi}\oint_{S'}\frac{{\bf n}'\cdot {\bf M}({\bf r}')}{|{\bf r}-{\bf r}'|}dS'
$$

where ${\bf r}$ denotes the position, $V$ is the magnetized volume with surface $S$ and normal vector ${\bf n}$ onto the surface. This solution is derived in detail e.g. in {cite}`jackson1999classical`.

The fields of currents are directly derived using the law of Biot-Savart with the current distribution ${\bf J}({\bf r})$:

$$
{\bf B}({\bf r}) = \frac{\mu_0}{4\pi}\int_{V'} {\bf J}({\bf r}')\times \frac{{\bf r}-{\bf r}'}{|{\bf r}-{\bf r}'|^3} dV'
$$

In some special cases (simple shapes, homogeneous magnetisations, and current distributions) the above integrals can be worked out directly to give analytical formulas (or simple, fast converging series). The derivations can be found in the respective references. A noteworthy comparison between the Coulombian approach and the Amperian current model is given in {cite}`ravaud2009comparison`.


------------------------------------

(guide-physics-demag)=
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

- Even with perfect initial magnetization, [material response](https://en.wikipedia.org/wiki/Demagnetizing_field) creates inhomogeneous magnetization distributions. However, modern high-grade permanent magnets (NdFeB, SmCo, Ferrite) have very weak material responses (remanent permeability) as low as $\mu_r \approx 1.05$. A detailed demagnetization error analysis for cuboids is presented in the appendix of {cite}`malago2020magnetic`.

**Advanced modeling techniques**:

- Demagnetization factors can be used to correct for self-interaction effects. Analytical expressions for cuboid demagnetization factors are available at [magpar.net](http://www.magpar.net/static/magpar/doc/html/demagcalc.html).

- The [Magpylib material response](https://github.com/magpylib/magpylib-material-response) extension enables modeling of self-interactions and inhomogeneous magnetization distributions. It also supports **soft-magnetic** materials with large permeabilities ($\mu_r \sim 1000$) and low remanence.

- The [Method of Images](examples-misc-image-method) can be used to model magnets near soft-magnetic plates, demonstrated impressively in the [Holding Force Example](examples-force-holding-force).

- Consult the [Tutorial on Modeling Datasheet Magnets](examples-tutorial-modeling-magnets) for practical guidance on handling demagnetization effects.

------------------------------------

(docu-performance)=
## Computation Performance

Magpylib code is fully [vectorized](https://en.wikipedia.org/wiki/Array_programming), written almost completely in NumPy native. Magpylib automatically vectorizes computations with complex inputs (many sources, many observers, paths) and never falls back on using loops.

```{note}
Maximal performance is achieved when `getB()` is called only a single time in your program. Try not to use loops --- unless you run out of memory.
```

The object-oriented interface comes with an overhead. If you want to achieve maximal performance this overhead can be avoided with the {ref}`docs-field-functional`.

The analytical solutions provide extreme performance. Single field evaluations take of the order of 100 µs. For large input arrays (e.g. many observer positions or many similar magnets) the computation time can drop below 1 µs per evaluation point on single state-of-the-art x86 mobile cores (tested on *Intel Core i5-8365U @ 1.60GHz*), depending on the expression complexity (e.g. Dipole is fast, CylinderSegment is slow).

------------------------------------

## Numerical Stability

Many expressions provided in the literature have very questionable numerical stability, meaning that naive, straight-forward implementation of the provided expression results in a low number of significant digits about singular evaluation points {cite}`ortner2022numerically`. Many of these problems are fixed in Magpylib, but one should be aware that accuracy can be problematic
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

```{bibliography}
:filter: docname in docnames
:labelprefix: P1-
```
