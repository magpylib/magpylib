# Units and Types

(guide-docs-units)=
## Units

The important vacuum permeability $\mu_0$ is provided at the package top-level <span style="color: orange">**mu_0**</span>. It's value is not $4 \pi 10^{-7}$ since [the redefinition of the SI base units](https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units), but a value close to it.

For historical reasons Magpylib used non-SI units until Version 4. Starting with version 5 all inputs and outputs are SI-based.

::::{grid} 3
:::{grid-item}
:columns: 1
:::

:::{grid-item}
:columns: 10
| PHYSICAL QUANTITY | MAGPYLIB PARAMETER | UNITS from v5| UNITS until v4|
|:---:|:---:|:---:|:---:|
| Magnetic Polarization $\vec{J}$  | `polarization`, `getJ()`      | **T**      | -        |
| Magnetization $\vec{M}$          | `magnetization`, `getM()`     | **A/m**    | mT       |
| Electric Current $i_0$           | `current`                     | **A**      | A        |
| Magnetic Dipole Moment $\vec{m}$ | `moment`                      | **A·m²**   | mT·mm³   |
| B-field $\vec{B}$                | `getB()`                      | **T**      | mT       |
| H-field $\vec{H}$                | `getH()`                      | **A/m**    | kA/m     |
| Length-inputs                    | `position`, `dimension`, `vertices`, ...  | **m**      | mm       |
| Angle-inputs                     | `angle`, `dimension`, ...     | **°**      | °        |
:::

::::

```{warning}
Up to version 4, Magpylib was unfortunately contributing to the naming confusion in magnetism that is explained well [here](https://www.e-magnetica.pl/doku.php/confusion_between_b_and_h). The input `magnetization` in Magpylib < v5 was referring to the magnetic polarization (and not the magnetization), the difference being only in the physical unit. From version 5 onwards this is fixed.
```

```{note}
The connection between the magnetic polarization J, the magnetization M and the material parameters of a real permanent magnet are shown in {ref}`examples-tutorial-modelling-magnets`.
```

(guide-docs-io-scale-invariance)=
## Arbitrary unit Convention

```{hint}
All input and output units in Magpylib (version 5 and higher) are SI-based, see table above. However, for advanced use one should be aware that the analytical solutions are **scale invariant** - _"a magnet with 1 mm sides creates the same field at 1 mm distance as a magnet with 1 m sides at 1 m distance"_. The choice of length input unit is therefore not relevant, but it is critical to keep the same length unit for all inputs in one computation.

In addition, `getB` returns the same unit as given by the `polarization` input. With polarization input in mT, `getB` will return mT as well. At the same time when the `magnetization` input is kA/m, then `getH` returns kA/m as well. The B/H-field outputs are related to a M/J-inputs via a factor of $µ_0$.
```

## Types

Magpylib requires no special input format. All scalar types (`int`, `float`, ...) and vector types (`list`, `tuple`, `np.ndarray`, ... ) are accepted. Magpylib returns everything as `np.ndarray`.