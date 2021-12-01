All notable changes to magpylib are documented here.


# Releases

## [Unreleased]
### Added
- Documentation and Example codes now available on read the docs. [#389](https://github.com/magpylib/magpylib/issues/389)
- New object classes:
  - New `CylinderSegment` computation with sector angles `(r1,r2,h,phi1,phi2)`. [#386](https://github.com/magpylib/magpylib/issues/386) [#385](https://github.com/magpylib/magpylib/issues/385)
  - New `CustomSource`class for user defined field functions [#349](https://github.com/magpylib/magpylib/issues/349) [#409](https://github.com/magpylib/magpylib/issues/409) [#411](https://github.com/magpylib/magpylib/pull/411)
- New display features:
  - Display defaults can be now set in `magpylib.defaults.display`[#291](https://github.com/magpylib/magpylib/issues/291)
  - Ability to define extra 3d-model for any object. [#416](https://
  github.com/magpylib/magpylib/pull/416)
  - Zoom option in display function.
  - Display specific path positions.
  - `matplotlib` plotting backend:
    - Added `matplotlib` pixel display [#279](https://github.com/magpylib/magpylib/issues/279)
  - 🚀 New `plotly` plotting backend: [#396](https://github.com/magpylib/magpylib/pull/396) [#353](https://github.com/magpylib/magpylib/issues/353)
    - `plotly` remains a optional dependency, needs extra installation [#395](https://github.com/magpylib/magpylib/issues/395)
    - Interactive `animation` feature with `animation_time` and `animation_fps` `display` arguments


### Changed
- `Box` class renamed to `Cuboid`. [#350](https://github.com/magpylib/magpylib/issues/350)
- `Circular` class renamed to `Loop`. [#402](https://github.com/magpylib/magpylib/pull/402)
- `magpylib.defaults.edgesize` set to `1e-8` by default to avoid problems in `Cuboid` corners.
- Magpylib objects can now be _initialized_ **without excitation** and **without dimension** attributes.
- `Collection` can contain `sources`, `sensors` or `both`. The `getB` and `getH` functions accommodate for all cases. [#410](https://github.com/magpylib/magpylib/issues/410) [#415](https://github.com/magpylib/magpylib/pull/415) [#297](https://github.com/magpylib/magpylib/issues/297)
- `magpylib.Config` parameters are now in `magpylib.defaults` [#387](https://github.com/magpylib/magpylib/issues/387)
- `getBv` and `getHv` are renamed to `getB_dict` and `getH_dict` [#294](https://github.com/magpylib/magpylib/issues/294)
- renamed `display` arguments
  - `axis` ➡️ `canvas`
  - `show_direction` ➡️ `style_magnetization_show`
  - `show_path` ➡️ `path`
  - `size_sensors`&`size_dipoles` ➡️ `style_size`
  - `size_direction` ➡️ `style_magnetization_size`

### Updated
- Computation:
  - Updated `Cylinder` computation with a new exact closed form implementation without iteration, from a new paper (F.Slanovc, preprint-2021). Computation times are around 50-100 µs.with some performance improvement planned in the future.
  - Added `__lend__` dunder for `Collection` [#383](https://github.com/magpylib/magpylib/issues/383)
  - Improved performance of `getB` and `getH` functions and methods.
  - Improve numerical stability [#374](https://github.com/magpylib/magpylib/issues/374)
- Docs:
  - General docs improvements [#399](https://github.com/magpylib/magpylib/issues/399) [#294](https://github.com/magpylib/magpylib/issues/294)
  - [MATLAB example](https://magpylib.readthedocs.io/en/latest/_pages/3_MATLAB/#example) updated [#346](https://github.com/magpylib/magpylib/issues/346) [#366](https://github.com/magpylib/magpylib/pull/366)

### Fixed
- Computation:
  - Diametral Solid Cylinder Field singularity at the axis. [#370](https://github.com/magpylib/magpylib/issues/370)
  - Fix empty display [#401](https://github.com/magpylib/magpylib/issues/401)
  - `Collection` of `Lines` return incorret field [#368](https://github.com/magpylib/magpylib/issues/368)
- UI:
  - Adding multiple `Collection` does not mutate the first element.

### Removed

### Deprecated
- (!!!not confirmed yet!!!) `.rotate_from_angax` method is replaced by the more generic and single `.rotate` method. [#420](https://github.com/magpylib/magpylib/issues/420)


---
## [3.0.2] - 2021-06-27

### Overview

This is a major update that includes

- API changes
- New features
- Improved internal workings

### Interface changes

- The `.magnet` and `.current` sub-packages were moved to the top level.
- The `.moment` sub-package was renamed to `.misc` and was moved to the top level.
- The `.vector` sub-package was completely removed. Functionalities are mostly replaced by new top-level function `getBv()`.
- The `.math` sub-package was removed. Functionalities are mostly provided by the `scipy - Rotation` package.
- The top level function `displaySystem()` was renamed to `display()`.

### Source class attribute changes

All parameters are now in explicit format following the Zen of Python and cannot be initialized in their short forms anymore.

- `angle` and `axis` are replaced by `orientation`
- `dimension` is replaced by `diameter` for Loop and Sphere classes.

### The new orientation attribute (CORE FEATURE OF v3)

- The `orientation` attribute stores the relative rotation of an object with respect to the reference orientation (defined in each class docstring).
- The default (`orientation=None`) corresponds to a unit rotation.
- `orientation` is stored as a `scipy.spatial.transform.Rotation` object.
- Calling the attribute `source.orientation` returns a scipy Rotation object `R`.
- Make use of all advantages of this great scipy package:
  - define `R.from_rotvec()` or `R.from_quat()` or ...
  - view with `R.as_rotvec()` or `R.as_quat()` or ...
  - combine subsequent rotations `R1 * R2 * R3`

### New ways to work with Collections and Sources

- The construction `col = Collection(src1, col1, [src2, ...],...)` now features
  - arbitrary input levels,
  - input oder is kept,
  - duplicates are automatically removed.
- The method `.addSources()` is now called `.add()`
- The method `.removeSource()` is now called `.remove()`
- Sources and Collections now have `+/-` operations defined:
  - construct as `col1 = src1 + src2 + col2`
  - remove as `col1 - src1`
- Collection objects are now iterable themselves. The iteration goes over their `col.sources` attribute. `[s for s in col.sources]` is similar to `[s for s in col]`
- Collections have `getitem` defined. `col.sources[i]` is similar to `col[i]`.

### The Sensor class

- The new `Sensor(position, pixel, orientation)` class has the argument `pixel` which is `(0,0,0)` by default and refers to pixel positions inside the Sensor (in the Sensor local CS). `pixel` is an arbitrary array_like of the shape (N1, N2, ..., 3).

### Streamlining operation with all Magpylib objects

All objects (Sensors, Sources, Collections) have additional direct access to
- `.display()` method for quick self-inspection.
- `getB()` and `getH()` methods for fast field computations
- `__repr__` attribute defined and will return their type and their `id`.

### Class method changes

- The class methods `.rotate(angle, axis, anchor)` have been replaced by a new `.rotate(rotation, anchor, increment, start)` method where `rotation` ist a scipy `Rotation` object.
- The original angle-axis-anchor rotation is now provided by the new method `.rotate_from_angax(angle, axis, anchor, increment, start, degrees)`.
  - The argument `axis` can now easily be set to the global CS axes with `"x"`, `"y"`, `"z"`.
  - The anchor argument `anchor=0` represents the origin `(0,0,0)`.
  - `angle` argument is in units of deg by default. It can now be set to rad using the `degrees` argument.
- The "move"-class method is now `.move(displacement, increment, start)`
- Rotation and move methods can now be used to generate paths using vector input and the `increment` and `start` arguments.
- All operations can now be chained (e.g. `.move_by().rotate().move_to()`)

### Paths (CORE FEATURE OF v3)

- The `position` and `orientation` attributes can now store paths in the global CS. For a path of length M the attribute `position` is an array of the shape (M,3) and `orientation` is a Rotation object with length M. Each path position is associated with a respective rotation.
- Field computations `getB()` and `getH()` will evaluate the field for all source path positions.
- Paths can be set by hand `position = X`, `orientation = Y`, but they can also conveniently be generated using the `rotate` and `move` methods.
- Paths can be shown via the `show_path=True` kwarg in `display()`. By setting `show_path=x` the object will be displayed at every `x`'th path step. This helps to follow up on object rotation along the path.
- All objects have a `reset_path()` method defined to set their paths to `position=(0,0,0)` and `orientation=None`.

### Compute magnetic fields (CORE FEATURE OF v3)

- There are two fundamental arguments for field computation:
  - The argument `sources` refers to a source/Collection or to a 1D list of L sources and/or Collections.
  - The argument `observers` refers to a set of positions of shape (N1, N2, ..., 3) or a Sensor with `pixel` shape (N1, N2, ..., 3) or a 1D list of K Sensors.
- With Magpylib3 there are several ways to compute the field:
  1. `source.getB(*observers)`
  2. `sensor.getB(*sources)`
  3. `magpylib.getB(sources, observers)`
     - The output shape is always (L, M, K, N1, N2, ..., 3) with L sources, M path positions, K sensors and N (pixel) positions.
     - Objects with shorter paths will be considered as static once their path ends while other paths still continue.
  4. `magpylib.getBv(**kwargs)` gives direct access to the field formulas and mostly replaces the `getBv_XXX()` functionality of v2. All inputs must be arrays of length N or of length 1 (statics will be tiled).
- While `getBv` is the fastest way to compute the fields it is much more convenient to use `getB()` which mostly provides the same performance. Specifically,the new `getB()` automatically groups all inputs for combined vectorized evaluation. This leads to a massive speedup when dealing with large Collections of similar sources.
- In addition to `getB`, the new `getH` returns the field in [kA/m].

### Miscellaneous and Config

- The top-level `Config` allows users to access and edit Magpylib default values.
- In a finite region (size defined by `Config.EDGESIZE`) about magnet edges and line currents the field evaluates to `(0,0,0)` instead of `(NaN, NaN, NaN)`. Special case catching reduces performance slightly.
- The Box field is now more stable. Numerical instabilities in the outfield were completely removed.
- By default (turn off in `Config.CHECK_INPUTS`) Magpylib now performs some checks of the input format to alert the user and avoid cryptic error messages.
- the kwarg `niter=50` does not exist anymore for the Cylinder field computation. The functionality was completely replaced by the config setting `Config.ITER_CYLINDER=50`.

---
## [2.3.0b] - 2020-01-17

### Changed
- Improved performance of getB for diametral magnetized Cylinders by 20%.
- GetB of Line current now uses vectorized code which leads to massive performance enhancement.
- **IMPORTANT:** position arguments of `getBv` functions have been flipped! First comes the source position POSm THEN the observer position POSo!
- - getB(pos) now takes single AND vector position arguments. If a vector is handed to getB it will automatically execute vectorized code from the vector module.

### Added
- completed the library vector functionality adding magnet Cylinder, moment Dipole, current Loop and Line. This includes adding several private vectorized functions (e.g. ellipticV) to mathLib_vector, adding respective tests and docs examples.

---

## [2.1.0b] - 2019-12-06

### Added
- Docstrings for vector functions.
- displaySystem kwarg `figsize`
- bringing documentation up to speed

### Fixes
- init file bug

---

## [2.0.0b] - 2019-11-29
### Changed
- Restructuring
  - displaySystem is now a top-level function, not a Collection method anymore.
  - getBsweep and multiprocessing options have been completely removed, this functionality
    should be overtaken by the new vector functionality which uses the numpy native vectorized
    code paradigm. If mkl library is set (test by numpy.show_config()) numpy will also
    automatically use multiprocessing. Code parallelization at magpylib level should be done
    by hand.
- Docstrings are adjusted to work better with intellisense. (Problems with *.rst code)
- public rotatePosition() is now called angleAxisRotation(), former private angleAxisRotation
    is now called angleAxisRotation_priv().
- Major rework of the documentation and examples.

### Added
- Performance computation trough vector functionality included in new top-level subpackage "vector"
- Vectorized versions of math functions added to "math" subpackage

---

## [1.2.1b0] - 2019-07-31
### Changed
- Optimized getB call (utility integrated)
- Improved Documentation (added Sensor class v1)

---

## [1.2.0b0] - 2019-07-16
### Added
- Sensor Class
  - This allows users to create a coordinate system-enabled Sensor object, which can be placed, rotated, moved and oriented.
  - This object can take the B-Field of a system (be it single source or a Collection) with the added functionality of having its own reference in the coordinate space, allowing users to easily acquire relative B-Field measurements of a system from an arbitrarily placed sensor object.
  - Sensors in a list may be displayed in the `Collection.displaySystem()` by using the `sensors` keyword argument.
- Added content to the `__repr__` builtin to all source classes for quick console evaluations, simply call a defined object in your Python shell to print out its attributes.
### Changed
- Edge cases in field calculations now return a proper [RuntimeWarning](https://docs.python.org/3/library/exceptions.html#RuntimeWarning) instead of console prints
### Fixed
- Unused imports and variables

---

## [1.1.1b0] - 2019-06-25
### Added
- Changelog
### Changed
- Change `Collection.displaySystem()` not having the `block=False` setting for matplotlib's `pyplot.show()` by default, this meant that outside interactive mode calling this function would hang the script until the plot was closed.
  - If for some reason you want to block the application, you may still use `Collection.displaySystem()`'s `suppress=True` kwarg then call pyplot.show() normally.
  - This should cause no API changes, if you have problems please notify us.

### Fixed
- Fix multiprocessing enabled `Collection.getBsweep()` for lots of objects with few positions causing great performance loss. This functionality now behaves as expected for the use case.
- Fix `Collection.displaySystem()`'s drawing of Dipole objects in external axes (plots) using the `subplotAx` kwarg crashing the application. This functionality now behaves as expected for the use case.

---

## [1.1.0b0] - 2019-06-14
### Added
- Implemented one new kwarg for `Collection.displaySystem()`:

    > `subplotAx=None`
        Draw into a subplot axe that already exists. The subplot needs to be 3D projected

  This allows for creating side-by-side plots using displaySystem.
  Figure information must be set manually in pyplot.figure() in order to not squash the plots upon subplotting.


    <details>
    <summary> Click here for Example </summary>

    Code: https://gist.github.com/lucasgcb/77d55f2fda688e2fb8e1e4a68bb830b8

    **Output:**
    ![image](https://user-images.githubusercontent.com/7332704/58973138-86b4a600-87bf-11e9-9e63-35892b7a6713.png)

    </details>

### Changed

- `getBsweep()` for Collections and Sources now always returns a numpy array
- Zero-length segments in Line sources now return `[0,0,0]` and a warning, making it easier to draw spirals without letting users do this unaware.

### Fixed
- Added a workaround fix for a rotation bug we are still working on.

---

## [1.0.2b0] - 2019-05-29

### Added

- `MANIFEST.in` file containing the LICENSE for bundling in PyPi

---

## [1.0.1b0] - 2019-05-28

### Added

- Issue and Pull Request Templates to Repository
- Continuous Integration settings (Azure and Appveyor)
- Code Coverage Reports with codecov



### Removed

- Support for Python 3.5 and under.

---

## [1.0.0b0] - 2019-05-21

The first official release of the magpylib library.

### Added

- Source classes:
   - Box
   - Cylinder
   - Sphere
   - Loop Current
   - Current Line
   - Dipole
- Collection class

---
[Unreleased]:https://github.com/magpylib/magpylib/compare/3.0.2...HEAD
[3.0.2]:https://github.com/magpylib/magpylib/compare/2.3.0-beta...3.0.2
[2.3.0b]:https://github.com/magpylib/magpylib/compare/2.1.0-beta...2.3.0-beta
[2.1.0b]:https://github.com/magpylib/magpylib/compare/2.0.0-beta...2.1.0-beta
[2.0.0b]:https://github.com/magpylib/magpylib/compare/1.2.1-beta...2.0.0-beta
[1.2.1b0]: https://github.com/magpylib/magpylib/compare/1.2.0-beta...1.2.1-beta
[1.2.0b0]: https://github.com/magpylib/magpylib/compare/1.1.1-beta...1.2.0-beta
[1.1.1b0]: https://github.com/magpylib/magpylib/compare/1.1.0-beta...1.1.1-beta
[1.1.0b0]: https://github.com/magpylib/magpylib/compare/1.0.1-beta...1.1.0-beta
[1.0.2b0]: https://github.com/magpylib/magpylib/compare/1.0.1-beta...1.0.2-beta
[1.0.1b0]: https://github.com/magpylib/magpylib/compare/1.0.0-beta...1.0.1-beta
[1.0.0b0]: https://github.com/magpylib/magpylib/releases/tag/1.0.0-beta


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
