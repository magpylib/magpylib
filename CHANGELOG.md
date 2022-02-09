All notable changes to magpylib are documented here.


# Releases

## [Unreleased v4]
This is a major update that includes

- API changes
- New features
- Improved internal workings
### Added
- New computaion core: ([#376](https://github.com/magpylib/magpylib/issues/376))
  - top level subpackage `magpylib.core` where all field expression implementations can be accessed directly.
- New sources classes:
  - New `CylinderSegment` computation with dimension `(r1,r2,h,phi1,phi2)` with the inner radius `r1`, the outer radius `r2` the height `h` and the cylinder section angles `phi1 < phi2`. ([#386](https://github.com/magpylib/magpylib/issues/386), [#385](https://github.com/magpylib/magpylib/issues/385))
  - New `CustomSource` class for user defined field functions ([#349](https://github.com/magpylib/magpylib/issues/349), [#409](https://github.com/magpylib/magpylib/issues/409), [#411](https://github.com/magpylib/magpylib/pull/411))
- New `display` (now `show`) features:
  - 🚀 New `plotly` plotting backend: ([#396](https://github.com/magpylib/magpylib/pull/396), [#353](https://github.com/magpylib/magpylib/issues/353))
    - `plotly` remains a optional dependency, needs extra installation ([#395](https://github.com/magpylib/magpylib/issues/395))
    - Interactive `animation` feature with `animation_time` and `animation_fps` `display` arguments

  - Display defaults can be now set in `magpylib.defaults.display` ([#291](https://github.com/magpylib/magpylib/issues/291), [#396](https://github.com/magpylib/magpylib/pull/396))
  - Ability to define extra 3d-model for any object. ([#416](https://github.com/magpylib/magpylib/pull/416))
  - Zoom option in display function.
  - Display specific path positions.
  - `matplotlib` plotting backend:
    - Added `matplotlib` pixel display ([#279](https://github.com/magpylib/magpylib/issues/279))
- Documentation and Example codes now available on read the docs with binder links and live code. ([#389](https://github.com/magpylib/magpylib/issues/389))
- `Collection` objects can now be composed of `sources`, `sensors` or both types. The `getB` and `getH` functions accommodate for all cases. ([#410](https://github.com/magpylib/magpylib/issues/410), [#415](https://github.com/magpylib/magpylib/pull/415), [#297](https://github.com/magpylib/magpylib/issues/297))
- `Collection` has it's own `position`, `orientation` and `style`. This is useful to build compound objects. ([#444](https://github.com/magpylib/magpylib/issues/444))
- All scipy Rotation forms are now also object methods ([#427](https://github.com/magpylib/magpylib/pull/427))
- The new `absolute` argument for the `move` method enables the absolute positioning in the global coordinate system ([#444](https://github.com/magpylib/magpylib/issues/444))


### Changed
- `Box` class renamed to `Cuboid`. ([#350](https://github.com/magpylib/magpylib/issues/350))
- `Circular` class renamed to `Loop`. ([#402](https://github.com/magpylib/magpylib/pull/402))
- Magpylib objects can now be _initialized_ **without excitation** and **without dimension** attributes.
- `magpylib.Config` parameters are now in `magpylib.defaults` ([#387](https://github.com/magpylib/magpylib/issues/387))
- `getBv` and `getHv` are now integrated into `getB` and `getH` ([#449](https://github.com/magpylib/magpylib/pull/449))
- `display` is renamed to `show` ([#453](https://github.com/magpylib/magpylib/pull/453), [#451](https://github.com/magpylib/magpylib/issues/451))
- Renamed `show` (prev. `display`) arguments
  - `axis` ➡️ `canvas`
  - `show_direction` ➡️ `style_magnetization_show`
  - `show_path` ➡️ `path`
  - `size_sensors`&`size_dipoles` ➡️ `style_size`
  - `size_direction` ➡️ `style_magnetization_size`
- `move` and `rotate` methods have `start='auto'` instead of `start=-1`. Apply to full path if input is a scalar, append/merge if input is a vector ([#438](https://github.com/magpylib/magpylib/discussions/438), [#444](https://github.com/magpylib/magpylib/issues/444))
- `show_path` in the `display` function is integrated in styling options with `style_path_show` ([#453](https://github.com/magpylib/magpylib/pull/453))

### Updated
- Computation:
  - Improoved field computations: ([#374](https://github.com/magpylib/magpylib/issues/374))
    - negative dimension input taken as absolute when only positive dimensions are allowed
    - scale invariant field evaluations
    - special cases caught within 1e-15 rtol and atol to account for numerical imprecision with positioning (e.g. object rotation).
    - supress numpy divide/invalid warnings. return np.nan as (0,0,0) (e.g. on magnet edges or on line currents) and allow return of np.inf.
  - Updated `Cylinder` diametral magnetization computation with novel closed form implementation. 
    - Much faster (100-1000x)
    - Numerically stable for small `r`.
  - Improved performance of `getB` and `getH` functions and methods.
  - Improved numerical stability of current Loop field. [#374](https://github.com/magpylib/magpylib/issues/374) 
- Other
  - Added `__len__` dunder for `Collection` ([#383](https://github.com/magpylib/magpylib/issues/383))
- Docs:
  - General docs improvements ([#399](https://github.com/magpylib/magpylib/issues/399), [#294](https://github.com/magpylib/magpylib/issues/294))
  - [MATLAB example](https://magpylib.readthedocs.io/en/latest/_pages/3_MATLAB/#example) updated ([#346](https://github.com/magpylib/magpylib/issues/346), [#366](https://github.com/magpylib/magpylib/pull/366))

### Fixed
- Computation:
  - Diametral Solid Cylinder Field singularity at the axis. ([#370](https://github.com/magpylib/magpylib/issues/370))
  - `Collection` of `Lines` return incorret field ([#368](https://github.com/magpylib/magpylib/issues/368))
  - Field array dimension wrongly reduced when `sumup=True` and `squeeze=False` ind `getB` and `getH` functions ([#425](https://github.com/magpylib/magpylib/issues/425), [#426](https://github.com/magpylib/magpylib/pull/426))
- UI:
  - Adding multiple `Collection` objects does not mutate the first element.
  - Fix empty display ([#401](https://github.com/magpylib/magpylib/issues/401))

### Removed
- `config.ITERCYLINDER` is now obsolete. The iterative solution replaced by a new analytical expression.
-  removed `increment` argument from `move` and `rotate` functions ([#438](https://github.com/magpylib/magpylib/discussions/438), [#444](https://github.com/magpylib/magpylib/issues/444))


---
## [3.0.2] - 2021-06-27

This is a major update that includes

- API changes
- New features
- Improved internal workings
### Added

- New `orientation` property:
  - The `orientation` attribute stores the relative rotation of an object with respect to the reference orientation (defined in each class docstring).
  - The default (`orientation=None`) corresponds to a unit rotation.
  - `orientation` is stored as a `scipy.spatial.transform.Rotation` object.
  - Calling the attribute `source.orientation` returns a scipy Rotation object `R`.
  - Make use of all advantages of this great scipy package:
    - define `R.from_rotvec()` or `R.from_quat()` or ...
    - view with `R.as_rotvec()` or `R.as_quat()` or ...
    - combine subsequent rotations `R1 * R2 * R3`
- Sensor pixel:
  - The new `Sensor(position, pixel, orientation)` class has the argument `pixel` which is `(0,0,0)` by default and refers to pixel positions inside the Sensor (in the Sensor local CS). `pixel` is an arbitrary array_like of the shape (N1, N2, ..., 3).
- Geometry paths:
  - The `position` and `orientation` attributes can now store paths in the global CS. For a path of length M the attribute `position` is an array of the shape (M,3) and `orientation` is a Rotation object with length M. Each path position is associated with a respective rotation.
  - Field computations `getB()` and `getH()` will evaluate the field for all source path positions.
  - Paths can be set by hand `position = X`, `orientation = Y`, but they can also conveniently be generated using the `rotate` and `move` methods.
  - Paths can be shown via the `show_path=True` kwarg in `display()`. By setting `show_path=x` the object will be displayed at every `x`'th path step. This helps to follow up on object rotation along the path.
  - All objects have a `reset_path()` method defined to set their paths to `position=(0,0,0)` and `orientation=None`.

- Streamlining operation with all Magpylib objects:
  - All objects (Sensors, Sources, Collections) have additional direct access to
    - `.display()` method for quick self-inspection.
    - `getB()` and `getH()` methods for fast field computations
    - `__repr__` attribute defined and will return their type and their `id`.
- Other new features:  
  - The top-level `Config` allows users to access and edit Magpylib default values.
### Changed
- Renamed modules:
  - `.magnet` and `.current` sub-packages were moved to the top level.
  - The `.moment` sub-package was renamed to `.misc` and was moved to the top level.
  - The `.vector` sub-package was completely removed. Functionalities are mostly replaced by new top-level function `getBv()`.
  - The `.math` sub-package was removed. Functionalities are mostly provided by the `scipy - Rotation` package.
- Renamed functions:
  - The top level function `displaySystem()` was renamed to `display()`.
- Renamed attributes (parameters cannot be initialized in their short forms anymore):
    - `angle` and `axis` are replaced by `orientation`
    - `dimension` is replaced by `diameter` for Loop and Sphere classes.
    - `angle`&`axis` are replaced by `orientation` (see [Added](#Added) Section)
  
- Modified rotate methods:
  - The class methods `.rotate(angle, axis, anchor)` have been replaced by a new `.rotate(rotation, anchor, increment, start)` method where `rotation` ist a scipy `Rotation` object.
  - The original angle-axis-anchor rotation is now provided by the new method `.rotate_from_angax(angle, axis, anchor, increment, start, degrees)`.
    - The argument `axis` can now easily be set to the global CS axes with `"x"`, `"y"`, `"z"`.
    - The anchor argument `anchor=0` represents the origin `(0,0,0)`.
    - `angle` argument is in units of deg by default. It can now be set to rad using the `degrees` argument.
  - The "move"-class method is now `.move(displacement, increment, start)`
  - Rotation and move methods can now be used to generate paths using vector input and the `increment` and `start` arguments.
  - All operations can now be chained (e.g. `.move_by().rotate().move_to()`)
- Miscellaneous:
  - `getB(pos)` now takes single AND vector position arguments. If a vector is handed to getB it will automatically execute vectorized code from the vector module.
  - In a finite region (size defined by `Config.EDGESIZE`) about magnet edges and line currents the field evaluates to `(0,0,0)` instead of `(NaN, NaN, NaN)`. Special case catching reduces performance slightly.
### Updated
- Improved Computation:
  - The Box field is now more stable. Numerical instabilities in the outfield were completely removed.
- Updated Field computation interface
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

### Removed
- the kwarg `niter=50` does not exist anymore for the Cylinder field computation. The functionality was completely replaced by the config setting `Config.ITER_CYLINDER=50`.

---
## [2.3.0b] - 2020-01-17

### Changed
- Improved performance of getB for diametral magnetized Cylinders by 20%.
- GetB of Line current now uses vectorized code which leads to massive performance enhancement.
- **IMPORTANT:** position arguments of `getBv` functions have been flipped! First comes the source position POSm THEN the observer position POSo!


### Added
- completed the library vector functionality adding magnet Cylinder, moment Dipole, current Loop and Line. This includes adding several private vectorized functions (e.g. ellipticV) to mathLib_vector, adding respective tests and docs examples.

---

## [2.1.0b] - 2019-12-06

### Added
- Docstrings for vector functions.
- displaySystem kwarg `figsize`
- bringing documentation up to speed

### Fixed
- init file bug

---

## [2.0.0b] - 2019-11-29
This is a major update that includes

API changes
New features
Improved internal workings
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
