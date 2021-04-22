# Magpylib Changelog

All notable changes to magpylib are documented here.

---

# Releases

## [3.0.0] - 2021-soonish :)

### Overview

This is a major update that includes

- API changes
- New features
- Improved internal workings

### Interface changes

- The `.magnet` and `.current` sub-packges were moved to the top level.
- The `.moment` sub-package was renamed to `.misc` and was moved to the top level.
- The `.vector` sub-package was completely removed. Functionalities are mostly replaced by new top-level function `getBv()`.
- The `.math` sub-package was removed. Functionalities are mostly provided by the `scipy - Rotation` package.
- The top level function `displaySystem()` was renamed to `display()`.

### Source class attribute changes

- `magnetization` is now called `mag`
- `dimension` is now called `dim`
- `position` is now called `pos`
- `angle` and `axis` are replaced by `rot`

### The new rot attribute (CORE FEATURE OF v3)

- The `rot` attribute stores the relative rotation of an object with respect to the init_state (defined in each class docstring).
- The default (`rot=None`) corresponds to a unit rotation.
- `rot` is stored as a `scipy.spatial.transform.Rotation` object.
- `src.rot` returns a scipy Rotation object `R`.
- Make use of all advantages of the scipy package:
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
- Sources and Collections nove have `+/-` operations defined:
  - construct as `col1 = src1 + src2 + col2`
  - remove as `col1 - src1`
- Collection objects are now iterable themselves. The iteration goes over their `col.sources` attribute. `[s for s in col.sources]` is similar to `[s for s in col]`
- Collections have `getitem` defined. `col.sources[i]` is similar to `col[i]`.
- Collections and sources now have a `.display()` method for quick self-inspection.
- Sources and Collections have a `__repr__` attribute defined and will return their type and their `id`.


### The Sensor class

- The new `Sensor(pos, pos_pix, rot)` class has the argument `pos_pix` which is `(0,0,0)` by default and refers to pixel positions inside the Sensor (in the Sensor local CS). `pos_pix` is an arbitrary array_like of the shape (N1, N2, ..., 3).


### Class method changes

- The class methods `.rotate(angle, axis, anchor)` have been replaced by a new `.rotate(R, anchor, steps)` method where `R` ist a scipy `Rotation` object and `steps` can be used to generate paths.
- The original angle-axis-anchor rotationis now provided by the new method `.rotate_from_angax(angle, axis, anchor, steps)`.
  - The argument `axis` can now easily be set to the gloabl CS axes with `"x"`, `"y"`, `"z"`.
  - The anchor argument `anchor=0` represents the origin `(0,0,0)`.
- The class method `.move()` was replaced by `move_by()` and `move_to()`.
- All operations can now be chained (e.g. `.move_by().rotate().move_to()`)


### Paths (CORE FEATURE OF v3)

- The `pos` and `rot` attributes can now store paths in the global CS. For a path of length N the attribute `pos` is of the shape (N,3) and `rot` is a Rotation object with lenghth N. Each path position is associated with a respective rotation.
- Field computations `getB()` will evaluate the field for all source path positions.
- Paths can be set by hand `pos = X`, `rot = Y`, but they can also conveniently be generated using the `steps` argument in the `rotate` and `move` methods.
- Paths can be merged (e.g. spiral = linear motion on top of rotation) using the top level `path_merge` context manager, or by setting negative values for `steps` in subsequent operations.
- Paths can be shown via the `show_path=True` kwarg in `display()`. By setting `show_path=x` the object will be displayed at every `x`'th path step. This helps to follow up on object rotation along the path.
- All objects have a `reset_path()` method defined to set their paths to `pos=(0,0,0)` and `rot=None`.


### Compute magnetic fields (CORE FEATURE OF v3)

- There are two fundamental arguments for field computation:
  - The argument `sources` refers to a source/Collection or to a 1D list of L sources and/or Collections.
  - The argument `observers` refers to a set of positions of shape (N1, N2, ..., 3) or a Sensor with `pos_pix` shape (N1, N2, ..., 3) or a 1D list of K Sensors.
- With v3 there are now several ways to compute the field:
  1. `source.getB(*observers)`
  2. `sensor.getB(*sources)`
  3. `magpylib.getB(sources, observers)`
     - The output shape is always (L, M, K, N1, N2, ..., 3) with L sources, M path positions, K sensors and N (pixel) positions.
     - Objects with path_length=1 are considered as static.
     - The new `getB` automatically groups all inputs for combined vectorized evaluation. Specifically, `Collection.getB()` experiences a massive performance increase with respect to v2.
  4. `magpylib.getBv(**kwargs)` gives direct access to the field formulas and mostly replaces the `getBv_XXX()` functionality of v2. All inputs must be arrays of length N or of length 1 (statics will be tiled).
- While `getBv` is the fastest way to compute the fields it is much more convenient to use `getB()` which mostly provides the same performance.
- In addition to `getB`, the new `getH` returns the field in [kA/m].

### Numerical stability

- In a finite region (size defined in `Config`) about magnet edges and corners the field evaluates to `(0,0,0)` instead of `(NaN, NaN, NaN)`. Special case catching reduces performance slightly.
- The Box field is now more stable. Numerical instabilities in the outfield were completely removed.

### Miscellaneous

- The top-level `Config` allows users to access and edit Magpylib default values.
- By default (turn off in Config) Magplyib now performs some checks of the input format to alert the user and avoid cryptic error messages.


## [2.3.0b] - 2020-01-17

### Changed
- Improved performance of getB for diametral magnetized Cylinders by 20%.
- GetB of Line current now uses vectorized code which leads to massive performance enhancement.
- **IMPORTANT:** position arguments of `getBv` functions have been flipped! First comes the source position POSm THEN the observer position POSo!
- - getB(pos) now takes single AND vector position arguments. If a vector is handed to getB it will automatically execute vectorized code from the vector module.

### Added
- completed the library vector functionality adding magnet Cylinder, moment Dipole, current Circular and Line. This includes adding several private vectorized functions (e.g. ellipticV) to mathLib_vector, adding respective tests and docu examples.

---

## [2.2.0b] - 2019-12-27
- unreleased version

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
    automatically use multiporcessing. Code parallelization at magpylib level should be done
    by hand.
- Docstrings are adjusted to work better with intellisense. (Problems with *.rst code)
- public rotatePosition() is now called angleAxisRotation(), former private angleAxisRotation
    is now called angleAxisRotation_priv().
- Major rework of the documentation and examples.

### Added
- Performance computation trough vector functionality included in new top-level subpackge "vector"
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
   - Circular Current
   - Current Line
   - Dipole
- Collection class

---
[Unreleased]:https://github.com/magpylib/magpylib/compare/2.3.0-beta...HEAD
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

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).