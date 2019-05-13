# Guide - Developer's Manual

This is meant as a quickstart guide to aid in contributing to the project. 
Please shoot an email at magpylib@gmail.com if there is any confusion.

## Code format guidelines

We (try) to follow the following guidelines:

- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)

With the following exception:
- [camelCase](https://en.wikipedia.org/wiki/Camel_case) for variables and most methods.

Please run an [aggressive autopep8](https://github.com/hhatto/autopep8) over your work before committing code. In VSCode for instance this can be done by [enabling the configuration](https://code.visualstudio.com/docs/python/editing#_formatting) and hitting `CTRL`+`Shift`+`i`.

## About the structure

All actual code is placed in `magpylib/_lib` in their relevant submodules. Organization is done by imports in top folders (`magpylib`, `math`, and `source`'s `__init__.py` files).

- `magpylib/_lib/classes/` are classes or modules for *sources*, *collection*, *base* and *field sampler* classes.

- `magpylib/_lib/field/` are the interfaces for calculating electromagnetic fields of different sources.

## Adding new Sources
  When adding new sources to an existing category:
  
  1. Define their class in their existing category file inside of `classes/`, inheriting their base class category as seen in `base.py`. 
     > _For instance, `magpylib/_lib/classes/magnets.py` is a **category file** for `magnet`. 
      >In `base.py`, `HomoMag` is the **class category** for `magnet`. 
      >Define `newMagnet` as `def newMagnet(HomoMag)` inside `magpylib/_lib/classes/magnets.py`._
  2. Define their field calculation in a new file at `magpylib/_lib/fields/`
      >_Override field getter (`getB`) code by importing the field calculation you've created._
  3. Update the class importer in `magpylib/source/[category]/__init__.py`
       >_Import the class and add it to the `__all__` variable:_
       ```python
        __all__ = ["newMagnet", "Box", "Cylinder", "Sphere"] #This is for Sphinx
        from magpylib._lib.classes.magnets import newMagnet
        ```
  4. Update the list in `isSource` function in `utility.py`:
        ```
            def isSource(theObject : any) -> bool:
                ...
                from magpylib import source
                sourcesList = ( ...
                                source.magnet.newMagnet
                                ... )
                ...
        ```
  - Add a test file with tests for creation, manipulation, field calculation and possible edge cases to `magpylib/tests/[relevant folder]`.

  ### Adding other things

  - When adding new source categories, put their definition in `base.py` and inherit their sources with it.
    - Add a folder structure and an `__init__.py` to their respective importers.
  - When adding new fields, put the method and docstring in `field sampler`. Override working code in each source.



  

### Adding functions

- Math helpers should be placed in `_lib/mathLibPrivate.py`.
- Math functions should be placed in `_lib/mathLibPublic.py`.
- Other Helper and Generic functions should be placed in `_lib/utility.py`.

Add them to their respective importers' `__init__.py` file.



