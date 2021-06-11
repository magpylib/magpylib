
<p align="center"><img align="center" src=docs/_static/images/magpylib_flag.png height="200"><p>

---
<div>
<p align="center"> Builds: 
<a href="https://anaconda.org/conda-forge/magpylib">
<img align='center' src="https://anaconda.org/conda-forge/magpylib/badges/platforms.svg"> 
  </a>
<a href="https://dev.azure.com/magpylib/magpylib/_build/latest?definitionId=1&branchName=master"> <img align='center' src="https://dev.azure.com/magpylib/magpylib/_apis/build/status/magpylib.magpylib?branchName=master"> </a>
<a href="https://circleci.com/gh/magpylib/magpylib"> <img align='center' src="https://circleci.com/gh/magpylib/magpylib.svg?style=svg"> </a>
<a href="https://ci.appveyor.com/project/OrtnerMichael/magpylib/branch/master"> <img align='center' src="https://ci.appveyor.com/api/projects/status/0mka52e1tqnkgnx3/branch/master?svg=true"> </a>

</p>

<p align="center"> Documentation: 
<a href="https://magpylib.readthedocs.io/en/latest/"> <img align='center' src="https://readthedocs.org/projects/magpylib/badge/?version=latest"> </a>
<a href="https://www.gnu.org/licenses/agpl-3.0"> <img align='center' src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg"> </a>
</p>

<p align="center"> Test Coverage: 
<a href="https://codecov.io/gh/magpylib/magpylib">
  <img src="https://codecov.io/gh/magpylib/magpylib/branch/master/graph/badge.svg" />
  
</a>
<a href="https://lgtm.com/projects/g/magpylib/magpylib/context:python"><img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/magpylib/magpylib.svg?logo=lgtm&logoWidth=18"/></a>
</p>

<p align="center"> Downloads: 
<a href="https://pypi.org/project/magpylib/">
<img src="https://badge.fury.io/py/magpylib.svg" alt="PyPI version" height="18"></a>
<a href="https://anaconda.org/conda-forge/magpylib"><img src="https://anaconda.org/conda-forge/magpylib/badges/version.svg" alt="Conda Cloud" height="18"></a>
<a href="https://anaconda.org/conda-forge/magpylib"><img src="https://anaconda.org/conda-forge/magpylib/badges/installer/conda.svg" alt="Conda Cloud" height="18"></a>
</p>

</div>

---

### What is magpylib ?
- Python package for calculating static magnetic fields of magnets, currents and other sources.
- The fields are computed using analytical solutions (very fast, simple geometries and superpositions thereof, no material response)
- The field computation is coupled to a geometry interface (position, orientation, paths) which makes it convenient to determine relative motion between sources and observers.

<p align="center">
    <img align='center' src=docs/_static/images/index/source_fundamentals.png height="290">
</p>

---

### Dependencies: 
_Python3.7+_, _Numpy_, _Matplotlib_, _Scipy_

---

### Docu & Install:

**Please check out our [documentation](https://magpylib.readthedocs.io/en/latest) for installation, examples and detailed information!**

Installing this project using pip
  ```
  pip install magpylib
  ```

or conda
  ```
  conda install magpylib
  ```

Installing this project locally:
- Clone this repository to your machine.
- In the directory, run `pip install .` in your (conda) terminal.




