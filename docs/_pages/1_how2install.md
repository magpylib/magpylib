(installation)=

# Installation

## Dependencies

Magpylib works with Python 3.7 or later ! The following packages will be automatically installed, or updated. See [Git Hub](https://github.com/magpylib/magpylib) for respective versions. Packages will never be downgraded.

- numpy
- matplotlib
- scipy (.spatial.transform, .special)

## Using a package manager

Magpylib works with PyPI and conda-forge repositories.

Install with [pip](https://pypi.org/project/pip/),

```console
pip install magpylib
```

or with [conda](https://docs.conda.io/en/latest/)

```console
conda install magpylib
```

## Using Anaconda

Or if you have little experience with Python we recommand using [Anaconda](https://www.anaconda.com).

1. Download & install Anaconda3

2. Start Anaconda Navigator

3. On the interface, go to `Environments` and choose the environment you wish to install magpylib in. For this example, we will use the base environment:

   > ```{image} ../_static/images/install_guide/anaconda0.png
   > ```

4. Click the arrow, and open the conda terminal

   > ```{image} ../_static/images/install_guide/anaconda1.png
   > ```

5. Input the following to install from conda-forge:

   > ```console
   > conda install -c conda-forge magpylib
   > ```

6. Dont forget to select the proper environment in your IDE.

   > ```{image} ../_static/images/install_guide/anaconda2.png
   > ```

## Download Sites

Currently magpylib is hosted at:

- [Conda Cloud](https://anaconda.org/conda-forge/magpylib)
- [Python Package Index](https://pypi.org/project/magpylib/)
- [GitHub repository](https://github.com/magpylib/magpylib)
