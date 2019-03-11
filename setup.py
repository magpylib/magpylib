####
# This is a basic setup.py structure so we can generate 
# distributable package information with setuptools.
# Testing environment for later: https://test.pypi.org/
# More information: https://packaging.python.org/tutorials/packaging-projects/
####

###
# Local install:
#   Create virtual environment:
#   $ conda create -n packCondaTest python=3.7.1 anaconda
#   Activate:
#   $ conda activate packCondaTest
#   Generate distribution files (untracked by git):
#   $ (packCondaTest) python3 setup.py sdist bdist_wheel
#   Install the generated library for the environment:
#   $ (packCondaTest) pip install .
# The library is now in the packCondaTest environment.
##
_SphinxVersion = "1.8.2"

_name = "magpylib"
_magPyVersion = "1.0a0"
_description = "A simple, user friendly Python 3.2+ toolbox for calculating magnetic fields from permanent magnets and current distributions."
_author_email = "magpylib@gmail.com"
_author = "Michael Ortner"
_projectUrl = "https://github.com/magpylib/magpylib"
_release = "alpha"

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=_name,
    version=_magPyVersion,
    author=_author,
    author_email= _author_email,
    description=_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=_projectUrl,
    packages=setuptools.find_packages(),
    zip_safe = False, ## Gives the environment files so we can access docs, enables tooltips but may decrease performance
    install_requires=[
          'numpy',
          'matplotlib',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', _name),
            'version': ('setup.py', _SphinxVersion),
            'release': ('setup.py', _release),
            'source_dir': ('setup.py', './..')}},
)
