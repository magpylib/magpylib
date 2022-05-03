####
# This is a basic setup.py structure so we can generate
# distributable package information with setuptools.
# More information: https://packaging.python.org/tutorials/packaging-projects/
###
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
import os
import sys

import setuptools
from setuptools.command.install import install

_magPyVersion = "4.0.2"
_SphinxVersion = "4.4.0"
_name = "magpylib"
_description = "Free Python3 package to compute magnetic fields."
_author_email = "magpylib@gmail.com"
_author = "Michael Ortner"
_projectUrl = "https://github.com/magpylib/magpylib"
_release = "release"
_license = "2-Clause BSD License, Simplified BSD License, FreeBSD License"

with open("README.md") as fh:
    long_description = fh.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches CircleCI version"""

    description = "verify that the git tag matches CircleCI version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != _magPyVersion:
            info = f"Git tag: {tag} does not match the version of this app: {_magPyVersion}"
            sys.exit(info)


setuptools.setup(
    name=_name,
    version=_magPyVersion,
    author=_author,
    author_email=_author_email,
    description=_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=_projectUrl,
    license=_license,
    packages=setuptools.find_packages(),
    zip_safe=False,  ## Gives the environment files so we can access docs,
    ## enables tooltips but may decrease performance
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.3",
        "plotly>=5.3",
    ],
    # kaleido, jupyterlab are needed for testing with display(renderer='json', backend='plotly')
    extras_require={
        "dev": [
            "kaleido",
            "pytest",
            "coverage",
            "pylint",
            "jupyterlab>=3.2",
            "sphinx==4.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.7",
    keywords="magnetism physics analytical parallel electromagnetic fields b-field",
    command_options={
        "build_sphinx": {
            "project": ("setup.py", _name),
            "version": ("setup.py", _SphinxVersion),
            "release": ("setup.py", _release),
            "source_dir": ("setup.py", "./docs"),
        }
    },
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
