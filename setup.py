####
# This is a basic setup.py structure so we can generate 
# distributable package information with setuptools.
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
_magPyVersion = "1.2.1-beta"

_SphinxVersion = "1.8.2"
_name = "magpylib"
_description = "A simple, user friendly Python 3 toolbox for calculating magnetic fields from permanent magnets and current distributions."
_author_email = "magpylib@gmail.com"
_author = "Michael Ortner"
_projectUrl = "https://github.com/magpylib/magpylib"
_release = "beta"
_license='GNU Affero General Public License v3 or later (AGPLv3+) (AGPL-3.0-or-later)'

import sys
import os
import setuptools
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches CircleCI version"""
    description = 'verify that the git tag matches CircleCI version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != _magPyVersion:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, _magPyVersion
            )
            sys.exit(info)
    
setuptools.setup(
    name=_name,
    version=_magPyVersion,
    author=_author,
    author_email= _author_email,
    description=_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=_projectUrl,
    license=_license,
    packages=setuptools.find_packages(),
    zip_safe = False, ## Gives the environment files so we can access docs, enables tooltips but may decrease performance
    install_requires=[
          'numpy>=1.16',
          'matplotlib>=3.1',
      ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Programming Language :: Python :: 3",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.6',
    keywords='magnetism physics analytical parallel electromagnetic fields b-field',
    command_options={
        'build_sphinx': {
            'project': ('setup.py', _name),
            'version': ('setup.py', _SphinxVersion),
            'release': ('setup.py', _release),
            'source_dir': ('setup.py', './docs')}},
    cmdclass={
    'verify': VerifyVersionCommand,
    }
)
