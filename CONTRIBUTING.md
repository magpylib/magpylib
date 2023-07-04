# Contributing to Magpylib

You dont have to be an expert programmer to contribute to Magpylib. Any feedback helps us understand what is really needed by the community.

Ask questions about Magpylib. Give feeback about things that you like or dislike. Start general discussions here:

**[Discussions](https://github.com/magpylib/magpylib/discussions)**

Make us aware of bugs and problems. Make feature requests and define new features e.g. following a fruitful discussion. Participate in tasks planning here:

**[Issues](https://github.com/magpylib/magpylib/issues)**

Once features are clearly defined in an issue, contribute with actual coding. We will review, help and guide.

## How to contribute with coding...

1. Clone the Magpylib repository on your local machine.
2. Create new personal branch locally.
3. Publish new branch into the Github Magpylib repository. At this point we will be aware that something is happening.
4. Modify, add and fix things in your new branch.
5. Rebase the new branch on a regular basis to avoid future merge conflicts.
6. Push your commits regularly so we can review and discuss your changes.
7. When **everything is complete** make a pull request to merge your branch into the original one.

## Everything is complete when...

- the desired functionality is achieved.
- all code is well documented and all top level doc strings abide the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- unit tests are written. We aim for 100% code coverage. We suggest you use the python package [coverage](https://coverage.readthedocs.io/en/)
- [Pylint](https://pylint.readthedocs.io/en/stable/) rates your code 10/10.
- your code is PEP8 compliant and formatted with [black](https://black.readthedocs.io/en/stable/) default settings.

We strongly suggest that you install [pre-commit](https://pre-commit.com/) which will stop you from pushing your code when it is not ready.

## For your orientation

the Magpylib repository is structured as follows:

- **magpylib**: contains the actual package.
- **docs**: containts the documentation that is displayed on [Read the Docs](https://readthedocs.org/) using [Sphinx](https://www.sphinx-doc.org/en/master/). We like to use [Markdown](https://daringfireball.net/projects/markdown/) for the documentation.
- **tests**: contains the unit tests
- All other top level files are configuration files
