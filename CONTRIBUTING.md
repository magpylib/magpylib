# Contributing to Magpylib

The sucess of Magpylib relies on its user-friedliness. Your feedback and discussion participation is highly valuable. Ask questions about Magpylib. Tell us what you like and what you dislike. Start general discussions in this very informal channel: **[Discussions](https://github.com/magpylib/magpylib/discussions)**

We use GitHub Issues and Milestones to plan and track the Magpylib project. Open new **[Issues](https://github.com/magpylib/magpylib/issues)** to report a bug, to point out a problem, or to  make a feature request, e.g. following a fruitful discussion. Within the Issue we will together define in detail what should be done.

Once an Issue documents an improvement that everyone agrees upon, its time for coding and you are most welcome to contribute. Please abide by the following procedure **how to contribute with coding**, to make things easy for us to review and manage the project.

## How to contribute with coding...

1. Clone the Magpylib repository to your local machine.
2. Create new personal branch locally.
3. Publish new branch in the Github Magpylib repository.
4. Modify, add and fix things in your local new branch. Push your commits to the Github.
5. Rebase the new branch on a regular basis to avoid future merge conflicts.
6. Make a draft pull request where we can review and discuss your changes.
7. When **everything is complete** mark your draft pull request as "ready for review". A Magpylib core team member will then review and merge. Write a note if you do not want your branch deleted.

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
