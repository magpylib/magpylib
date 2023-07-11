# Contributing to Magpylib

The sucess of Magpylib relies on its user-friedliness. Your feedback and discussion participation is highly valuable. Ask questions about Magpylib. Tell us what you like and what you dislike. Start general discussions in this very informal channel: **[Discussions](https://github.com/magpylib/magpylib/discussions)**

We use GitHub Issues and Milestones to plan and track the Magpylib project. Open new **[Issues](https://github.com/magpylib/magpylib/issues)** to report a bug, to point out a problem, or to  make a feature request, e.g. following a fruitful discussion. Within the Issue we will together define in detail what should be done. For small bug fixes, code cleanups, and other small improvements its not necessary to create issues.

Always feel free to reach out through the official email **magpylib@gmail.com**.

When it comes to coding (this incudes new features, code cleanup, documentation improvement, ...) you are most welcome to contribute. Please abide by the following procedure **how to contribute with coding**, to make things easy for us to review and to manage the project.

## How to contribute with coding...

1. Fork the Magpylib repository to your GitHub account
2. Edit your new repository (good practice: clone to local machine, edit, push changes).
3. Rebase your new repository regularly to include upstream changes.
4. Once your changes are complete (see **Coding guide** below), or you want some feedback, make a pull request (PR) targeting the Magpylib repository. Explain your feature in the PR. Add illustrative code examples. Reference the issue or discussion that you are addressing.
5. Once a PR is created, our pipelines tests will automatically check your code. A Magpylib member will review your contributions and discuss your changes. Possible improvements will be requested.
6. When satisfied, the reviewer will merge your PR and you become an official Magpylib contributor.

## Coding guide...

- all code is well documented and all top level doc strings abide the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- all unit tests are running. We recommend using the [pytest](https://docs.pytest.org/en/7.4.x/) package.
- new unit tests are written aiming for 100% code coverage. We use the python package [coverage](https://coverage.readthedocs.io/en/) to test this.
- [Pylint](https://pylint.readthedocs.io/en/stable/) rates your code 10/10 and there are no formatting issues reportet (e.g. line-to-long).
- your code is PEP8 compliant and formatted with [black](https://black.readthedocs.io/en/stable/) default settings.

We strongly suggest that you install [pre-commit](https://pre-commit.com/) which will stop you from pushing your code with bad format.

## For your orientation

the Magpylib repository is structured as follows:

- **magpylib**: contains the actual package.
- **docs**: containts the documentation that is displayed on [Read the Docs](https://readthedocs.org/) using [Sphinx](https://www.sphinx-doc.org/en/master/). We use [Markdown](https://daringfireball.net/projects/markdown/) for the documentation.
- **tests**: contains the unit tests
- All other top level files are configuration files
