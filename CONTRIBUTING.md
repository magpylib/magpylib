# Contribution Guide

The success of Magpylib relies on its user-friendliness. Your feedback and participation in discussions is strongly encouraged. Ask questions about Magpylib. Tell us what you like and what you dislike. Start general discussions in our informal [Discussions](https://github.com/magpylib/magpylib/discussions) channel on GitHub.

We use GitHub [Issues and Milestones](https://github.com/magpylib/magpylib/issues) to plan and track the Magpylib project. Open new Issues to report a bug, to point out a problem, or to make a feature request, e.g. following a fruitful discussion. Within the issue we will define in detail what should be done. For small bug fixes, code cleanups, and other small improvements it's not necessary to create issues.

Always feel free to reach out through the official email <magpylib@gmail.com>.

## How to Contribute with Coding

You are most welcome to become a project contributor by helping us with coding. This includes the implementation of new features, fixing bugs, code cleanup and restructuring as well as documentation improvements. Please abide by the following procedure to make things easy for us to review and to manage the project.

1. Fork the Magpylib repository to your GitHub account
2. Edit your new repository (good practice: clone to local machine, edit, push changes).
3. Rebase your new repository (or pull from upstream) regularly to include upstream changes.
4. Once your changes are complete (see [Coding requirements](coding-requ) below), or you want some feedback, make a pull request (PR) targeting the Magpylib repository. Explain your feature in the PR, and/or refer to the respective issue that you address. Add illustrative code examples.
5. Once a PR is created, our pipeline tests will automatically check your code. A Magpylib member will review your contributions and discuss your changes. Possible improvements will be requested.
6. When satisfied, the reviewer will merge your PR and you become an official Magpylib contributor.

(coding-requ)=
## Coding Requirements

- All code is well documented and all top level doc strings abide by the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- All unit tests are running. We recommend using the [Pytest](https://docs.pytest.org/en/7.4.x/) package.
- New unit tests are written aiming for 100% code coverage. We use [Coverage](https://coverage.readthedocs.io/en/) to test this.
- [Pylint](https://pylint.readthedocs.io/en/stable/) rates your code 10/10 and there are no formatting issues reported (e.g. line-too-long).
- Your code is PEP8 compliant and formatted with [Black](https://black.readthedocs.io/en/stable/) default settings.

We strongly suggest that you use the [Pre-Commit](https://pre-commit.com/) hooks that apply important code checks which each commit.

## For Your Orientation

The Magpylib repository is structured as follows:
- **magpylib**
    - **magpylib**: the actual package.
        - **_src**: source code
        - Other files generate the interface
    - **docs**: documentation that is displayed on [Read the Docs](https://readthedocs.org/) using [Sphinx](https://www.sphinx-doc.org/en/master/).
    - **tests**: unit tests
    - Other files are project configuration files, Readme, ...
