# Contributing to Magpylib

test123

The sucess of Magpylib relies on its user-friedliness. Your feedback and discussion participation is highly valuable. Ask questions about Magpylib. Tell us what you like and what you dislike. Start general discussions in this very informal channel: **[Discussions](https://github.com/magpylib/magpylib/discussions)**

We use GitHub Issues and Milestones to plan and track the Magpylib project. Open new **[Issues](https://github.com/magpylib/magpylib/issues)** to report a bug, to point out a problem, or to  make a feature request, e.g. following a fruitful discussion. Within the Issue we will together define in detail what should be done. For small bug fixes, code cleanups, and other small improvements its not necessary to create issues.

Always feel free to reach out through the official email **magpylib@gmail.com**.

When it comes to coding (this incudes new features, code cleanup, documentation improvement, ...) you are most welcome to contribute. Please abide by the following procedure **how to contribute with coding**, to make things easy for us to review and to manage the project.

## How to contribute with coding...

1. Everyone has push-rights in the Magpylib repository. It is therefore not necessary to fork the repository, but you can clone it directly to your local machine.
2. Once cloned, create new personal branch locally.
3. Publish (=push) new branch to the Github Magpylib repository.
4. Modify, add and fix things in your local new branch. Push your commits to Github.
5. Rebase the new branch on a regular basis to avoid future merge conflicts.
6. Make a draft pull request (PR) when your features are ready for first reviews and your code satisfies the **Code layout guide**. Explain your feature in the PR. Assign a Magpylib core team member for review. With the draft PR the pipeline tests will run automatically with every push you make to the branch.
7. Discuss, review and improve your branch. When ready (features are there, pipeline tests are all green, everyone is happy) mark your PR as "ready for review". A Magpylib core team member will then review and merge. Write a note if you do not want your branch deleted.

## Code layout guide...

- all code is well documented and all top level doc strings abide the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- all unit tests are running. We recommend using the [pytest](https://docs.pytest.org/en/7.4.x/) package.
- new unit tests are written aiming for 100% code coverage. We use the python package [coverage](https://coverage.readthedocs.io/en/) to test this.
- [Pylint](https://pylint.readthedocs.io/en/stable/) rates your code 10/10 and there are no formatting issues reportet (e.g. line-to-long).
- your code is PEP8 compliant and formatted with [black](https://black.readthedocs.io/en/stable/) default settings.

We strongly suggest that you install [pre-commit](https://pre-commit.com/) which will stop you from pushing your code when it has formatting problems.

## For your orientation

the Magpylib repository is structured as follows:

- **magpylib**: contains the actual package.
- **docs**: containts the documentation that is displayed on [Read the Docs](https://readthedocs.org/) using [Sphinx](https://www.sphinx-doc.org/en/master/). We use [Markdown](https://daringfireball.net/projects/markdown/) for the documentation.
- **tests**: contains the unit tests
- All other top level files are configuration files
