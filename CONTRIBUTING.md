# Contributing to Magpylib

The sucess of Magpylib relies on its user-friedliness. Your feedback and discussion participation is highly valuable. Ask questions about Magpylib. Tell us what you like and what you dislike. Start general discussions in this very informal channel: **[Discussions](https://github.com/magpylib/magpylib/discussions)**

We use GitHub Issues and Milestones to plan and track the Magpylib project. Open new **[Issues](https://github.com/magpylib/magpylib/issues)** to report a bug, to point out a problem, or to  make a feature request, e.g. following a fruitful discussion. Within the Issue we will together define in detail what should be done.

Once an Issue documents an improvement that everyone agrees upon, its time for coding and you are most welcome to contribute. Please abide by the following procedure **how to contribute with coding**, to make things easy for us to review and manage the project.

## How to contribute with coding...

1. Everyone has push-rights in the Magpylib repository. It is therefore not necessary to fork the repository, but you can clone it directly to your local machine.
2. Once cloned, create new personal branch locally.
3. Publish (=push) new branch to the Github Magpylib repository.
4. Modify, add and fix things in your local new branch. Push your commits to Github.
5. Rebase the new branch on a regular basis to avoid future merge conflicts.
6. Make a draft pull request where we can review and discuss your changes.
7. When **everything is complete** mark your draft pull request as "ready for review". A Magpylib core team member will then review and merge. Write a note if you do not want your branch deleted.

## Everything is complete when...

- the desired functionality is achieved.
- all code is well documented and all top level doc strings abide the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- all unit tests are running. We recommend using the [pytest](https://docs.pytest.org/en/7.4.x/) package.
- new unit tests are written aiming for 100% code coverage. We suggest you use the python package [coverage](https://coverage.readthedocs.io/en/)
- [Pylint](https://pylint.readthedocs.io/en/stable/) rates your code 10/10.
- your code is PEP8 compliant and formatted with [black](https://black.readthedocs.io/en/stable/) default settings.

We strongly suggest that you install [pre-commit](https://pre-commit.com/) which will stop you from pushing your code when it is not ready.

## For your orientation

the Magpylib repository is structured as follows:

- **magpylib**: contains the actual package.
- **docs**: containts the documentation that is displayed on [Read the Docs](https://readthedocs.org/) using [Sphinx](https://www.sphinx-doc.org/en/master/). We like to use [Markdown](https://daringfireball.net/projects/markdown/) for the documentation.
- **tests**: contains the unit tests
- All other top level files are configuration files

# Contribution guide

This guide is intended to give you a quick overview on how you can contribute to the Magpylib project. Our code development team welcomes various forms of contribution such as
- Contribution of new code
- Bug reports, bug fixes
- Maintenance work, see {internal link code maintenance}
- Feature requests, enhancements, ideas
- Code cleanup

Comments via email magpylib@gmail.com or through gitHub channels.

Please refer to {external  link to Github documentation} for learning how to clone, fork, branch and pull request for working on and submitting a specific feature.

## Reporting on bugs
- Please include a self-contained Python snippet, preferably in {external link to GitHub Flavored Markdown} to reproduce the problem:
```python
>>> import magpylib as magpy
>>> magnet = magpy.magnet.Cylinder(...)
```
- Name the version of Magyplib and its dependencies used
```python
>>> import magpylib as magpy
>>> magpy.show_versions()
```
- Give an explanation what is the expected correct instead of the actual behavior



## Code contributions
If you are experienced with analytic magnetic field computations and want to share new or improved analytical recipes that would fit into the project scope you are welcome to contribute to the code base, see also {internal link to Deciding on new features}. Please make sure first that your code can be distributed in Magpylib under a compatible license, see {internal link to License}. New code and feature requests should be addressed on the gitHub mailing list where new features and changes to existing code are discussed and decided on.

Before proposing changes/new code, please make sure to comply to the following guidelines:
- Code style: Please make sure to follow the PEP8 style guide see {PEP8}
- To make sure your code contribution runs correctly, you should consider creating {external link to: unit tests}
- Benchmarks?
- Documentation: It is important for us to understand the code you share, why we ask for a basic docstring documentation for any functions and classes.

Once your code is in a state ready for contribution, i.e. it should be correct, efficient and meeting the above mentioned guidelines, you can send a pull  request on Github.

## Using Git and GitHub for fixes and contribution
We strongly recommend the use of Git  for contributing to Magpylib as the code is hosted on GitHub to easily facilitate the collaboration of many people together on the project. By adhering to the guidelines given below, this should become feasible even for anyone unfamiliar so far with version control software.

Please sign up for a free {link to Github} account. Many hands-on tutorials and documentation resources can be found online and can be consulted for learning Git in more detail, see e.g.
- [GitHub getting started instructions](https://help.github.com/set-up-git-redirect)
- the [Git documentation](https://git-scm.com/doc)
- the [Git help pages](https://support.github.com/)

### Creating a fork of Magpylib
In order to start working on an issue, hit the Fork button on the Magpylib project page and clone your fork of the magpylib repository onto your local machine.
```
git clone https://github.com/magpylib/magpylib.git magpylib-yourname
cd magpylib-yourname
git remote add upstream https://github.com/magpylib-dev/magpylib.git
git fetch upstream
```
Now a directory magpylib-yourname has been created and your local repository has been connected to the main project magpylib repository.

### Creating a branch of Magpylib
When working on a new feature, you might want to create a designated feature branch as:
```
git branch your-new-feature
git checkout your-new-feature
```
You can work on multiple features, each dedicated to a specific feature branch simultaneously and switch between them by using the git checkout command. Before creating a new branch, ensure that you have the latest upstream modifications of the main branch, which can be done as follows:
```
git checkout main
git pull upstream main --ff-only
```

## Deciding on new features
Decisions on code changes or new contributions are made by consensus of the code dev team and everyone participating on the Github mailing list.


## Code maintenance & Code cleanup
Maintenance of the Magyplib codebase refers to fixing bugs and improving code speed and quality. The same guidelines as discussed above for making code contributions apply also for maintaining the existing code base. The {internal link to Magpylib issue list} covers all bugs reported, feature requests, build issues, etc. By helping to fix these issues, the code quality of Magpylib can be kept high and reliable for the Magpylib user base.

## Contributing to the Magpylib documentation
wip

