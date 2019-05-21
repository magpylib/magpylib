# About tests

The current tests are [standard unit tests](https://en.wikipedia.org/wiki/Unit_testing), running methods of each class with certain inputs, programatically expecting certain ouputs; accounting for certain edge-cases present within the intrinsic nature of magpylib. These make sure that the codebase is behaving as expected whenever changes are introduced.

>It is important that for every introduced method there should be at least one test case showcasing its main functionality.

## Implementation

Tests are implemented utilizing [pytest](https://docs.pytest.org/en/latest/). The automation interface [tox](https://tox.readthedocs.io/en/latest/) is utilized to create and install the library under a new environment, making sure the installation process works before running the tests. [With our configuration](../tox.ini), a code coverage html report is also generated when calling tox, demonstrating how much of the codebase has been tested.

To run rests, simply run `pytest` or run the `tox` interface .

```
$ pytest
```

```
$ tox
```

## Automated Checks

Automated "checks" are done by the continuous integration service [CircleCI](https://circleci.com/) whenever a push is made to any branch on the remote repository at GitHub. This makes sure the Codebase always remains tested. 

CircleCI will utilize [the configuration file](../.circleci/config.yml), running all the steps necessary (seen above) to properly perform the tests in a cloud environment.  

If a test in a latest commit fails in the cloud environment, a ❌ will appear next to responsible branch being tested, letting maintainers know the code is behaving unexpectedly and should be looked into before merged into a development or release branch.
