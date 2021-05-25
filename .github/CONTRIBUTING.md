*(This document is work-in-progress)*

# TL;DR
- Document all code (`numpy` [format](https://numpydoc.readthedocs.io/en/latest/format.html)).
- Ensure code follows [PEP8](https://www.python.org/dev/peps/pep-0008/).
- Provide tests.
- Ideally, provide examples and/or tutorials.
- Make sure you follow the checklists on the [pull request template](PULL_REQUEST_TEMPLATE.md)
- Branch off and request merges to the `dev` branch

# Workflow
We generally use the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow#:~:text=The%20overall%20flow%20of%20Gitflow,merged%20into%20the%20develop%20branch) workflow. In a nutshell, this means that if you want to add a new feature/modify some existing code, you should make your branch from the `dev` branch, implement your changes, then request that they are merged back into the dev branch. Once enough issues have been closed, a maintainer will make a release branch from `dev` that will then merge into master. To keep things tidy, please follow the branch naming convention below.

### Features
Adding an entirely new feature to `ukat`. These branches should start with `feature/` e.g. when adding T1 mapping to `ukat` the branch would be called `feature/t1_mapping`
### Maintenance
Work on the GitHub repository itself or changes to existing code e.g. updating `README.md` or changing the fitting bounds of a mapping method. These branches should start with `maintenance/`
### Releases
Branches taken from `dev` that will merge into `master`. These branches will contain commits changing version numbers and updating `CHANGELOG.md` etc and should conform to the standard `release/vX.Y.Z`
### Hotfix
These branches are used for fixing time sensitive bugs in already released code for example, if it was discovered that a previous release accidentally changed the unit T1 was returned in from milliseconds to years, a hotfix branch could be used to correct this mistake, however correcting "form" to "from" in a docstring should be performed on a maintenance branch. Hotfix branches should be forked from `master` and merged into both `master` and `dev`; they should also keep their scope as small as possible, the very bare essentials to fix the bug. These branches should start with `hotfix/` e.g. `hotfix/change_t1_unit_from_years_to_ms`

# Documentation
Code with bad/non-existent documentation will become useless sooner or later. All modules/classes/functions should be documented following the `numpy` docstring guide ([format](https://numpydoc.readthedocs.io/en/latest/format.html), [example](https://numpydoc.readthedocs.io/en/latest/example.html#example)). Note that adopting good naming practices (e.g. descriptively naming variables, functions, classes, etc...), helps self-documenting the code and reduces the amount of explicit documentation needed.

## Generating documentation
*To do*

# Code style
This repository will contain code written primarily in [Python](https://www.python.org/), release 3.x. Ensure your code style follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard. This ensures readability and consistency across the entire repository. Most modern editors can be configured to highlight code sections that do not comply to [PEP8](https://www.python.org/dev/peps/pep-0008/) ([example](https://code.visualstudio.com/docs/python/linting)), which makes this easier to achieve. In addition, we use a [pep8speaks](https://pep8speaks.com/) [bot](https://github.com/UKRIN-MAPS-PEP8SPEAKS) to do this automatically for us when pull requests (PRs) are submitted ([example](https://github.com/UKRIN-MAPS/UKRIN-MAPS/pull/11#issuecomment-620669120)). If you disagree with the bot, please do suggest [configuration](https://github.com/OrkoHunter/pep8speaks#configuration) changes by submitting a PR to modify the [.pep8speaks.yml](https://github.com/UKRIN-MAPS/UKRIN-MAPS/blob/master/.pep8speaks.yml) configuration file.

Code duplication should be avoided (i.e. [don't repeat yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)), unless there is a strong reason to do so (e.g. performance, comprehensibility).

And don't forget about [The Zen of Python](https://www.python.org/dev/peps/pep-0020/):

    >> import this
    The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


# Versioning
Use the [semver](https://semver.org/) convention for milestones, releases. More information on creating a new release can be found [here](https://github.com/UKRIN-MAPS/ukat/wiki/Creating-a-New-Release).

# Tests
We use the [pytest](https://docs.pytest.org/) framework to write tests for our code. These tests can then be run locally while you're developing or, when you open a pull request, via a continuous integration GitHub Action. Additionally [codecov](https://app.codecov.io/gh/UKRIN-MAPS/ukat) will evaluate the coverage of the whole project and your new patch, please try and keep coverage as close to 100% as possible. 

As well as the unit tests used to assess the functionality of the code, we also implement end-to-end tests of each module using 'real' MRI data i.e. data acquired with a scanner rather than generated on a computer. This ensures the results of analysis performed using `ukat` don't change without anyone realising.

# Examples/tutorials
We should strive to have examples/tutorials for the main methods implemented in this repository. Ideally in jupyter notebook format (`.ipynb` files) in the `tutorials` directory. See [here](/tutorials/t2star_calculation.ipynb) for an example.

# Test/Example Data
MRI data isn't stored in `ukat`, it's stored on external web-hosting and then [downloaded at runtime](https://github.com/UKRIN-MAPS/ukat/blob/master/ukat/data/fetch.py). We recommend the use of Zenodo for online storage as it generates a DOI for each dataset and records usage statistics. We have a [UKRIN community](https://zenodo.org/communities/ukrin/) to help keep data curated. If you want to add data to this community please use [this link](https://zenodo.org/deposit/new?c=ukrin). Alternatively, any public facing link to data can be incorporated into `ukat` i.e. if you would rather keep the data on your institutional web-hosting, that's fine.

# Misc
- If you are looking for ideas for contributing, look for any open issues.
- Other contributing guidelines for inspiration: [`dipy`](https://github.com/dipy/dipy/blob/master/CONTRIBUTING.md), [`fslpy`](https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/fslpy/latest/contributing.html).
