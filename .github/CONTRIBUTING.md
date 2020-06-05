*(This document is work-in-progress)*

# TLDR
- Document all code (`numpy` [format](https://numpydoc.readthedocs.io/en/latest/format.html)).
- Ensure code follows [PEP8](https://www.python.org/dev/peps/pep-0008/).
- Provide tests.
- Ideally, provide examples and/or tutorials.
- Make sure you follow the checklists on the [pull request template](PULL_REQUEST_TEMPLATE.md)

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
Use the [semver](https://semver.org/) convention for milestones, releases...

# Tests
*To do*

# Examples/tutorials
We should strive to have examples/tutorials for the main methods implemented in this repository. Ideally in jupyter notebook format (`.ipynb` files) in the `tutorials` directory. See [here](/tutorials/t2star_calculation.ipynb) for an example.

# Misc
- If you are looking for ideas for contributing, look for any open issues.
- Other contributing guidelines for inspiration: [`dipy`](https://github.com/dipy/dipy/blob/master/CONTRIBUTING.md), [`fslpy`](https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/fslpy/latest/contributing.html).
