<h2 align="center"><img src="images/logo.png" height="128"></h2>
<p align="center"><strong>UKRIN Kidney Analysis Toolbox (ukat) </strong></p>

[![Build Status](https://travis-ci.com/UKRIN-MAPS/ukat.svg?token=7aU73aCyDpzGTeY9Af2j&branch=master)](https://travis-ci.com/UKRIN-MAPS/ukat)
[![codecov](https://codecov.io/gh/UKRIN-MAPS/ukat/branch/master/graph/badge.svg?token=QJ9DQONJBP)](https://codecov.io/gh/UKRIN-MAPS/ukat)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Getting started
[Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the repository to your GitHub account. Then, [clone](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-2-create-a-local-clone-of-your-fork) the repository to your local machine. After doing this, you should see:

    $ git remote -v
    origin   https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (fetch)
    origin   https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (push)

Now, [configure](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) git to sync your fork with the original UKRIN-MAPS repository:

    $ git remote add upstream https://github.com/UKRIN-MAPS/ukat.git

Now the upstream repository should be set:

    $ git remote -v
    origin     https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (fetch)
    origin     https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (push)
    upstream   https://github.com/UKRIN-MAPS/ukat.git (fetch)
    upstream   https://github.com/UKRIN-MAPS/ukat.git (push)

Now you can suggest changes (e.g. suggest new code) to be added to the repository via [pull](https://help.github.com/en/github/getting-started-with-github/github-glossary#pull-request) [requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). Don't forget to keep your fork [in sync](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the upstream repository (i.e. the UKRIN-MAPS repository in the UKRIN-MAPS organisation).

If you are new to git/GitHub you may find the following cheat sheets handy ([web](https://github.github.com/training-kit/downloads/github-git-cheat-sheet/), [pdf](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)).

# Installing `ukat`

`ukat` is a python-based library so start by ensuring you have Python (>=3.6) installed. Assuming the repository has been cloned to your machine as described above, the following steps should install `ukat`.

1. Change to the `ukat` root directory (the one containing a file named `setup.py`).
2. Run the following command in your terminal: `pip install -e . `
3. Install [SimpleElastix](https://simpleelastix.github.io/) ([instructions](https://simpleelastix.readthedocs.io/GettingStarted.html)).

If you run into any problems or find any issues with the installation process please raise an [issue](https://github.com/UKRIN-MAPS/ukat/issues).

# Contributing guidelines
Please read our [contributing guidelines (*work-in-progress*)](.github/CONTRIBUTING.md).
