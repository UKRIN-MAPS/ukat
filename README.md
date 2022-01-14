<h2 align="center"><img src="https://raw.githubusercontent.com/UKRIN-MAPS/ukat/master/images/ukat_logo.png" height="180"></h2>
<p align="center"><strong>UKRIN Kidney Analysis Toolbox (ukat) </strong></p>

![Build and Test](https://github.com/UKRIN-MAPS/ukat/workflows/Build%20and%20Test/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/UKRIN-MAPS/ukat/branch/master/graph/badge.svg?token=QJ9DQONJBP)](https://codecov.io/gh/UKRIN-MAPS/ukat)
[![PyPI version](https://badge.fury.io/py/ukat.svg)](https://badge.fury.io/py/ukat)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/256993671.svg)](https://zenodo.org/badge/latestdoi/256993671)

`ukat` is a vendor agnostic framework for the analysis of quantitative renal MRI data. 

The [UKRIN-MAPS project](https://www.nottingham.ac.uk/research/groups/spmic/research/uk-renal-imaging-network/ukrin-maps.aspx) aims to standardise the acquisition and analysis of renal MRI data to enable multi-site, 
multi-vendor studies. 
Although many MRI vendors produce quantitative maps on the scanner, their methods are closed source and as such, 
potentially cause variability in multi-vendor studies. `ukat` provides an open-source and robust analysis platform that can be used to process data from multiple vendors. 

The focus of this package is analysis of data from the standardised UKRIN protocol, however the methods are intentionally 
left generic to enable analysis of data collected using different protocols or on different areas of the body.

More information can be found in [this ISMRM abstract](https://www.researchgate.net/publication/349830229_UKRIN_Kidney_Analysis_Toolbox_UKAT_A_Framework_for_Harmonized_Quantitative_Renal_MRI_Analysis).

# Installing `ukat`
There are a few different ways you can install `ukat` based on what you want to do with it
### "I just want to process my data with this package"
1. Make sure you're running Python >=3.7
2. Install `ukat` with `pip install ukat`

### "I want to modify this code to do something a bit different but don't want my modifications to go back into `ukat`"
1. Clone this repository with `git clone https://github.com/UKRIN-MAPS/ukat.git`
2. Change to the `ukat` root directory (the one containing a file named `setup.py`).
3. Run the following command in your terminal: `pip install -e . `

Now if you make any changes to the `ukat` code, they'll permeate into any analysis you perform where you've imported `ukat`.

### "I want to contribute to `ukat` and write code that ends up back on this repository for others to use"
Great! 

[Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the repository to your GitHub account. Then, [clone](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-2-create-a-local-clone-of-your-fork) the repository to your local machine. After doing this, you should see:

    $ git remote -v
    origin   https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (fetch)
    origin   https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (push)

Now, [configure](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) git to sync your fork with the original `ukat` repository:

    $ git remote add upstream https://github.com/UKRIN-MAPS/ukat.git

Now the upstream repository should be set:

    $ git remote -v
    origin     https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (fetch)
    origin     https://github.com/<YOUR-GITHUB-USERNAME>/ukat.git (push)
    upstream   https://github.com/UKRIN-MAPS/ukat.git (fetch)
    upstream   https://github.com/UKRIN-MAPS/ukat.git (push)

Now you can suggest changes (e.g. suggest new code) to be added to the repository via [pull](https://help.github.com/en/github/getting-started-with-github/github-glossary#pull-request) [requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). Don't forget to keep your fork [in sync](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the upstream repository (i.e. the `ukat` repository in the UKRIN-MAPS organisation).

If you are new to git/GitHub you may find the following cheat sheets handy ([web](https://github.github.com/training-kit/downloads/github-git-cheat-sheet/), [pdf](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)).

You'll probably also want to follow the instructions in the section above so you can use `ukat` for your normal analysis.

If you run into any problems or find any issues with the installation process please raise an [issue](https://github.com/UKRIN-MAPS/ukat/issues).

# Contributing guidelines
Please read our [contributing guidelines (*work-in-progress*)](.github/CONTRIBUTING.md).
