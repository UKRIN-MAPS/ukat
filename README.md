<h2 align="center"><img src="images/logo.png" height="128"></h2>
<p align="center"><strong>Repository of software for renal MRI analysis</strong></p>

# Getting started
[Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the repository to your GitHub account. Then, [clone](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-2-create-a-local-clone-of-your-fork) the repository to your local machine. After doing this, you should see:

    $ git remote -v
    > origin   https://github.com/<YOUR-GITHUB-USERNAME>/UKRIN-MAPS.git (fetch)
    > origin   https://github.com/<YOUR-GITHUB-USERNAME>/UKRIN-MAPS.git (push)

Now, [configure](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) git to sync your fork with the original UKRIN-MAPS repository:

    $ git remote add upstream https://github.com/UKRIN-MAPS/UKRIN-MAPS.git

Now the upstream repository should be set:

    $ git remote -v
    > origin     https://github.com/<YOUR-GITHUB-USERNAME>/UKRIN-MAPS.git (fetch)
    > origin     https://github.com/<YOUR-GITHUB-USERNAME>/UKRIN-MAPS.git (push)
    > upstream   https://github.com/UKRIN-MAPS/UKRIN-MAPS.git (fetch)
    > upstream   https://github.com/UKRIN-MAPS/UKRIN-MAPS.git (push)

Now you can suggest changes (e.g. suggest new code) to be added to the repository via [pull](https://help.github.com/en/github/getting-started-with-github/github-glossary#pull-request) [requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). Don't forget to keep your fork [in sync](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the upstream repository (i.e. the UKRIN-MAPS repository in the UKRIN-MAPS organisation).

If you are new to git/GitHub you may find the following cheat sheets handy ([web](https://github.github.com/training-kit/downloads/github-git-cheat-sheet/), [pdf](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)).

# Requirements
For development, it is recommended that you have Python >= 3.6 installed in your machine. In order to install all dependencies, please run the following command in your terminal:
    > pip install -r requirements.txt

# Contributing guidelines
Please read our [contributing guidelines (*work-in-progress*)](https://github.com/UKRIN-MAPS/UKRIN-MAPS/blob/master/.github/CONTRIBUTING.md).