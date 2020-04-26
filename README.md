<h2 align="center"><img src="images/logo.png" height="128"></h2>
<p align="center"><strong>Repository of software for renal MRI analysis</strong></p>

## Getting started
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