## Release Checklist

### Pull Request
- [ ] Branch is from `dev` to `rel/v[x.y.z]`
- [ ] Update the [`CHANGELOG.md`](https://github.com/UKRIN-MAPS/ukat/blob/master/CHANGELOG.md) file with the new version number, release date and add the new features/fixes along with any tags of PRs/issues
- [ ] Bump the version number in [`setup.py`](https://github.com/UKRIN-MAPS/ukat/blob/master/setup.py)
- [ ] Add any new contributors and bump the version number in [`CITATION.cff`](https://github.com/UKRIN-MAPS/ukat/blob/master/CITATION.cff)
- [ ] Do a final check if anything in [readme.MD](rhttps://github.com/UKRIN-MAPS/ukat/blob/master/README.md) needs updating
- [ ] The pull requests is from `rel/v[x.y.z]` to `master`
- [ ] All tests pass
- [ ] Merge `rel/v[x.y.z]` into `master`
- [ ] Merge `rel/v[x.y.z]` into `dev`
- [ ] Create a tag on the merge commit on master with the same version number as the release and push to upstream

### Post Pull Request Checks
- [ ] Close any issues from the milestone that didn't automatically close on merge
- [ ] Close the milestone itself
- [ ] Check the new version has appeared in the [releases list](https://github.com/UKRIN-MAPS/ukat/releases)
- [ ] Check the new version has appeared on [PyPI](https://pypi.org/project/ukat/)
- [ ] Check a now DOI has been minted on [Zenodo](https://zenodo.org/account/settings/github/repository/UKRIN-MAPS/ukat)