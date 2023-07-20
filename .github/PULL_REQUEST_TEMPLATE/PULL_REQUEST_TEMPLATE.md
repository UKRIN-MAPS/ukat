### Proposed changes

Describe the big picture of your changes here to communicate to the maintainers why we should accept this pull request. If it fixes a bug or resolves a feature request, be sure to link to that issue.

### Checklists

- [ ] I have read and followed the [CONTRIBUTING](.github/CONTRIBUTING.md) document
- [ ] This pull request is from and to the dev branch
- [ ] I have added tests that demonstrate the feature/fix works
- [ ] I have added necessary documentation (if appropriate)
- [ ] I have updated documentation which becomes obsolete after my changes (if appropriate)
- [ ] I have added/updated a notebook to demonstrate the changes (if appropriate)
- [ ] Files added follow the repository structure (if appropriate)

If adding test data?
- [ ] Data is anonymised
- [ ] Ensure imaging data is in NIfTI format and was converted using [`d2n`](https://github.com/UKRIN-MAPS/d2n)
- [ ] Update the [`data/README.md`](data/README.md) file in both `ukat/data/` and `ukat/data/contrast` with info about the origin of the data
- [ ] Added tests for the new functions in `data.fetch.py`
