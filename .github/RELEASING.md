# Releasing the client

A code release is a GitHub release whose tag is `v<version>`, e.g. `v0.2.0rc1` or `v0.2.0`. Publishing it runs `.github/workflows/release.yaml`, which builds the package with `uv build` and uploads it to PyPI by trusted publishing (environment `pypi`). The version comes from the tag through hatch-vcs. Data releases (`catalog-*`, `primers-*`, `tech-primers-*`) never publish, and `release-guard.yaml` re-marks the newest catalog release as Latest after a code release. The release notes are GitHub's generated notes; the repository has no CHANGELOG.

## Checklist
1. Catalog schema: `SUPPORTED_CATALOG_SCHEMA` in `src/sddb/_catalog_schema.py` must be at least the `catalog_schema_version` of the newest catalog release, because the client refuses a newer catalog. Check with `gh release download <newest catalog-*> -p release_info.json -O - | jq .catalog_schema_version`.
2. Behaviour changes: read the merged PRs since the last `v*` tag. If a default changes, write it at the top of the release notes in one sentence, e.g. `to_anndata` and `materialize` keep only in-tissue rows by default (#85).
3. Tests: the `Test` workflow on `main` passes. Run `pytest -m network` locally once against the newest catalog.
4. First release only: the pending trusted publishers for `spatialdata-db` exist on PyPI (environment `pypi`) and TestPyPI (environment `testpypi`). Run the workflow once from the Actions tab on the `v*` tag ("Run workflow") to upload to TestPyPI, and check `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ spatialdata-db==<version>`.
5. Publish the GitHub release (generate the notes). Mark a release candidate as a pre-release. Don't mark it Latest.
6. After the release: move the install pins in README, docs and the primer envs to the new tag, as in #88.
