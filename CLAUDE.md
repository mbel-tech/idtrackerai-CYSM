# Notes for Claude

## idtracker.ai exit code is inverted: never judge a run by it

`idtrackerai` exits **1 on success** and 0 on a caught failure (`sys.exit(main())`,
and `main()` returns `True`). Any notebook, Colab cell or script that runs it must
judge success by `session_<name>/trajectories/trajectories.npy` existing **and**
`"Success"` in `session_<name>/idtrackerai.log`, and copy results to Drive only on
that check. `idtrackerai_d2_export` is ordinary (0 = success, 1 = real failure).
Stream subprocess output into the cell, since Colab hides it otherwise.

Full rules and a helper to copy: [docs/colab-idtrackerai-rules.md](docs/colab-idtrackerai-rules.md).
Read it before writing or editing any notebook that runs idtracker.ai.

## Repository

The repository is `mbel-tech/idtrackerai-CYSM`, and the pip distribution
has the same name, `idtrackerai-CYSM`, published to PyPI by
`.github/workflows/publish.yml` when a GitHub release is made. Install from
Colab with `pip install "git+https://github.com/mbel-tech/idtrackerai-CYSM.git"`
so a notebook runs against the code next to it, not the last release.

## Releasing to PyPI

When a change should reach `pip install idtrackerai-CYSM` users, remind the
user to cut a release once it is merged. Pushing to `master` alone publishes
nothing.

1. Bump `version` in `pyproject.toml` in the same PR as the change, or in a
   follow-up PR. The scheme is `<upstream base>.postN`: increment N
   (`6.0.15a0.post1` → `6.0.15a0.post2`). Never use a local label like
   `+cysm.1`, because PyPI rejects it. Add a line to the fork's section of
   `docs/source/user_guide/changelog.rst`.
2. Merge, then create the release with the tag `v` + that exact version:

   ```bash
   gh release create v6.0.15a0.post2 --repo mbel-tech/idtrackerai-CYSM --target master --prerelease --title "v6.0.15a0.post2" --notes "..."
   ```

   Keep `--prerelease` while the version contains `a`/`b`/`rc`.
3. The `publish` workflow builds the package, checks that the tag matches
   `pyproject.toml`, and uploads the package through PyPI trusted publishing.
   Follow it with `gh run watch --repo mbel-tech/idtrackerai-CYSM`.

PyPI never accepts the same version twice. A failed or repeated upload needs
a new version number, so don't re-tag. Only use `gh workflow run publish.yml`
to retry when the upload itself did not happen.
