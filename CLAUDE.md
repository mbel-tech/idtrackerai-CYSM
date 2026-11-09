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

The repository is `mbel-tech/idtrackerai-CYSM` (renamed from
`idtrackerai-detectron2`; the pip distribution name is still
`idtrackerai-detectron2`). Install from Colab with
`pip install "git+https://github.com/mbel-tech/idtrackerai-CYSM.git"`.
