# Running idtracker.ai from a notebook or script: rules

Read this before writing any Colab cell, notebook or script that runs
`idtrackerai` (or `idtrackerai_d2_export`) as a subprocess and then copies or
judges its results. Each rule here comes from a run whose finished results were
thrown away or whose errors were hidden.

## 1. idtracker.ai's exit code is inverted

The `idtrackerai` console script does `sys.exit(main())`, and `main()` returns
`True` when tracking succeeds. `sys.exit(True)` exits with **1**, so:

| exit code | meaning |
|---|---|
| **1** | **success** |
| 0 | a failure that idtracker.ai caught and logged |

**Never judge a tracking run by its exit code.** A run succeeded when both of
these are true:

- `<output_dir>/session_<name>/trajectories/trajectories.npy` exists, and
- `"Success"` appears in `<output_dir>/session_<name>/idtrackerai.log` or in the
  run's captured output.

Copy results to Drive, mark an arm as done and skip it on resume using that
check, never `returncode == 0`.

## 2. The exporter's exit code is ordinary

`idtrackerai_d2_export` exits **0 on success and 1 on a real failure** (a clip it
could not export, a bad session file, a model that would not load). Only
`idtrackerai` itself is inverted. Do not "fix" the exporter check by flipping it.

## 3. Show and keep the subprocess output

In Colab, `subprocess.run(cmd)` without capturing sends the child's output
nowhere visible, so a failure shows up only as an exit code. Stream the output
into the cell and keep a copy on Drive, so the error is readable after a
disconnect too. Stop a loop at the first export failure, since the next clips
usually fail the same way.

## 4. Track on the Colab disk, copy to Drive only when finished

Run tracking with `output_dir` on `/content/...`, then copy the session folder
to Drive under a temporary name and rename it, only after the check in rule 1
passes. Drive then never holds a half-finished session. Keep a RESCUE cell that
copies runs which finished but were judged failed. It has to run in the same
runtime, because `/content` is lost on disconnect.

## Helper to copy

```python
import os, shutil, subprocess
from pathlib import Path

def run_logged(cmd, log_file: Path):
    """Run cmd, stream its output to the cell, keep a copy, return (rc, tail)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tail = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc, open(log_file, "w") as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            tail = (tail + [line])[-40:]
    return proc.returncode, "".join(tail)

def tracking_succeeded(session_dir: Path, output: str = "") -> bool:
    """idtracker.ai exits 1 on success: never use the exit code."""
    log = session_dir / "idtrackerai.log"
    text = (log.read_text(errors="ignore") if log.is_file() else "") + output
    return (session_dir / "trajectories" / "trajectories.npy").is_file() and "Success" in text

def copy_to_drive(local_session: Path, drive_session: Path):
    """Copy under a temporary name, then rename, so Drive never holds half a session."""
    tmp = drive_session.with_name(".copying_" + drive_session.name)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(local_session, tmp)
    if drive_session.exists():          # os.replace cannot overwrite a non-empty folder
        shutil.rmtree(drive_session)
    os.replace(tmp, drive_session)

# usage
rc, tail = run_logged(["idtrackerai", "--load", str(toml), "--track"], logs / f"{stem}.txt")
session = local_out / f"session_{stem}"
if tracking_succeeded(session, tail):        # NOT: if rc == 0
    copy_to_drive(session, drive_out / f"session_{stem}")
```

The broilercamp notebooks that follow these rules are
`colab_downscales_benchmark_fixed.ipynb` (Detectron2) and
`colab_downscales_sam3_v2.ipynb` (SAM 3).
