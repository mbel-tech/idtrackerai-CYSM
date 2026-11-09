"""The Colab notebooks follow docs/colab-idtrackerai-rules.md.

idtracker.ai exits 1 on success, Colab hides uncaptured subprocess output, and
a session tracked straight onto Drive is left half-written by a disconnect.
These tests pin the notebooks to the rules that avoid all three, and run the
shared helper cell itself so its behaviour is checked, not just its presence.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / (
    "src/idtrackerai/extra_tools/detectron2_pipeline/notebooks"
)
PIPELINE = NOTEBOOKS / "colab_detectron2_pipeline.ipynb"
ROI = NOTEBOOKS / "colab_inference_roi.ipynb"
BOTH = [PIPELINE, ROI]


def cells(path: Path, kind: str = "code") -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == kind
    ]


def all_text(path: Path) -> str:
    return "\n".join(cells(path, "code") + cells(path, "markdown"))


def helper_cell(path: Path) -> str:
    found = [c for c in cells(path) if "def run_logged(" in c]
    assert len(found) == 1, f"{path.name}: expected one helper cell"
    return found[0]


def tracking_cell(path: Path) -> str:
    found = [c for c in cells(path) if "'idtrackerai'" in c and "'--track'" in c]
    assert len(found) == 1, f"{path.name}: expected one tracking cell"
    return found[0]


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_tracking_uses_load_not_the_unknown_parameters_flag(path):
    # idtracker.ai has no --parameters: it rejects it, exits 0, and tracks nothing.
    assert "--parameters" not in all_text(path)
    assert "'--load'" in tracking_cell(path)


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_tracking_is_judged_by_its_output_not_its_exit_code(path):
    cell = tracking_cell(path)
    assert "tracking_succeeded(" in cell
    assert not re.search(r"(returncode|\brc\b|code_?)\s*==\s*0", cell)
    assert "subprocess.run(" not in cell  # streamed and kept through run_logged


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_tracking_runs_on_the_colab_disk_and_copies_finished_sessions(path):
    cell = tracking_cell(path)
    assert "'--output_dir', LOCAL_SESSIONS" in cell
    assert "copy_to_drive(" in cell
    rescue = [c for c in cells(path) if c.lstrip().startswith("# RESCUE")]
    assert len(rescue) == 1 and "copy_to_drive(" in rescue[0]


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_exporter_and_prelabeller_go_through_run_step(path):
    for cell in cells(path):
        for tool in ("detectron2_export_contours.py", "sam3_prelabel.py",
                     "idtrackerai_d2_export"):
            if tool in cell and not cell.lstrip().startswith("#"):
                assert f"!{tool}" not in cell and "!cd" not in cell, cell[:80]
                assert "run_step(" in cell, cell[:80]


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_the_helper_cell_behaves(path, tmp_path, capsys):
    namespace = {"PROJECT": tmp_path / "drive"}
    exec(helper_cell(path), namespace)  # noqa: S102 - our own notebook cell

    # run_logged streams, keeps a copy, and returns the exit code and tail
    log = tmp_path / "drive" / "logs" / "x.txt"
    rc, tail = namespace["run_logged"](
        [sys.executable, "-c", "print('hello'); raise SystemExit(3)"], log
    )
    assert rc == 3 and "hello" in tail and "hello" in log.read_text()
    assert "hello" in capsys.readouterr().out

    # run_step stops the notebook on an ordinary failure, and passes success
    with pytest.raises(SystemExit, match="exit 2"):
        namespace["run_step"]([sys.executable, "-c", "raise SystemExit(2)"], log)
    namespace["run_step"]([sys.executable, "-c", "pass"], log, cwd=tmp_path)

    # tracking_succeeded needs both the trajectories and "Success"
    session = tmp_path / "local" / "session_a"
    (session / "trajectories").mkdir(parents=True)
    succeeded = namespace["tracking_succeeded"]
    assert not succeeded(session)
    (session / "trajectories" / "trajectories.npy").write_bytes(b"x")
    assert not succeeded(session)
    assert succeeded(session, "... Success ...")
    (session / "idtrackerai.log").write_text("[green]Success")
    assert succeeded(session)

    # copy_to_drive replaces an older copy and leaves no temporary folder
    drive = tmp_path / "drive" / "tracked" / "session_a"
    drive.mkdir(parents=True)
    (drive / "stale.txt").write_text("old")
    namespace["copy_to_drive"](session, drive)
    assert (drive / "trajectories" / "trajectories.npy").is_file()
    assert not (drive / "stale.txt").exists()
    assert [p.name for p in drive.parent.iterdir()] == ["session_a"]


def test_idtrackerai_really_rejects_parameters(tmp_path):
    """The reason for the first test: the flag is unknown to the CLI parser."""
    from idtrackerai.start.arg_parser import get_parser  # noqa: PLC0415

    toml = tmp_path / "x.toml"
    toml.write_text("")
    parser = get_parser()
    _, unknown = parser.parse_known_args(["--parameters", str(toml), "--track"])
    assert "--parameters" in unknown
    known, unknown = parser.parse_known_args(["--load", str(toml), "--track"])
    assert known.parameters == [toml] and not unknown
