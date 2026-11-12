"""The Colab notebook written from a session, for every segmentation model.

One button (or one command) writes a notebook with everything the session set
already in it. On Colab the only thing left to say is which Drive folder holds
the videos and the weights. These tests pin that promise for each backend:
nothing from the local machine leaks into the notebook, exactly one value is
left to edit, and the cells it runs follow docs/colab-idtrackerai-rules.md.

The cells that are plain Python are executed here against a stand-in Drive
folder, so the session files the notebook would write are checked for real.
"""

import json
import sys
from pathlib import Path

import colab_notebooks_rules_test as rules
import pytest
import toml

from idtrackerai.extra_tools.detectron2_pipeline import colab_notebook as cn
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm

ROI = [
    "+ Polygon [[577, 85], [1370, 92], [1317, 843], [605, 843]]",
    "- Polygon [[900, 400], [1000, 400], [1000, 500], [900, 500]]",
]

MODELS = {
    "yolo": {
        "backend": "yolo",
        "weights": r"D:\broiler camp\models\best.pt",
        "score_threshold": 0.5,
        "imgsz": 1280,
        "contours_dir": r"D:\broiler camp\contours\pen5",
    },
    "sam3": {
        "backend": "sam3",
        "weights": "/home/me/models/sam3.pt",
        "prompt": "bird",
        "score_threshold": 0.5,
        "contours_dir": "/home/me/contours/pen5",
    },
    "sam3-huggingface": {
        "backend": "sam3",
        "weights": "huggingface",
        "prompt": "bird",
    },
    "detectron2": {
        "backend": "detectron2",
        "weights": "/home/me/model/model_final.pth",
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "min_size": 800,
    },
}


def session(model: str, name: str = "R2_01_10_P5", **extra) -> dict:
    return {
        "name": name,
        "video_paths": [rf"D:\broiler camp\sessions\{name}_SB_fps24.mp4"],
        "number_of_animals": 9,
        "track_wo_identities": False,
        "tracking_intervals": [[0, 25920]],
        "area_ths": [300.0, float("inf")],
        "roi_list": ROI,
        "output_dir": r"D:\broiler camp\out",
        "external_contours": r"D:\broiler camp\contours\pen5\x.h5",
        "segmentation_model": dict(MODELS[model]),
    } | extra


def code(notebook: dict, cell_id: str) -> str:
    found = [c for c in notebook["cells"] if c.get("id") == cell_id]
    assert len(found) == 1, f"expected one cell with id {cell_id!r}"
    return "".join(found[0]["source"])


def text(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"])


@pytest.fixture(params=list(MODELS))
def backend(request):
    return request.param


@pytest.fixture
def saved(tmp_path, backend):
    """The notebook of one session, as a file, as the rule tests read it."""
    path = tmp_path / "notebook.ipynb"
    cn.write_notebook(path, [session(backend)])
    return path


# ------------------------------------------------------------- the notebook


def test_it_is_a_notebook_colab_will_open_on_a_gpu(saved):
    notebook = json.loads(saved.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(cell["cell_type"] in ("code", "markdown") for cell in notebook["cells"])
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")


def test_the_drive_folder_is_the_only_thing_left_to_set(saved):
    notebook = json.loads(saved.read_text(encoding="utf-8"))
    first_code = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    assert first_code["id"] == "folder"
    source = "".join(first_code["source"])
    assert "FOLDER = Path('/content/drive/MyDrive/" in source
    assignments = [
        line for line in source.splitlines() if "=" in line and not line.startswith("#")
    ]
    assert len(assignments) == 1
    # every other cell works from FOLDER and names no Drive path of its own
    others = [
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and c["id"] not in ("folder", "mount")
    ]
    assert not any("/content/drive" in source for source in others)


def test_nothing_of_the_local_machine_travels(saved):
    everything = saved.read_text(encoding="utf-8")
    for local in ("broiler camp", "D:", "/home/me", "x.h5"):
        assert local not in everything
    assert "R2_01_10_P5_SB_fps24.mp4" in everything


def test_the_first_cell_says_what_to_put_on_drive(saved, backend):
    notebook = json.loads(saved.read_text(encoding="utf-8"))
    intro = "".join(notebook["cells"][0]["source"])
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "R2_01_10_P5_SB_fps24.mp4" in intro
    weights = {
        "yolo": "best.pt",
        "sam3": "sam3.pt",
        "detectron2": "model_final.pth",
    }.get(backend)
    if weights:
        assert weights in intro
    else:
        assert "Hugging Face" in intro and "HF_TOKEN" in intro
    assert "GPU" in intro and "Run all" in intro


def test_only_the_models_own_package_is_installed(saved, backend):
    install = code(json.loads(saved.read_text(encoding="utf-8")), "install")
    wanted = {
        "yolo": "ultralytics",
        "sam3": "sam3",
        "sam3-huggingface": "sam3",
        "detectron2": "detectron2",
    }[backend]
    assert f"'{wanted}'" in install or f"{wanted}.git" in install
    for other in {"ultralytics", "'sam3'", "detectron2"} - {wanted, f"'{wanted}'"}:
        assert other not in install
    assert "idtrackerai-CYSM" in install
    assert ("HF_TOKEN" in install) == (backend == "sam3-huggingface")


def test_the_checklist_names_the_files_to_upload(backend):
    files = cn.files_to_upload([session(backend)])
    assert "R2_01_10_P5_SB_fps24.mp4" in files
    assert ("huggingface" not in files) and ("model_zoo" not in files)
    if backend == "detectron2":
        assert "model_final.pth" in files


# -------------------------------------------------- the rules for subprocesses


def test_tracking_uses_load_and_is_judged_by_its_output(saved):
    rules.test_tracking_uses_load_not_the_unknown_parameters_flag(saved)
    rules.test_tracking_is_judged_by_its_output_not_its_exit_code(saved)


def test_tracking_runs_on_the_colab_disk_and_copies_finished_sessions(saved):
    rules.test_tracking_runs_on_the_colab_disk_and_copies_finished_sessions(saved)


def test_the_exporter_goes_through_run_step(saved):
    rules.test_exporter_and_prelabeller_go_through_run_step(saved)
    export = code(json.loads(saved.read_text(encoding="utf-8")), "export")
    assert "'idtrackerai_d2_export', '--session'" in export
    assert "'--local-cache'" in export


def test_the_helper_cell_is_the_tested_one(saved, tmp_path, capsys):
    rules.test_the_helper_cell_behaves(saved, tmp_path, capsys)


# -------------------------------------------------- running the plain cells


class Drive:
    """A stand-in Drive folder, and the notebook's namespace run against it."""

    def __init__(self, tmp_path: Path, sessions: list[dict]):
        self.folder = tmp_path / "drive" / "project"
        self.folder.mkdir(parents=True)
        self.notebook = cn.build_notebook(sessions)
        self.namespace: dict = {}
        source = code(self.notebook, "folder")
        assert "'/content/drive/MyDrive/idtrackerai_CYSM'" in source
        self.run_source(
            source.replace("'/content/drive/MyDrive/idtrackerai_CYSM'", repr(str(self.folder)))
        )
        for cell_id in ("settings", "helpers"):
            self.run(cell_id)
        # keep the runtime's own folders inside the test's
        self.namespace["LOCAL_SESSIONS"] = tmp_path / "content" / "sessions_out"
        self.namespace["CACHE"] = tmp_path / "content" / "cache"

    def run_source(self, source: str) -> None:
        exec(source, self.namespace)  # noqa: S102 - our own notebook cell

    def run(self, cell_id: str) -> None:
        self.run_source(code(self.notebook, cell_id))

    def put(self, relative: str, content: bytes = b"x") -> Path:
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


def test_the_session_files_are_written_for_the_drive_folder(tmp_path, backend):
    drive = Drive(tmp_path, [session(backend)])
    video = drive.put("videos/day14/R2_01_10_P5_SB_fps24.mp4")
    weights = {
        "yolo": "models/best.pt",
        "sam3": "sam3.pt",
        "detectron2": "model/model_final.pth",
    }.get(backend)
    if weights:
        weights = drive.put(weights)

    drive.run("sessions")

    written = drive.folder / "sessions" / "R2_01_10_P5.toml"
    params = toml.loads(written.read_text(encoding="utf-8"))
    # found wherever it sits under the folder, by name
    assert params["video_paths"] == [str(video)]
    assert params["roi_list"] == ROI
    assert params["area_ths"] == [300.0, float("inf")]
    assert params["number_of_animals"] == 9
    assert params["tracking_intervals"] == [[0, 25920]]
    assert "output_dir" not in params and "external_contours" not in params

    spec = sm.normalize_model(params["segmentation_model"])
    expected = MODELS[backend]
    assert spec["backend"] == expected["backend"]
    assert spec["contours_dir"] == str(drive.folder / "contours" / "R2_01_10_P5")
    if weights:
        assert spec["weights"] == str(weights)
    else:
        assert spec["weights"] == "huggingface"
    for key in ("prompt", "imgsz", "score_threshold", "config", "min_size"):
        if key in expected:
            assert spec[key] == expected[key]


def test_a_missing_file_stops_the_notebook_and_is_named(tmp_path):
    drive = Drive(tmp_path, [session("yolo")])
    drive.put("R2_01_10_P5_SB_fps24.mp4")
    with pytest.raises(SystemExit, match="best.pt"):
        drive.run("sessions")
    assert not (drive.folder / "sessions" / "R2_01_10_P5.toml").exists()


def test_several_sessions_share_one_notebook(tmp_path):
    sessions = [session("yolo", "R2_01_10_P5"), session("yolo", "R3_08_10_P26")]
    drive = Drive(tmp_path, sessions)
    drive.put("best.pt")
    drive.put("R2_01_10_P5_SB_fps24.mp4")
    drive.put("R3_08_10_P26_SB_fps24.mp4")
    drive.run("sessions")
    names = sorted(p.name for p in (drive.folder / "sessions").glob("*.toml"))
    assert names == ["R2_01_10_P5.toml", "R3_08_10_P26.toml"]


def test_export_track_and_timetable_cells_run_in_order(tmp_path):
    """The export and tracking commands are stood in for; what is checked is
    what the cells do around them: which files they pass, where a finished
    session ends up, and that it is not tracked twice."""
    drive = Drive(tmp_path, [session("yolo")])
    drive.put("best.pt")
    drive.put("R2_01_10_P5_SB_fps24.mp4")
    drive.run("sessions")
    toml_path = drive.folder / "sessions" / "R2_01_10_P5.toml"

    steps = []
    drive.namespace["run_step"] = lambda cmd, log, cwd=None: steps.append(
        [str(c) for c in cmd]
    )
    drive.run("export")
    assert steps[0][:2] == ["idtrackerai_d2_export", "--session"]
    assert str(toml_path) in steps[0]

    tracked = []

    def fake_tracking(cmd, log, cwd=None):
        cmd = [str(c) for c in cmd]
        tracked.append(cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1]) / "session_R2_01_10_P5"
        (out / "trajectories").mkdir(parents=True)
        (out / "trajectories" / "trajectories.npy").write_bytes(b"x")
        (out / "idtrackerai.log").write_text("Success", encoding="utf-8")
        return 1, "Success"  # idtracker.ai exits 1 when it succeeds

    drive.namespace["run_logged"] = fake_tracking
    drive.run("track")
    assert tracked[0][:3] == ["idtrackerai", "--load", str(toml_path)]
    on_drive = drive.folder / "tracked" / "session_R2_01_10_P5"
    assert (on_drive / "trajectories" / "trajectories.npy").is_file()

    drive.run("track")  # a rerun finds it on Drive and does not track again
    assert len(tracked) == 1

    drive.run("timetable")
    rows = (drive.folder / "tracking_timetable.csv").read_text(encoding="utf-8")
    header, row = rows.strip().splitlines()
    assert header.startswith("session,backend,gpu,")
    assert row.startswith("R2_01_10_P5,yolo,")
    drive.run("timetable")  # updating keeps one row per session
    assert len((drive.folder / "tracking_timetable.csv").read_text().splitlines()) == 2


def test_a_failed_tracking_run_is_not_copied(tmp_path):
    drive = Drive(tmp_path, [session("yolo")])
    drive.put("best.pt")
    drive.put("R2_01_10_P5_SB_fps24.mp4")
    drive.run("sessions")
    drive.namespace["run_logged"] = lambda cmd, log, cwd=None: (0, "IdtrackeraiError")
    drive.run("track")
    assert not (drive.folder / "tracked" / "session_R2_01_10_P5").exists()


# ------------------------------------------------------------- what is refused


def test_a_session_without_a_model_is_refused():
    plain = session("yolo")
    del plain["segmentation_model"]
    with pytest.raises(cn.ColabNotebookError, match="segmentation model"):
        cn.build_notebook([plain])


def test_two_sessions_with_one_name_are_refused():
    with pytest.raises(cn.ColabNotebookError, match="same name"):
        cn.build_notebook([session("yolo"), session("sam3")])


def test_two_different_files_with_one_name_are_refused():
    """Files are found on Drive by name, so one name must mean one file."""
    other = session("yolo", "R3_08_10_P26")
    other["segmentation_model"]["weights"] = r"E:\older\best.pt"
    with pytest.raises(cn.ColabNotebookError, match="best.pt"):
        cn.build_notebook([session("yolo"), other])


# ------------------------------------------------------------ the command line


def test_the_command_line_writes_a_notebook_from_toml_files(tmp_path, monkeypatch, capsys):
    tomls = []
    for name in ("R2_01_10_P5", "R3_08_10_P26"):
        path = tmp_path / f"{name}.toml"
        # forward slashes: the toml package writes a backslash unescaped
        params = json.loads(json.dumps(session("yolo", name)).replace("\\\\", "/"))
        path.write_text(toml.dumps(params), encoding="utf-8")
        tomls.append(path)
    output = tmp_path / "out" / "run.ipynb"
    monkeypatch.setattr(
        sys,
        "argv",
        ["idtrackerai_colab_notebook", "--session", str(tmp_path / "R*.toml"),
         "--output", str(output)],
    )  # fmt: skip
    cn.main()

    notebook = json.loads(output.read_text(encoding="utf-8"))
    settings = code(notebook, "settings")
    assert "R2_01_10_P5" in settings and "R3_08_10_P26" in settings
    printed = capsys.readouterr().out
    assert "best.pt" in printed and "R3_08_10_P26_SB_fps24.mp4" in printed
