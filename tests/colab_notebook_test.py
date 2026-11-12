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


def pip_commands(notebook, monkeypatch, backends, huggingface=False, folder=None):
    """What the install cell would pip, for the models the sessions need."""
    import subprocess
    from types import SimpleNamespace

    commands = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: commands.append(" ".join(cmd[2:])) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setenv("HF_TOKEN", "from-the-test")
    namespace = {"BACKENDS": backends, "NEEDS_HUGGINGFACE": huggingface,
                 "FOLDER": folder or Path("unused")}  # fmt: skip
    exec(code(notebook, "install"), namespace)  # noqa: S102 - our own notebook cell
    return commands


@pytest.mark.parametrize(
    "backend, wanted", [("yolo", "ultralytics"), ("sam3", "sam3"), ("detectron2", "detectron2")]
)
def test_only_the_models_own_package_is_installed(tmp_path, monkeypatch, backend, wanted):
    """Which models run is only known once the folder is read, so the cell
    decides when it runs; this runs it with pip stood in for."""
    notebook = cn.build_notebook([session("yolo")])
    commands = pip_commands(notebook, monkeypatch, [backend], folder=tmp_path)
    packages = " ".join(commands)
    for package in ("ultralytics", " sam3", "detectron2"):
        assert (package in packages) == (package.strip() == wanted), commands
    assert "idtrackerai-CYSM" in commands[-1]


def test_the_install_source_can_be_a_branch(monkeypatch):
    notebook = cn.build_notebook([session("yolo")], install_from="git+https://x/y.git@b")
    assert pip_commands(notebook, monkeypatch, ["yolo"])[-1].endswith("git+https://x/y.git@b")


def test_hugging_face_weights_need_the_token(monkeypatch):
    notebook = cn.build_notebook([session("sam3-huggingface")])
    assert pip_commands(notebook, monkeypatch, ["sam3"], huggingface=True)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        monkeypatch.delenv("HF_TOKEN")
        exec(  # noqa: S102 - our own notebook cell
            code(notebook, "install"),
            {"BACKENDS": ["sam3"], "NEEDS_HUGGINGFACE": True, "FOLDER": Path("unused")},
        )


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


# ------------------------------------------- companion .toml files on Drive


def companion(folder: Path, relative: str, **params) -> Path:
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in params.items():
        if isinstance(value, dict):
            inner = ", ".join(f"{k} = {json.dumps(v)}" for k, v in value.items())
            lines.append(f"{key} = {{{inner}}}")
        else:
            lines.append(f"{key} = {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def written(drive, name: str) -> dict:
    return toml.loads((drive.folder / "sessions" / f"{name}.toml").read_text("utf-8"))


def test_a_toml_named_after_the_video_is_used_for_it(tmp_path, capsys):
    """The file beside the video says how that video is run, whatever the
    notebook was written with."""
    drive = Drive(tmp_path, [session("yolo")])
    drive.put("best.pt")
    drive.put("other.pt")
    video = drive.put("videos/R2_01_10_P5_SB_fps24.mp4")
    companion(
        drive.folder, "toml/R2_01_10_P5_SB_fps24.toml",
        number_of_animals=7,
        roi_list=["+ Polygon [[1, 1], [9, 1], [9, 9]]"],
        segmentation_model={"backend": "yolo", "weights": "C:/models/other.pt", "imgsz": 640},
    )  # fmt: skip
    drive.run("sessions")

    params = written(drive, "R2_01_10_P5_SB_fps24")
    assert params["number_of_animals"] == 7
    assert params["roi_list"] == ["+ Polygon [[1, 1], [9, 1], [9, 9]]"]
    assert params["video_paths"] == [str(video)]
    model = params["segmentation_model"]
    assert model["weights"] == str(drive.folder / "other.pt") and model["imgsz"] == 640
    assert model["contours_dir"] == str(drive.folder / "contours" / "R2_01_10_P5_SB_fps24")
    # the notebook's own session for that video stands back
    assert not (drive.folder / "sessions" / "R2_01_10_P5.toml").exists()
    assert "R2_01_10_P5_SB_fps24.toml" in capsys.readouterr().out


def test_a_toml_that_lists_the_video_is_its_companion(tmp_path):
    """Session files written on Windows name their videos by a Windows path."""
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    drive.put("clip_a.mp4")
    drive.put("clip_b.mp4")
    companion(
        drive.folder, "pen5.toml",
        name="pen5",
        video_paths=["D:\\rec\\clip_a.mp4", "D:\\rec\\clip_b.mp4"],
        number_of_animals=9,
        output_dir="D:\\out",
        segmentation_model={"backend": "yolo", "weights": "best.pt",
                            "contours_dir": "D:\\contours"},
    )  # fmt: skip
    drive.run("sessions")
    params = written(drive, "pen5")
    assert params["video_paths"] == [
        str(drive.folder / "clip_a.mp4"), str(drive.folder / "clip_b.mp4")
    ]  # fmt: skip
    assert "output_dir" not in params
    assert params["segmentation_model"]["contours_dir"] == str(
        drive.folder / "contours" / "pen5"
    )


def test_a_toml_for_a_video_that_is_not_there_is_left_out(tmp_path, capsys):
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    drive.put("here.mp4")
    model = {"backend": "yolo", "weights": "best.pt"}
    companion(drive.folder, "here.toml", number_of_animals=9, segmentation_model=model)
    companion(drive.folder, "absent.toml", video_paths=["absent.mp4"],
              number_of_animals=9, segmentation_model=model)  # fmt: skip
    drive.run("sessions")
    names = sorted(p.name for p in (drive.folder / "sessions").glob("*.toml"))
    assert names == ["here.toml"]
    assert "absent.toml" in capsys.readouterr().out


def test_a_video_without_a_toml_stops_at_the_settings(tmp_path):
    """Nothing says how to run it, so the notebook asks instead of guessing."""
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    drive.put("lonely.mp4")
    with pytest.raises(SystemExit, match="settings cell") as stop:
        drive.run("sessions")
    assert "lonely.mp4" in str(stop.value)
    assert "USE_FOR_VIDEOS_WITHOUT_A_TOML" in str(stop.value)


def test_the_settings_cell_runs_the_videos_without_a_toml_once_confirmed(tmp_path):
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    video = drive.put("lonely.mp4")
    drive.namespace.update(
        BACKEND="yolo", WEIGHTS="best.pt", NUMBER_OF_ANIMALS=4, IMAGE_SIZE=1280,
        ROI_LIST=["+ Polygon [[1, 1], [9, 1], [9, 9]]"],
        USE_FOR_VIDEOS_WITHOUT_A_TOML=True,
    )  # fmt: skip
    drive.run("sessions")
    params = written(drive, "lonely")
    assert params["video_paths"] == [str(video)] and params["number_of_animals"] == 4
    assert params["roi_list"] == ["+ Polygon [[1, 1], [9, 1], [9, 9]]"]
    spec = sm.normalize_model(params["segmentation_model"])
    assert (spec["backend"], spec["imgsz"]) == ("yolo", 1280)
    assert spec["weights"] == str(drive.folder / "best.pt")


def test_confirmed_settings_still_need_what_tracking_needs(tmp_path):
    drive = Drive(tmp_path, [])
    drive.put("lonely.mp4")
    drive.namespace.update(USE_FOR_VIDEOS_WITHOUT_A_TOML=True)
    with pytest.raises(SystemExit) as stop:
        drive.run("sessions")
    assert "WEIGHTS" in str(stop.value) and "NUMBER_OF_ANIMALS" in str(stop.value)


def test_the_settings_are_prefilled_from_the_session_the_notebook_was_saved_from(tmp_path):
    drive = Drive(tmp_path, [session("yolo")])
    ns = drive.namespace
    assert (ns["BACKEND"], ns["WEIGHTS"], ns["IMAGE_SIZE"]) == ("yolo", "best.pt", 1280)
    assert ns["NUMBER_OF_ANIMALS"] == 9 and ns["ROI_LIST"] == ROI
    assert ns["USE_FOR_VIDEOS_WITHOUT_A_TOML"] is False


def test_another_video_in_the_folder_is_not_run_with_this_sessions_settings(
    tmp_path, capsys
):
    """One pen's region of interest must not be applied to another pen's
    video just because both are in the folder."""
    drive = Drive(tmp_path, [session("yolo")])
    drive.put("best.pt")
    drive.put("R2_01_10_P5_SB_fps24.mp4")
    drive.put("R3_08_10_P26_SB_fps24.mp4")
    drive.run("sessions")
    names = sorted(p.name for p in (drive.folder / "sessions").glob("*.toml"))
    assert names == ["R2_01_10_P5.toml"]
    said = capsys.readouterr().out
    assert "R3_08_10_P26_SB_fps24.mp4" in said and "settings cell" in said


def test_one_model_can_replace_the_one_in_every_toml(tmp_path):
    """Session files written for SAM 3, run with the YOLO of the settings."""
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    drive.put("pen5.mp4")
    companion(
        drive.folder, "pen5.toml", number_of_animals=9,
        segmentation_model={"backend": "sam3", "weights": "huggingface", "prompt": "bird"},
    )  # fmt: skip
    drive.namespace.update(
        BACKEND="yolo", WEIGHTS="best.pt", USE_THIS_MODEL_FOR_EVERY_VIDEO=True
    )
    drive.run("sessions")
    model = written(drive, "pen5")["segmentation_model"]
    assert model["backend"] == "yolo" and "prompt" not in model
    assert drive.namespace["BACKENDS"] == ["yolo"]
    assert drive.namespace["NEEDS_HUGGINGFACE"] is False


def test_two_tomls_for_one_video_are_refused(tmp_path):
    drive = Drive(tmp_path, [])
    drive.put("best.pt")
    drive.put("pen5.mp4")
    model = {"backend": "yolo", "weights": "best.pt"}
    companion(drive.folder, "pen5.toml", number_of_animals=9, segmentation_model=model)
    companion(drive.folder, "again.toml", video_paths=["pen5.mp4"],
              number_of_animals=9, segmentation_model=model)  # fmt: skip
    with pytest.raises(SystemExit, match="pen5.mp4"):
        drive.run("sessions")


def test_a_notebook_can_be_written_without_any_session():
    notebook = cn.build_notebook([])
    intro = "".join(notebook["cells"][0]["source"])
    assert ".toml" in intro and "settings cell" in intro
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")


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


def test_the_command_line_needs_no_session(tmp_path, monkeypatch, capsys):
    output = tmp_path / "generic.ipynb"
    monkeypatch.setattr(sys, "argv", ["idtrackerai_colab_notebook", "--output", str(output)])
    cn.main()
    assert json.loads(output.read_text(encoding="utf-8"))["nbformat"] == 4
    assert ".toml" in capsys.readouterr().out
