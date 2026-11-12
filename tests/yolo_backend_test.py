"""YOLO as a segmentation backend, checked against a stand-in ultralytics.

A YOLO segmentation model the user trained (best.pt) is the third thing that
can make a session's contours. These tests pin what is easy to get silently
wrong: what the model is asked for, that raw frames reach it unless the session
says otherwise, and that one .toml drives both the export and the checks
idtracker.ai makes before tracking from it.

No torch, no GPU, no real weights.
"""

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from idtrackerai import IdtrackeraiError, Session
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm

HEIGHT, WIDTH = 120, 160
N_FRAMES = 12


class FakeResult:
    def __init__(self, masks, scores):
        self.masks = None if masks is None else types.SimpleNamespace(data=masks)
        self.boxes = types.SimpleNamespace(conf=np.asarray(scores, np.float32))


class FakeYOLO:
    """Records what it was loaded with and asked for, and answers like the
    real one: a list with one result per image."""

    last = None
    task = "segment"

    def __init__(self, weights):
        self.weights = weights
        self.calls = []
        self.frames = []
        self.task = type(self).task
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.ellipse(mask, (80, 60), (14, 7), 0, 0, 360, 1, -1)
        self.result = FakeResult(mask[None], [0.9])
        FakeYOLO.last = self

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        self.frames.append(frame)
        return [self.result]


@pytest.fixture
def fake_ultralytics(monkeypatch, tmp_path):
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch)
    FakeYOLO.last = None
    monkeypatch.setattr(FakeYOLO, "task", "segment")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not really a model")
    return weights


def make(weights, **kwargs):
    from idtrackerai.extra_tools.detectron2_pipeline.yolo_predictor import (
        YoloPredictor,
    )

    options = dict(device="cpu", score_threshold=0.4, max_instances=9)
    options.update(kwargs)
    return YoloPredictor(weights=weights, **options)


def frame():
    return np.full((HEIGHT, WIDTH, 3), 90, np.uint8)


# ------------------------------------------------------------------ the table


def test_a_yolo_table_needs_only_weights():
    spec = sm.normalize_model({"backend": "YOLO", "weights": "best.pt"})
    assert spec["backend"] == "yolo"
    assert spec["imgsz"] is None
    assert set(spec) == set(sm.DEFAULTS)


@pytest.mark.parametrize(
    "extra, complaint",
    [
        ({"prompt": "bird"}, "prompt only applies"),
        ({"min_size": 640}, "min_size"),
        ({"config": "x.yaml"}, "config"),
        ({"weights": "model_zoo"}, "model_zoo"),
        ({"weights": "huggingface"}, "huggingface"),
        ({"imgsz": 16}, "at least 32"),
        ({"imgsz": "1280"}, "number"),
    ],
)
def test_bad_yolo_tables_are_named(extra, complaint):
    with pytest.raises(sm.SegmentationModelError, match=complaint):
        sm.normalize_model({"backend": "yolo", "weights": "best.pt"} | extra)


def test_imgsz_belongs_to_yolo_alone():
    with pytest.raises(sm.SegmentationModelError, match="imgsz only applies"):
        sm.normalize_model({"weights": "model_final.pth", "imgsz": 1280})


def test_yolo_takes_classes_and_keep_everything():
    spec = sm.normalize_model(
        {"backend": "yolo", "weights": "best.pt", "classes": 0, "max_instances": 0}
    )
    assert spec["classes"] == [0]
    assert sm.effective_max_instances(spec, 9) == 0


def test_a_file_made_at_another_image_size_is_a_mismatch():
    spec = sm.normalize_model(
        {"backend": "yolo", "weights": "/drive/best.pt", "imgsz": 1280}
    )
    attrs = {
        "backend": "yolo",
        "weights": "best.pt",
        "score_threshold": 0.7,
        "max_instances": 9,
        "on_overlap": "merge",
        "imgsz": 1280,
    }
    assert sm.mismatches(attrs, spec, 9) == []
    problems = sm.mismatches(attrs | {"imgsz": 640}, spec, 9)
    assert problems == ["imgsz is 640, the table says 1280"]
    # a table that leaves imgsz to the model accepts whatever the file records
    assert sm.mismatches(attrs, spec | {"imgsz": None}, 9) == []


# -------------------------------------------------------------- the predictor


def test_the_model_is_asked_for_what_the_session_says(fake_ultralytics):
    predictor = make(fake_ultralytics, imgsz=1280, classes=[0])
    masks, scores = predictor.predict(frame())

    model = FakeYOLO.last
    assert model.weights == str(fake_ultralytics)
    call = model.calls[0]
    assert call["conf"] == 0.4
    assert call["max_det"] == 9
    assert call["imgsz"] == 1280
    assert call["classes"] == [0]
    assert call["device"] == "cpu"
    # masks at the frame's own size, not the network's: contours are in pixels
    assert call["retina_masks"] is True
    assert call["verbose"] is False
    assert len(masks) == 1 and scores == [pytest.approx(0.9)]
    assert masks[0].shape == (HEIGHT, WIDTH) and masks[0].dtype == np.uint8
    assert set(np.unique(masks[0])) == {0, 1}


def test_image_size_and_classes_are_left_to_the_model_unless_set(fake_ultralytics):
    predictor = make(fake_ultralytics, max_instances=0)
    predictor.predict(frame())
    call = FakeYOLO.last.calls[0]
    assert "imgsz" not in call and "classes" not in call
    # 0 keeps everything the model reports, so no cap of ours is passed
    assert "max_det" not in call


def test_the_frame_reaches_yolo_as_opencv_read_it(fake_ultralytics):
    """ultralytics takes BGR arrays, which is what OpenCV hands us. Converting
    to RGB here would swap the channels the model was trained on."""
    predictor = make(fake_ultralytics)
    image = frame()
    image[..., 0] = 10
    predictor.predict(image)
    assert FakeYOLO.last.frames[0] is image


def test_a_frame_with_no_animals_is_empty_not_an_error(fake_ultralytics):
    predictor = make(fake_ultralytics)
    FakeYOLO.last.result = FakeResult(None, [])
    assert predictor.predict(frame()) == ([], [])


def test_the_most_confident_are_kept_when_there_are_too_many(fake_ultralytics):
    predictor = make(fake_ultralytics, max_instances=2, score_threshold=0.3)
    masks = np.zeros((4, HEIGHT, WIDTH), np.float32)
    for i in range(4):
        masks[i, 10 * i : 10 * i + 5, 10:20] = 1.0
    FakeYOLO.last.result = FakeResult(masks, [0.5, 0.95, 0.2, 0.8])

    out_masks, scores = predictor.predict(frame())

    assert scores == [pytest.approx(0.95), pytest.approx(0.8)]
    assert out_masks[0][12, 12] == 1 and out_masks[1][32, 12] == 1


def test_masks_at_the_networks_size_are_brought_to_the_frames(fake_ultralytics):
    predictor = make(fake_ultralytics)
    small = np.zeros((1, HEIGHT // 2, WIDTH // 2), np.float32)
    small[0, 20:40, 30:50] = 1.0
    FakeYOLO.last.result = FakeResult(small, [0.9])

    masks, _ = predictor.predict(frame())

    assert masks[0].shape == (HEIGHT, WIDTH)
    assert masks[0][60, 80] == 1 and masks[0][5, 5] == 0


def test_detection_only_weights_are_refused_with_the_reason(fake_ultralytics, monkeypatch):
    """Boxes carry no outline, and idtracker.ai identifies from the pixels
    inside one. Loading such weights must not produce empty contour files."""
    from idtrackerai.extra_tools.detectron2_pipeline import yolo_predictor

    monkeypatch.setattr(FakeYOLO, "task", "detect")
    with pytest.raises(yolo_predictor.YoloWeightsError, match="segmentation"):
        make(fake_ultralytics)


def test_missing_weights_are_named(fake_ultralytics, tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.pt"):
        make(tmp_path / "nowhere.pt")


def test_without_ultralytics_it_says_how_to_get_it(monkeypatch, tmp_path):
    from idtrackerai.extra_tools.detectron2_pipeline import yolo_predictor

    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import fails
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    with pytest.raises(yolo_predictor.YoloWeightsError, match="pip install ultralytics"):
        make(weights)


# --------------------------------------------------------------- the exporter


def make_video(path: Path, n_frames: int = N_FRAMES) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(n_frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), 20 + 10 * i, np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    readable = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if readable != n_frames:
        pytest.skip(f"video round-tripped as {readable} frames, not {n_frames}")
    return path


def write_session(tmp_path, name, videos, **params) -> Path:
    import toml

    path = tmp_path / f"{name}.toml"
    path.write_text(
        toml.dumps({"video_paths": [str(v) for v in videos]} | params), encoding="utf-8"
    )
    return path


def yolo_table(weights, contours_dir, **extra):
    return {
        "backend": "yolo",
        "weights": str(weights),
        "contours_dir": str(contours_dir),
        "min_component": 0,
        "dilate": 0,
    } | extra


def run_exporter(monkeypatch, *argv):
    from idtrackerai.extra_tools.detectron2_pipeline import inference

    monkeypatch.setattr(sys, "argv", ["idtrackerai_d2_export", *map(str, argv)])
    inference.main()


def test_yolo_exports_and_tracks_from_one_toml(tmp_path, monkeypatch, fake_ultralytics):
    video = make_video(tmp_path / "clip.avi")
    contours_dir = tmp_path / "contours"
    table = yolo_table(fake_ultralytics, contours_dir, imgsz=1280, score_threshold=0.4)
    toml_path = write_session(
        tmp_path,
        "pen1",
        [video],
        number_of_animals=1,
        tracking_intervals=[[3, 9]],
        segmentation_model=table,
    )

    run_exporter(monkeypatch, "--session", toml_path, "--device", "cpu")

    model = FakeYOLO.last
    assert len(model.calls) == 6
    assert model.calls[0]["imgsz"] == 1280 and model.calls[0]["max_det"] == 1
    with h5py.File(contours_dir / "clip.h5") as file:
        attrs = dict(file.attrs)
    assert attrs["backend"] == "yolo"
    assert attrs["weights"] == "best.pt"
    assert int(attrs["imgsz"]) == 1280
    assert sm.exported_intervals_of(attrs) == [[3, 9]]

    # a second run finds nothing to redo
    FakeYOLO.last = None
    run_exporter(monkeypatch, "--session", toml_path, "--device", "cpu")
    assert FakeYOLO.last is None

    session = Session()
    session.set_parameters(
        reset=True,
        video_paths=[str(video)],
        number_of_animals=1,
        output_dir=str(tmp_path / "out"),
        tracking_intervals=[[3, 9]],
        segmentation_model=table,
    )
    session.prepare_tracking()  # same model, same settings: accepted

    session = Session()
    session.set_parameters(
        reset=True,
        video_paths=[str(video)],
        number_of_animals=1,
        output_dir=str(tmp_path / "out"),
        tracking_intervals=[[3, 9]],
        segmentation_model=table | {"imgsz": 640},
    )
    with pytest.raises(IdtrackeraiError, match="imgsz is 1280"):
        session.prepare_tracking()


def test_yolo_sees_raw_frames_unless_the_session_enhances(
    tmp_path, monkeypatch, fake_ultralytics
):
    """The exporter's built-in default enhances, which suits a Mask R-CNN
    trained that way. A YOLO model was trained on the frames as recorded, so
    without an `enhancement` key it must get exactly those."""
    video = make_video(tmp_path / "clip.avi")
    contours_dir = tmp_path / "contours"
    toml_path = write_session(
        tmp_path,
        "pen1",
        [video],
        number_of_animals=1,
        segmentation_model=yolo_table(fake_ultralytics, contours_dir),
    )
    run_exporter(monkeypatch, "--session", toml_path, "--device", "cpu")

    with h5py.File(contours_dir / "clip.h5") as file:
        assert json.loads(file.attrs["enhancement"])["enhance"] is False
    # frame 5 was written as grey level 70; raw means it arrives that way
    seen = FakeYOLO.last.frames[5]
    assert abs(float(seen.mean()) - 70) < 3

    # ... and a flag on the command line still wins
    run_exporter(monkeypatch, "--session", toml_path, "--device", "cpu", "--enhance")
    with h5py.File(contours_dir / "clip.h5") as file:
        assert json.loads(file.attrs["enhancement"])["enhance"] is True


def test_the_videos_command_line_takes_yolo(tmp_path, monkeypatch, fake_ultralytics):
    video = make_video(tmp_path / "clip.avi")
    out = tmp_path / "out"
    run_exporter(
        monkeypatch,
        "--backend", "yolo", "--videos", video, "--weights", fake_ultralytics,
        "--output-dir", out, "--max-instances", 3, "--imgsz", 960,
        "--device", "cpu", "--min-component", 0, "--dilate", 0,
    )  # fmt: skip
    call = FakeYOLO.last.calls[0]
    assert call["imgsz"] == 960 and call["max_det"] == 3
    with h5py.File(out / "clip.h5") as file:
        assert file.attrs["backend"] == "yolo"
        assert json.loads(file.attrs["enhancement"])["enhance"] is False

    # changing the image size makes the file out of date
    FakeYOLO.last = None
    run_exporter(
        monkeypatch,
        "--backend", "yolo", "--videos", video, "--weights", fake_ultralytics,
        "--output-dir", out, "--max-instances", 3, "--imgsz", 640,
        "--device", "cpu", "--min-component", 0, "--dilate", 0,
    )  # fmt: skip
    assert FakeYOLO.last is not None and FakeYOLO.last.calls[0]["imgsz"] == 640


@pytest.mark.parametrize(
    "flags, complaint",
    [
        (["--backend", "yolo", "--prompt", "bird"], "--prompt only applies"),
        (["--backend", "yolo", "--min-size", "640"], "--min-size"),
        (["--imgsz", "1280"], "--imgsz only applies to --backend yolo"),
        (["--backend", "sam3", "--prompt", "bird", "--imgsz", "1280"], "--imgsz only"),
    ],
)
def test_flags_of_another_backend_are_refused(
    tmp_path, monkeypatch, capsys, fake_ultralytics, flags, complaint
):
    video = make_video(tmp_path / "clip.avi")
    with pytest.raises(SystemExit):
        run_exporter(
            monkeypatch, *flags, "--videos", video, "--weights", fake_ultralytics,
            "--output-dir", tmp_path / "out",
        )  # fmt: skip
    assert complaint in capsys.readouterr().err


# ------------------------------------------------------- capability and bundle


def test_the_gpu_report_answers_for_yolo_separately():
    from idtrackerai.extra_tools.detectron2_pipeline.gpu import GpuReport

    ready = GpuReport(
        torch_version="2.7", ultralytics_version="8.4.0", cuda_available=True, checked=True
    )
    assert ready.yolo_usable and ready.yolo_missing == []
    # neither Detectron2 nor SAM 3 is needed, and neither is implied
    assert not ready.usable and not ready.sam3_usable

    bare = GpuReport(torch_version="2.7", checked=True)
    assert not bare.yolo_usable
    assert bare.yolo_missing == [
        "PyTorch cannot see a CUDA GPU",
        "the ultralytics package is not installed",
    ]


def test_the_probe_looks_for_ultralytics():
    from idtrackerai.extra_tools.detectron2_pipeline import gpu

    report = gpu.describe_gpu()
    assert report.checked
    import importlib.util

    installed = importlib.util.find_spec("ultralytics") is not None
    assert (report.ultralytics_version is not None) == installed


def test_the_colab_bundle_carries_the_yolo_backend(tmp_path):
    import zipfile

    from idtrackerai.extra_tools.detectron2_pipeline import bundle

    path = bundle.build(tmp_path / "bundle.zip")
    with zipfile.ZipFile(path) as zf:
        assert "tools/yolo_predictor.py" in zf.namelist()


def test_the_backend_loads_by_path_as_it_does_in_the_bundle(tmp_path, fake_ultralytics):
    """On Colab the files sit in one flat folder with no package around them."""
    import importlib.util

    source = Path(sm.__file__).with_name("yolo_predictor.py")
    target = tmp_path / "flat" / "yolo_predictor.py"
    target.parent.mkdir()
    target.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("flat_yolo_predictor", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = module.YoloPredictor(weights=fake_ultralytics, device="cpu")
    assert predictor.description == "yolo (best.pt)"
