"""The YOLO mode of the Segmentation App, checked without a display or a GPU.

Pins what is easy to break silently: that the mode is offered and opens on the
right panel, that nothing runs before it could work, that what the panel
writes to the session .toml is what comes back from it, and that the export
batch can be interrupted and resumed.
"""

import os

import pytest
import toml

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm  # noqa: E402

TOOLTIPS = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/segmentation_app/tooltips.toml"
)


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def report(cuda=False, ultralytics=None, torch="2.7.0"):
    from idtrackerai.extra_tools.detectron2_pipeline.gpu import GpuReport

    return GpuReport(
        torch_version=torch,
        ultralytics_version=ultralytics,
        cuda_available=cuda,
        device_name="Test GPU" if cuda else None,
        checked=True,
    )


@pytest.fixture
def panel(qt_app):
    from idtrackerai.segmentation_app.widgets import EnhancementWidget, YoloPanel

    widget = YoloPanel(EnhancementWidget())
    widget.gpu_thread.wait(30000)
    widget.gpu = report()
    widget._refresh_enabled()
    yield widget
    widget.close()


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "pen5.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(20):
        writer.write(np.full((120, 160, 3), 40 + i, np.uint8))
    writer.release()
    return path


# ------------------------------------------------------------------ the mode


def test_the_selector_offers_yolo_last(qt_app):
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        YOLO,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    modes = [widget.source.itemText(i) for i in range(widget.source.count())]
    assert modes[-1] == YOLO == "YOLO"


def test_choosing_yolo_does_not_open_a_file_dialog(qt_app):
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        YOLO,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    emitted = []
    widget.valueChanged.connect(emitted.append)
    widget.source.setCurrentText(YOLO)

    assert widget.mode() == YOLO
    assert emitted == [None]


# ------------------------------------------------------------ what is offered


def test_nothing_runs_before_weights_are_chosen(panel, tmp_path):
    panel.set_video_context([tmp_path / "clip.mp4"])
    panel.gpu = report(cuda=True, ultralytics="8.4.0")
    panel._refresh_enabled()
    assert not panel.export_button.isEnabled()


def test_export_needs_ultralytics_but_not_a_gpu(panel, tmp_path):
    """YOLO on a CPU is slow, not impossible, so it is offered with a warning;
    without the package there is nothing to run at all."""
    panel._set_weights(tmp_path / "best.pt")
    panel.set_video_context([tmp_path / "clip.mp4"])
    assert not panel.export_button.isEnabled()
    assert "ultralytics package is not installed" in panel.gpu_status.text()

    panel.gpu = report(ultralytics="8.4.0")
    panel._refresh_enabled()
    assert panel.export_button.isEnabled()
    assert panel.colab_hint.isVisibleTo(panel)

    panel.gpu = report(cuda=True, ultralytics="8.4.0")
    panel._refresh_enabled()
    assert panel.export_button.isEnabled()
    assert not panel.colab_hint.isVisibleTo(panel)


def test_importing_label_files_needs_only_a_video(panel, tmp_path):
    panel.set_video_context([tmp_path / "clip.mp4"])
    assert panel.import_button.isEnabled()


# -------------------------------------------------------------------- the table


def test_the_panel_writes_a_yolo_table(panel, tmp_path):
    assert panel.model_spec() is None  # no weights, no model
    panel._set_weights(tmp_path / "best.pt")
    panel.set_video_context([tmp_path / "clip.mp4"], tmp_path / "session")
    panel.score_threshold.setValue(0.35)

    spec = panel.model_spec()

    assert spec == {
        "backend": "yolo",
        "weights": str(tmp_path / "best.pt"),
        "score_threshold": 0.35,
        "contours_dir": str(tmp_path / "session" / "yolo_contours"),
    }
    # left out, the cap is the session's number of animals and the image size
    # the model's own
    normal = sm.normalize_model(spec)
    assert normal["max_instances"] is None and normal["imgsz"] is None


def test_image_size_and_cap_are_saved_once_set(panel, tmp_path):
    panel._set_weights(tmp_path / "best.pt")
    panel.imgsz.setValue(1280)
    panel.max_instances.setValue(12)
    spec = panel.model_spec()
    assert spec["imgsz"] == 1280 and spec["max_instances"] == 12


def test_a_table_from_the_toml_fills_the_panel(panel, tmp_path):
    panel.set_model_spec(
        {
            "backend": "yolo",
            "weights": "/drive/best.pt",
            "score_threshold": 0.4,
            "imgsz": 960,
            "contours_dir": str(tmp_path / "c"),
            "on_overlap": "split",
        }
    )
    assert str(panel.weights) == "/drive/best.pt"
    assert panel.score_threshold.value() == pytest.approx(0.4)
    assert panel.imgsz.value() == 960
    assert panel.max_instances.value() == 0  # "number of animals"
    assert panel.export_dir() == tmp_path / "c"
    # keys with no control here survive the round trip
    assert panel.model_spec()["on_overlap"] == "split"


def test_another_backends_table_is_ignored(panel):
    panel.set_model_spec({"backend": "sam3", "weights": "sam3.pt", "prompt": "bird"})
    assert panel.weights is None and panel.model_spec() is None


def test_a_yolo_toml_opens_on_the_yolo_panel_and_saves_it_back(qt_app, video, tmp_path):
    from idtrackerai import Session
    from idtrackerai.segmentation_app import SegmentationGUI
    from idtrackerai.segmentation_app.widgets import segmentation_source

    table = {
        "backend": "yolo",
        "weights": "best.pt",
        "contours_dir": str(tmp_path / "contours"),
    }
    session = Session()
    session.set_parameters(
        video_paths=[str(video)], number_of_animals=9, segmentation_model=table
    )
    window = SegmentationGUI(session)
    window.load_parameters()
    try:
        assert window.segmentation_source.mode() == segmentation_source.YOLO
        assert window.yolo_panel.isVisibleTo(window)
        assert not window.sam3_panel.isVisibleTo(window)
        assert window.yolo_panel.number_of_animals == 9
        saved = window.out_parameters()["segmentation_model"]
        assert saved["backend"] == "yolo" and saved["weights"] == "best.pt"
        assert saved["contours_dir"] == table["contours_dir"]

        window.segmentation_source.source.setCurrentText(
            segmentation_source.THRESHOLDING
        )
        assert "segmentation_model" not in window.out_parameters()
    finally:
        window.close()


def test_loaded_yolo_contours_keep_their_model_in_the_toml(qt_app, video, tmp_path):
    """With contour files loaded, the table saved is the YOLO panel's when the
    files say YOLO made them -- not Detectron2's, the default for a file."""
    from idtrackerai import Session
    from idtrackerai.base.animals_detection.external_contours import write_contours
    from idtrackerai.segmentation_app import SegmentationGUI
    from idtrackerai.segmentation_app.widgets import segmentation_source

    contours = tmp_path / "pen5.h5"
    write_contours(contours, [[] for _ in range(20)], 160, 120, backend="yolo")
    session = Session()
    session.set_parameters(video_paths=[str(video)], number_of_animals=9)
    window = SegmentationGUI(session)
    window.load_parameters()
    try:
        window.yolo_panel._set_weights(tmp_path / "best.pt")
        assert window.segmentation_source.accept_paths([contours])
        assert window.segmentation_source.mode() == segmentation_source.EXTERNAL
        assert window.out_parameters()["segmentation_model"]["backend"] == "yolo"
    finally:
        window.close()


# ---------------------------------------------------------------- the command


def test_the_command_exports_what_the_panel_would(panel, tmp_path):
    panel._set_weights(tmp_path / "best.pt")
    panel.set_video_context([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "s")
    panel.set_number_of_animals(9)
    panel.score_threshold.setValue(0.4)
    panel.imgsz.setValue(1280)

    command = panel.export_command.toPlainText()

    assert "--backend yolo" in command
    assert f'--weights "{tmp_path / "best.pt"}"' in command
    assert f'"{tmp_path / "a.mp4"}" "{tmp_path / "b.mp4"}"' in command
    assert "--score-threshold 0.4" in command
    assert "--max-instances 9" in command
    assert "--imgsz 1280" in command
    assert f'--output-dir "{tmp_path / "s" / "yolo_contours"}"' in command
    assert "--prompt" not in command


def test_the_command_leaves_image_size_to_the_model_by_default(panel, tmp_path):
    panel._set_weights(tmp_path / "best.pt")
    assert "--imgsz" not in panel.export_command.toPlainText()
    assert "<best.pt>" not in panel.export_command.toPlainText()
    panel._set_weights(None)
    assert '"<best.pt>"' in panel.export_command.toPlainText()


def test_every_control_has_a_tooltip(panel):
    tips = toml.load(TOOLTIPS)
    wanted = {
        "segmentation_source",
        "yolo_weights",
        "yolo_score_threshold",
        "yolo_max_instances",
        "yolo_imgsz",
        "yolo_export",
        "yolo_reexport",
        "yolo_contours_dir",
        "yolo_command",
        "yolo_import",
    }
    assert wanted <= set(tips)
    assert "YOLO" in tips["segmentation_source"]
    panel.setToolTips(tips)
    for widget in (
        panel.weights_button, panel.score_threshold, panel.max_instances,
        panel.imgsz, panel.export_button, panel.reexport, panel.contours_button,
        panel.export_command, panel.import_button,
    ):  # fmt: skip
        assert widget.toolTip()


# ------------------------------------------------------------ the export batch


@pytest.fixture
def export_thread(monkeypatch, tmp_path):
    """A YoloExportThread with the model and the exporter stubbed out; run()
    is called directly, since this is about the batch and not the threading."""
    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod
    from idtrackerai.extra_tools.detectron2_pipeline import yolo_predictor
    from idtrackerai.segmentation_app.widgets.pipeline_threads import YoloExportThread

    exported = []

    def fake_export_video(video, output, args, *rest, **kwargs):
        exported.append((Path(video).name, args))
        Path(output).write_bytes(b"contours")
        return {"video": Path(video).name, "frames": 1}

    monkeypatch.setattr(inference_mod, "export_video", fake_export_video)
    monkeypatch.setattr(inference_mod, "load_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(inference_mod, "weights_sha256", lambda weights: "fake")
    monkeypatch.setattr(inference_mod, "stale_reason", lambda *a, **k: None)
    monkeypatch.setattr(inference_mod, "video_frame_count", lambda video: 100)
    built = []

    def fake_predictor(**kwargs):
        built.append(kwargs)
        return type("P", (), {"description": "yolo (best.pt)"})()

    monkeypatch.setattr(yolo_predictor, "YoloPredictor", fake_predictor)
    return YoloExportThread(), exported, built, tmp_path, monkeypatch


def make_clips(folder, names):
    clips = []
    for name in names:
        path = folder / name
        path.write_bytes(b"video")
        clips.append(path)
    return clips


def start(thread, tmp_path, clips, out, **overrides):
    options = dict(
        videos=clips, output_dir=out, weights=tmp_path / "best.pt", device="cpu",
        score_threshold=0.4, max_instances=9, imgsz=1280, enhancement={"enhance": False},
    )  # fmt: skip
    options.update(overrides)
    thread.set_parameters(**options)
    thread.run()


def test_the_export_runs_yolo_with_the_panels_settings(export_thread):
    thread, exported, built, tmp_path, _ = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    start(thread, tmp_path, make_clips(tmp_path, ["a.mp4"]), out,
          tracking_intervals=[[10, 60]])  # fmt: skip

    assert built == [
        dict(weights=tmp_path / "best.pt", device="cpu", score_threshold=0.4,
             max_instances=9, imgsz=1280, classes=None)
    ]  # fmt: skip
    name, args = exported[0]
    assert name == "a.mp4"
    assert args.backend == "yolo" and args.imgsz == 1280 and args.max_instances == 9
    assert args.prompt is None
    assert args.intervals == [[10, 60]]
    assert [p.name for p in thread.written] == ["a.h5"]


def test_clips_that_already_have_contours_are_not_redone(export_thread):
    thread, exported, _, tmp_path, _ = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    clips = make_clips(tmp_path, ["a.mp4", "b.mp4"])
    (out / "a.h5").write_bytes(b"done earlier")

    start(thread, tmp_path, clips, out)

    assert [name for name, _ in exported] == ["b.mp4"]
    assert [p.name for p in thread.skipped] == ["a.h5"]
    assert [p.name for p in thread.written] == ["a.h5", "b.h5"]


def test_weights_that_cannot_segment_are_reported_not_raised(export_thread):
    from idtrackerai.extra_tools.detectron2_pipeline import yolo_predictor

    thread, exported, _, tmp_path, monkeypatch = export_thread

    def refuse(**kwargs):
        raise yolo_predictor.YoloWeightsError("best.pt is a YOLO detect model")

    monkeypatch.setattr(yolo_predictor, "YoloPredictor", refuse)
    failures = []
    thread.failed.connect(failures.append)
    start(thread, tmp_path, make_clips(tmp_path, ["a.mp4"]), tmp_path)

    assert exported == []
    assert failures == ["best.pt is a YOLO detect model"]


# -------------------------------------------------------- importing label files


def test_label_files_become_this_sessions_contours(panel, video, tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "pen5_1.txt").write_text(
        "0 0.25 0.25 0.5 0.25 0.5 0.75 0.25 0.75 0.9\n", encoding="utf-8"
    )
    panel.set_video_context([video], tmp_path / "session")
    panel.set_number_of_animals(3)
    ready = []
    panel.contoursReady.connect(ready.append)

    problem = panel.import_labels(labels)

    assert problem is None
    written = tmp_path / "session" / "yolo_contours" / "pen5.h5"
    assert ready == [[written]] and written.is_file()


def test_a_folder_that_does_not_fit_is_explained(panel, video, tmp_path):
    empty = tmp_path / "labels"
    empty.mkdir()
    panel.set_video_context([video], tmp_path / "session")
    ready = []
    panel.contoursReady.connect(ready.append)

    problem = panel.import_labels(empty)

    assert "pen5_<frame>.txt" in problem
    assert ready == []
