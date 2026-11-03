"""The [segmentation_model] table in the Segmentation App.

Weights you already have can be chosen in the Detectron2 panel's export step or
the SAM 3 panel, are written to the session .toml, and come back when that
.toml is opened again. Checked offscreen, without a GPU.
"""

import os

import pytest

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "GX010561.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(20):
        writer.write(np.full((120, 160, 3), 40 + i, np.uint8))
    writer.release()
    return path


@pytest.fixture
def d2_panel(qt_app):
    from idtrackerai.segmentation_app.widgets import Detectron2Panel, EnhancementWidget

    panel = Detectron2Panel(EnhancementWidget())
    panel.gpu_thread.wait(30000)
    yield panel
    panel.close()


@pytest.fixture
def sam3_panel(qt_app):
    from idtrackerai.segmentation_app.widgets import EnhancementWidget, Sam3Panel

    panel = Sam3Panel(EnhancementWidget())
    panel.gpu_thread.wait(30000)
    yield panel
    panel.close()


def test_detectron2_panel_takes_weights_you_already_have(d2_panel, tmp_path):
    assert d2_panel.model_spec() is None  # nothing chosen, nothing trained

    weights = tmp_path / "model_final.pth"
    d2_panel.weights_path.setText(str(weights))
    d2_panel.contours_dir.setText(str(tmp_path / "contours"))
    d2_panel.score_threshold.setValue(0.6)
    d2_panel.set_number_of_animals(9)

    spec = d2_panel.model_spec()
    assert spec == {
        "backend": "detectron2",
        "weights": str(weights),
        "score_threshold": 0.6,
        "contours_dir": str(tmp_path / "contours"),
    }
    assert sm.normalize_model(spec)  # valid as written
    command = d2_panel.export_command.toPlainText()
    assert f'--weights "{weights}"' in command
    assert "--max-instances 9" in command
    assert "--score-threshold 0.6" in command


def test_detectron2_panel_restores_the_table(d2_panel):
    d2_panel.set_model_spec(
        {
            "weights": "/drive/model_final.pth",
            "score_threshold": 0.8,
            "contours_dir": "/drive/contours/p1",
            "min_size": 800,
        }
    )
    assert d2_panel.weights_path.text() == "/drive/model_final.pth"
    assert d2_panel.score_threshold.value() == pytest.approx(0.8)
    assert d2_panel.steps.currentIndex() == 5  # opens on the export step
    # keys the panel has no control for survive the round trip
    assert d2_panel.model_spec()["min_size"] == 800


def test_each_panel_ignores_the_other_backend(d2_panel, sam3_panel):
    sam3_table = {"backend": "sam3", "weights": "sam3.pt", "prompt": "chicken"}
    d2_panel.set_model_spec(sam3_table)
    assert d2_panel.weights_path.text() == ""
    sam3_panel.set_model_spec({"weights": "model_final.pth"})
    assert sam3_panel.weights is None


def test_sam3_panel_writes_and_restores_the_table(sam3_panel, tmp_path):
    assert sam3_panel.model_spec() is None
    sam3_panel.set_model_spec(
        {
            "backend": "sam3",
            "weights": str(tmp_path / "sam3.pt"),
            "prompt": "chicken",
            "score_threshold": 0.4,
            "max_instances": 0,
            "contours_dir": str(tmp_path / "c"),
        }
    )
    assert sam3_panel.prompt.text() == "chicken"
    assert sam3_panel.export_dir() == tmp_path / "c"
    spec = sam3_panel.model_spec()
    assert spec["backend"] == "sam3" and spec["max_instances"] == 0
    assert sm.normalize_model(spec)["prompt"] == "chicken"


@pytest.mark.parametrize(
    "table, mode_name",
    [
        ({"weights": "model_final.pth"}, "DETECTRON2"),
        ({"backend": "sam3", "weights": "sam3.pt", "prompt": "chicken"}, "SAM3"),
    ],
)
def test_a_toml_with_a_model_opens_on_its_panel_and_saves_it_back(
    qt_app, video, tmp_path, table, mode_name
):
    from idtrackerai import Session
    from idtrackerai.segmentation_app import SegmentationGUI
    from idtrackerai.segmentation_app.widgets import segmentation_source

    table = table | {"contours_dir": str(tmp_path / "contours")}
    session = Session()
    session.set_parameters(
        video_paths=[str(video)],
        number_of_animals=9,
        segmentation_model=table,
    )
    window = SegmentationGUI(session)
    window.load_parameters()  # normally queued for the event loop
    try:
        assert window.segmentation_source.mode() == getattr(
            segmentation_source, mode_name
        )
        saved = window.out_parameters()["segmentation_model"]
        assert saved["backend"] == table.get("backend", "detectron2")
        assert saved["weights"] == table["weights"]
        assert saved["contours_dir"] == table["contours_dir"]

        # choosing thresholding drops the model from what is saved
        window.segmentation_source.source.setCurrentText(
            segmentation_source.THRESHOLDING
        )
        assert "segmentation_model" not in window.out_parameters()
    finally:
        window.close()
