"""The "Save Colab notebook..." button of the three model panels.

Every route that runs a segmentation model offers the same thing: the session
as a notebook that needs only a Drive folder. Checked without a display.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

TOOLTIPS = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/segmentation_app/tooltips.toml"
)
PANELS = {
    "yolo": ("YoloPanel", "yolo_notebook"),
    "sam3": ("Sam3Panel", "sam3_notebook"),
    "detectron2": ("Detectron2Panel", "d2_notebook"),
}


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture(params=list(PANELS))
def backend(request):
    return request.param


@pytest.fixture
def panel(qt_app, backend):
    from idtrackerai.segmentation_app import widgets

    widget = getattr(widgets, PANELS[backend][0])(widgets.EnhancementWidget())
    widget.gpu_thread.wait(30000)
    yield widget
    widget.close()


def choose_weights(panel, backend, folder: Path) -> str:
    """Sets the panel's model the way its own controls do; returns the name."""
    if backend == "yolo":
        panel.set_model_spec({"backend": "yolo", "weights": str(folder / "best.pt")})
        return "best.pt"
    if backend == "sam3":
        panel.set_model_spec(
            {"backend": "sam3", "weights": str(folder / "sam3.pt"), "prompt": "bird"}
        )
        return "sam3.pt"
    panel.weights_path.setText(str(folder / "model_final.pth"))
    return "model_final.pth"


def session(folder: Path) -> dict:
    return {
        "name": "R2_01_10_P5",
        "video_paths": [str(folder / "R2_01_10_P5_SB_fps24.mp4")],
        "number_of_animals": 9,
        "track_wo_identities": False,
        "tracking_intervals": None,
        "roi_list": ["+ Polygon [[5, 5], [100, 5], [100, 100]]"],
        "output_dir": str(folder / "out"),
    }


def test_every_model_panel_has_the_button(panel):
    assert panel.notebook_button.text() == "Save Colab notebook..."


def test_the_button_has_a_tooltip(panel, backend):
    import tomllib

    tips = tomllib.loads(TOOLTIPS.read_text(encoding="utf-8"))
    key = PANELS[backend][1]
    assert "Drive" in tips[key]
    panel.setToolTips(tips)
    assert panel.notebook_button.toolTip() == tips[key]


def test_it_writes_this_sessions_notebook_with_this_panels_model(
    panel, backend, tmp_path
):
    weights = choose_weights(panel, backend, tmp_path)
    panel.session_parameters = lambda: session(tmp_path)
    target = tmp_path / "run.ipynb"

    problem, files = panel.save_colab_notebook(target)

    assert problem is None
    assert files == [weights, "R2_01_10_P5_SB_fps24.mp4"]
    notebook = json.loads(target.read_text(encoding="utf-8"))
    settings = next(
        "".join(c["source"]) for c in notebook["cells"] if c["id"] == "settings"
    )
    assert f'"backend": "{backend}"' in settings
    assert weights in settings and "R2_01_10_P5" in settings
    assert str(tmp_path) not in target.read_text(encoding="utf-8")


def test_without_a_session_it_says_what_is_missing(panel, backend, tmp_path):
    choose_weights(panel, backend, tmp_path)
    target = tmp_path / "run.ipynb"
    problem, _ = panel.save_colab_notebook(target)  # no video open
    assert "video" in problem and not target.exists()


def test_without_weights_it_says_so(qt_app, tmp_path):
    from idtrackerai.segmentation_app import widgets

    panel = widgets.YoloPanel(widgets.EnhancementWidget())
    panel.gpu_thread.wait(30000)
    panel.session_parameters = lambda: session(tmp_path)
    target = tmp_path / "run.ipynb"
    problem, _ = panel.save_colab_notebook(target)
    panel.close()
    assert "weights" in problem and not target.exists()


def test_without_a_number_of_animals_it_says_so(panel, backend, tmp_path):
    choose_weights(panel, backend, tmp_path)
    panel.session_parameters = lambda: session(tmp_path) | {"number_of_animals": 0}
    problem, _ = panel.save_colab_notebook(tmp_path / "run.ipynb")
    assert "number of animals" in problem


@pytest.fixture
def video(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "pen5.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(20):
        writer.write(np.full((120, 160, 3), 40 + i, np.uint8))
    writer.release()
    return path


def test_in_the_app_the_button_writes_the_open_session(qt_app, video, tmp_path):
    from idtrackerai import Session
    from idtrackerai.segmentation_app import SegmentationGUI

    session = Session()
    session.set_parameters(
        video_paths=[str(video)],
        number_of_animals=9,
        segmentation_model={"backend": "yolo", "weights": str(tmp_path / "best.pt")},
    )
    window = SegmentationGUI(session)
    window.load_parameters()
    try:
        target = tmp_path / "nb" / "run.ipynb"
        for panel in (window.detectron2_panel, window.sam3_panel, window.yolo_panel):
            assert panel.session_parameters() is not None
        problem, files = window.yolo_panel.save_colab_notebook(target)
        assert problem is None and files == ["best.pt", "pen5.avi"]
        saved = target.read_text(encoding="utf-8")
        assert '\\"name\\": \\"pen5\\"' in saved  # the session's name, set or not
        assert '\\"number_of_animals\\": 9' in saved
        assert str(tmp_path) not in saved
    finally:
        window.close()


def test_in_the_app_with_no_video_open_it_says_so(qt_app, tmp_path):
    from idtrackerai import Session
    from idtrackerai.segmentation_app import SegmentationGUI

    window = SegmentationGUI(Session())
    try:
        window.yolo_panel._set_weights(tmp_path / "best.pt")
        problem, _ = window.yolo_panel.save_colab_notebook(tmp_path / "run.ipynb")
        assert "video" in problem
    finally:
        window.close()
