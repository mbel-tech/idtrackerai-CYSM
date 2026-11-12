"""The frame enhancement (CLAHE) panel is offered only where it does something.

Thresholding separates animals from background by contrast, so it is the mode
enhancement was made for, and a Detectron2 model is trained on enhanced frames
(step 1 of its panel). SAM 3 and YOLO models were trained on frames as
recorded, and contour files are already made: there the panel is noise. It
still shows when the session has enhancement switched on, because then it is
in effect and must not be hidden from whoever is about to track with it.
"""

import os

import pytest

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "pen5.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(20):
        writer.write(np.full((120, 160, 3), 40 + i, np.uint8))
    writer.release()
    return path


def open_app(video, **parameters):
    from idtrackerai import Session
    from idtrackerai.segmentation_app import SegmentationGUI

    session = Session()
    session.set_parameters(video_paths=[str(video)], number_of_animals=9, **parameters)
    window = SegmentationGUI(session)
    window.load_parameters()
    return window


def shown(window) -> bool:
    return window.enhancement.isVisibleTo(window)


def test_the_panel_follows_the_segmentation_mode(qt_app, video):
    from idtrackerai.segmentation_app.widgets import segmentation_source as src

    window = open_app(video)
    try:
        expected = {
            src.THRESHOLDING: True,
            src.DETECTRON2: True,
            src.SAM3: False,
            src.YOLO: False,
        }
        for mode, visible in expected.items():
            window.segmentation_source.source.setCurrentText(mode)
            assert shown(window) is visible, mode
        window.segmentation_source.source.setCurrentText(src.THRESHOLDING)
        assert shown(window)
    finally:
        window.close()


def test_a_yolo_session_opens_without_the_panel(qt_app, video):
    window = open_app(video, segmentation_model={"backend": "yolo", "weights": "best.pt"})
    try:
        assert not shown(window)
        assert "enhancement" not in window.out_parameters()
    finally:
        window.close()


def test_enhancement_that_is_on_is_never_hidden(qt_app, video):
    """A session saved with enhancement keeps applying it, so it stays in view."""
    window = open_app(
        video,
        segmentation_model={"backend": "sam3", "weights": "sam3.pt", "prompt": "bird"},
        enhancement={"enhance": True, "clahe_clip": 2.0},
    )
    try:
        assert shown(window)
        assert window.out_parameters()["enhancement"]["enhance"] is True
    finally:
        window.close()
