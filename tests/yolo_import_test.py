"""Importing YOLO segmentation label files as a contour sidecar.

When YOLO already ran somewhere else with ``save_txt``, its outlines are on
disk as one text file per frame. The importer turns a folder of those into the
file the External contours route reads, so the model does not have to run
again. What matters is that each outline lands on the right frame at the right
size, and that files which cannot give outlines are refused out loud.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from idtrackerai.base.animals_detection.external_contours import ExternalContours
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm
from idtrackerai.extra_tools.detectron2_pipeline import yolo_import

HEIGHT, WIDTH = 100, 200
N_FRAMES = 6

SQUARE = "0 0.25 0.25 0.5 0.25 0.5 0.75 0.25 0.75"  # x 50..100, y 25..75
SMALL = "0 0.6 0.1 0.7 0.1 0.7 0.3 0.6 0.3"


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "pen5.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(N_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 40 + i, np.uint8))
    writer.release()
    return path


def labels(tmp_path, files: dict[str, str]) -> Path:
    folder = tmp_path / "labels"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_polygons_come_back_in_pixels(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text(SQUARE + " 0.83\n", encoding="utf-8")

    contours, scores, classes = yolo_import.read_label_file(path, WIDTH, HEIGHT)

    assert scores == [pytest.approx(0.83)] and classes == [0]
    assert contours[0].tolist() == [[50, 25], [100, 25], [100, 75], [50, 75]]


def test_a_line_without_confidence_scores_one(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text(SQUARE + "\n", encoding="utf-8")
    _, scores, _ = yolo_import.read_label_file(path, WIDTH, HEIGHT)
    assert scores == [1.0]


@pytest.mark.parametrize("line", ["0 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1 0.1 0.9"])
def test_box_labels_are_refused_with_the_reason(tmp_path, line):
    """A detection model's files hold boxes. A box is not an outline, and
    importing it as one would hand idtracker.ai rectangles of litter."""
    path = tmp_path / "one.txt"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(yolo_import.YoloImportError, match="boxes, not outlines"):
        yolo_import.read_label_file(path, WIDTH, HEIGHT)


def test_each_file_lands_on_its_own_frame(tmp_path, video):
    """ultralytics numbers a video's label files from 1; frames here from 0."""
    folder = labels(
        tmp_path,
        {"pen5_1.txt": SQUARE + " 0.9\n", "pen5_4.txt": SQUARE + " 0.9\n" + SMALL + " 0.8\n"},
    )
    output = tmp_path / "pen5.h5"

    stats = yolo_import.import_video(video, folder, output)

    contours = ExternalContours(output)
    counts = [len(contours.contours_in_frame(i)) for i in range(N_FRAMES)]
    assert counts == [1, 0, 0, 2, 0, 0]
    assert (contours.width, contours.height) == (WIDTH, HEIGHT)
    assert contours.attrs["backend"] == "yolo"
    contours.close()
    assert stats["label_files"] == 2 and stats["contours"] == 3


def test_zero_based_files_can_be_said_so(tmp_path, video):
    folder = labels(tmp_path, {"pen5_0.txt": SQUARE + "\n"})
    output = tmp_path / "pen5.h5"
    yolo_import.import_video(video, folder, output, first_index=0)
    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 1
    contours.close()


def test_threshold_and_cap_are_applied_and_recorded(tmp_path, video):
    folder = labels(
        tmp_path,
        {"pen5_1.txt": SQUARE + " 0.9\n" + SMALL + " 0.3\n" + SMALL + " 0.6\n"},
    )
    output = tmp_path / "pen5.h5"

    yolo_import.import_video(
        video, folder, output, score_threshold=0.5, max_instances=1,
        weights_name="best.pt",
    )

    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 1
    attrs = dict(contours.attrs)
    contours.close()
    # so a [segmentation_model] table can be checked against the file
    spec = sm.normalize_model(
        {"backend": "yolo", "weights": "/drive/best.pt", "score_threshold": 0.5}
    )
    assert sm.mismatches(attrs, spec, 1) == []
    assert sm.mismatches(attrs, spec | {"score_threshold": 0.7}, 1)


def test_other_classes_can_be_left_out(tmp_path, video):
    feeder = "1" + SMALL[1:]
    folder = labels(tmp_path, {"pen5_1.txt": SQUARE + "\n" + feeder + "\n"})
    output = tmp_path / "pen5.h5"
    yolo_import.import_video(video, folder, output, classes=[0])
    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 1
    contours.close()


def test_a_folder_with_no_files_for_the_video_is_an_error(tmp_path, video):
    folder = labels(tmp_path, {"otherpen_1.txt": SQUARE + "\n"})
    with pytest.raises(yolo_import.YoloImportError, match="pen5_<frame>.txt"):
        yolo_import.import_video(video, folder, tmp_path / "pen5.h5")
    assert not (tmp_path / "pen5.h5").exists()


def test_files_beyond_the_videos_end_are_an_error(tmp_path, video):
    """They mean the labels were made from another video, or another frame
    rate; importing the part that fits would hide that."""
    folder = labels(tmp_path, {"pen5_1.txt": SQUARE + "\n", "pen5_99.txt": SQUARE + "\n"})
    with pytest.raises(yolo_import.YoloImportError, match="frame 98"):
        yolo_import.import_video(video, folder, tmp_path / "pen5.h5")


def test_the_command_line_imports_a_video(tmp_path, video, monkeypatch, capsys):
    import sys

    folder = labels(tmp_path, {"pen5_2.txt": SQUARE + " 0.7\n"})
    output = tmp_path / "out" / "pen5.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        ["idtrackerai_yolo_import", "--video", str(video), "--labels", str(folder),
         "--output", str(output), "--weights-name", "best.pt"],
    )  # fmt: skip
    yolo_import.main()

    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(1)) == 1
    assert contours.attrs["weights"] == "best.pt"
    contours.close()
    assert "1 label file" in capsys.readouterr().out
