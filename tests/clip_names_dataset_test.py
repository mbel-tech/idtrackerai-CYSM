"""The training split reads the clip names that were recorded with the frames.

When clips share a file name, sampling names them from their folders
(day1-cam1, day2-cam1). The split already groups on the recorded name; these
tests pin down that the notebook can preview the grouping the split will really
use, from the same folder, before anything is built.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline.dataset import (
    DatasetRequest,
    build_coco_dataset,
    group_videos,
    sources_in_folder,
)
from idtrackerai.extra_tools.detectron2_pipeline.prelabel import labelme_document
from idtrackerai.extra_tools.detectron2_pipeline.preprocessing import DEFAULT_SETTINGS

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/extra_tools/detectron2_pipeline/notebooks"
    / "colab_detectron2_pipeline.ipynb"
)


def make_video(path: Path, frames: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(frames):
        writer.write(np.full((48, 64, 3), 30 + i, np.uint8))
    writer.release()
    return path


def sample(videos, out: Path, n: int = 8, seed: int = 1):
    return sampling_mod.sample_frames(videos, out, n, dict(DEFAULT_SETTINGS), seed=seed)


def two_days(tmp_path: Path) -> Path:
    """Frames sampled from two clips that share the file name cam1.avi."""
    out = tmp_path / "frames"
    sample(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            make_video(tmp_path / "day2" / "cam1.avi"),
        ],
        out,
    )
    return out


def touch_png(path: Path) -> Path:
    path.write_bytes(b"")
    return path


# ------------------------------------------------------------ reading sources
def test_sources_come_from_the_manifest(tmp_path):
    out = two_days(tmp_path)

    assert sources_in_folder(out) == ["day1-cam1", "day2-cam1"]


def test_the_manifest_names_the_frames_it_records_and_file_names_the_rest(tmp_path):
    touch_png(tmp_path / "a_f000001.png")
    touch_png(tmp_path / "b_f000002.png")
    recorded = {"frames": [{"file_name": "a_f000001.png", "source_video": "day1-a"}]}
    (tmp_path / "sampling_manifest.json").write_text(
        json.dumps(recorded), encoding="utf-8"
    )

    assert sources_in_folder(tmp_path) == ["b", "day1-a"]


def test_without_a_manifest_the_file_name_gives_the_source(tmp_path):
    touch_png(tmp_path / "day1-cam1_f000012.png")

    assert sources_in_folder(tmp_path) == ["day1-cam1"]


def test_a_malformed_manifest_falls_back_to_file_names(tmp_path):
    touch_png(tmp_path / "day1-cam1_f000012.png")
    touch_png(tmp_path / "day2-cam1_f000003.png")
    (tmp_path / "sampling_manifest.json").write_text("{not json", encoding="utf-8")

    assert sources_in_folder(tmp_path) == ["day1-cam1", "day2-cam1"]


def test_non_image_files_are_ignored(tmp_path):
    touch_png(tmp_path / "day1-cam1_f000012.png")
    (tmp_path / "day2-cam1_f000003.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes_f000001.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sampling_manifest.json").write_text("{}", encoding="utf-8")

    assert sources_in_folder(tmp_path) == ["day1-cam1"]


# ----------------------------------------------------------- grouping the names
def test_pieces_of_one_qualified_recording_stay_together(tmp_path):
    out = tmp_path / "frames"
    sample(
        [
            make_video(tmp_path / day / f"cam1_segment_{piece}.avi")
            for day in ("day1", "day2")
            for piece in (1, 2)
        ],
        out,
        n=12,
    )

    sources = sources_in_folder(out)
    groups = group_videos(sources)

    assert sorted(groups) == ["day1-cam1", "day2-cam1"]
    assert all(len(members) == 2 for members in groups.values()), sources


def test_the_split_keeps_each_recording_whole(tmp_path):
    out = two_days(tmp_path)
    for frame in out.glob("*.png"):
        height, width = cv2.imread(str(frame)).shape[:2]
        polygon = np.array([[4, 4], [40, 4], [40, 30], [4, 30]])
        document = labelme_document(frame, width, height, [polygon], "fish")
        frame.with_suffix(".json").write_text(json.dumps(document), encoding="utf-8")

    report = build_coco_dataset(
        DatasetRequest(
            input_dir=out,
            output_dir=tmp_path / "dataset",
            val_fraction=0.5,
            single_class="fish",
        )
    )

    assert not report.problems, report.problems
    assert len(report.train_videos) == 1
    assert len(report.val_videos) == 1
    assert not set(report.train_videos) & set(report.val_videos)
    assert set(report.train_videos) | set(report.val_videos) == {
        "day1-cam1",
        "day2-cam1",
    }


# ------------------------------------------------------------------- the notebook
def test_the_notebook_previews_the_recorded_names():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    previews = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "before it is acted on" in "".join(cell["source"])
    ]

    assert len(previews) == 1
    assert "sources_in_folder(FRAMES)" in previews[0]
    assert "v.stem" not in previews[0]
