"""The [segmentation_model] table of a session .toml, end to end.

Covers the table itself, the exporter's --session mode that reads it (with a
stand-in model, so no torch or detectron2), and the checks idtracker.ai makes
before tracking from the files that mode writes.
"""

import json
from argparse import Namespace
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from idtrackerai import IdtrackeraiError, Session
from idtrackerai.base.animals_detection.external_contours import (
    ExternalContours,
    write_contours,
)
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm
from idtrackerai.extra_tools.detectron2_pipeline.inference import (
    export_video,
    session_jobs,
    stale_reason,
)

HEIGHT, WIDTH = 120, 160
N_FRAMES = 12


# --------------------------------------------------------------------- table


def test_a_minimal_table_gets_the_exporter_defaults():
    spec = sm.normalize_model({"weights": "model_final.pth"})
    assert spec["backend"] == "detectron2"
    assert spec["score_threshold"] == 0.7
    assert spec["on_overlap"] == "merge"
    assert spec["max_instances"] is None
    assert set(spec) == set(sm.DEFAULTS)


def test_no_table_means_no_model():
    assert sm.normalize_model(None) is None


@pytest.mark.parametrize(
    "table, complaint",
    [
        ({"weights": "w.pth", "score_treshold": 0.5}, "score_treshold"),
        ({}, "weights is missing"),
        ({"weights": "w.pth", "backend": "yolo"}, "backend"),
        ({"weights": "sam3.pt", "backend": "sam3"}, "prompt"),
        ({"weights": "w.pth", "prompt": "chicken"}, "only applies"),
        (
            {"weights": "sam3.pt", "backend": "sam3", "prompt": "x", "min_size": 640},
            "min_size",
        ),
        ({"weights": "w.pth", "score_threshold": 1.5}, "at most 1"),
        ({"weights": "w.pth", "max_instances": 0}, "only for backend"),
        ({"weights": "w.pth", "max_instances": -1}, "at least 0"),
        ({"weights": "w.pth", "on_overlap": "fuse"}, "on_overlap"),
        ({"weights": "model_zoo"}, "classes"),
        ({"weights": "model_zoo", "classes": ["bird"]}, "class indices"),
        ({"weights": "w.pth", "min_size": "640"}, "number"),
        ("w.pth", "table"),
    ],
)
def test_bad_tables_are_named(table, complaint):
    with pytest.raises(sm.SegmentationModelError, match=complaint):
        sm.normalize_model(table)


def test_empty_strings_mean_unset_as_elsewhere_in_the_toml():
    spec = sm.normalize_model({"weights": "w.pth", "min_size": "", "config": ""})
    assert spec["min_size"] is None and spec["config"] is None


def test_coco_checkpoint_with_the_bird_class():
    spec = sm.normalize_model({"weights": "MODEL_ZOO", "classes": 14})
    assert spec["weights"] == sm.MODEL_ZOO
    assert spec["classes"] == [sm.COCO_BIRD]


def test_the_cap_follows_the_number_of_animals():
    spec = sm.normalize_model({"weights": "w.pth"})
    assert sm.effective_max_instances(spec, 11) == 11
    assert sm.effective_max_instances(spec | {"max_instances": 4}, 11) == 4
    with pytest.raises(sm.SegmentationModelError, match="number_of_animals"):
        sm.effective_max_instances(spec, 0)


def test_contours_dir_gives_one_file_per_video(tmp_path):
    spec = sm.normalize_model({"weights": "w.pth", "contours_dir": str(tmp_path)})
    paths = sm.default_contour_paths(spec, ["a/GX010561.MP4", "a/GX020561.MP4"])
    assert paths == [tmp_path / "GX010561.h5", tmp_path / "GX020561.h5"]
    with pytest.raises(sm.SegmentationModelError, match="same file name"):
        sm.default_contour_paths(spec, ["a/GX010561.MP4", "b/GX010561.MP4"])
    with pytest.raises(sm.SegmentationModelError, match="contours_dir"):
        sm.default_contour_paths(sm.normalize_model({"weights": "w"}), ["a.mp4"])


def test_global_intervals_split_into_each_videos_own_frames():
    # two chapters of 100 frames, tracked from 30 to 150
    assert sm.local_intervals([[30, 150]], [100, 100]) == [[[30, 100]], [[0, 50]]]
    assert sm.local_intervals(None, [100, 100]) == [[[0, 100]], [[0, 100]]]
    assert sm.local_intervals([10, 20], [100]) == [[[10, 20]]]
    # a chapter outside every interval needs nothing
    assert sm.local_intervals([[0, 50]], [100, 100]) == [[[0, 50]], []]


def test_coverage():
    assert sm.covers(None, [[0, 100]])
    assert sm.covers([[0, 50], [50, 100]], [[10, 90]])
    assert not sm.covers([[0, 50]], [[10, 90]])
    assert sm.covers([[0, 50]], [])


def test_provenance_mismatches_are_listed():
    spec = sm.normalize_model({"weights": "/drive/model_final.pth"})
    attrs = {
        "backend": "detectron2",
        "weights": "model_final.pth",
        "weights_sha256": "a" * 64,
        "score_threshold": 0.7,
        "max_instances": 9,
        "on_overlap": "merge",
    }
    assert sm.mismatches(attrs, spec, 9) == []
    assert sm.mismatches(attrs, spec, 9, "a" * 64) == []
    problems = sm.mismatches(
        attrs | {"score_threshold": 0.5, "max_instances": 5}, spec, 9, "b" * 64
    )
    assert len(problems) == 3
    assert any("other weights" in p for p in problems)
    assert any("score_threshold" in p for p in problems)
    assert any("max_instances is 5" in p for p in problems)


# ------------------------------------------------------------- the exporter


class FakePredictor:
    """One ellipse per frame, and a record of which frames it was asked about."""

    description = "fake"

    def __init__(self):
        self.frames_seen = []

    def predict(self, frame):
        # the frame's grey level encodes its index (see make_video)
        self.frames_seen.append(
            int(round((float(frame[HEIGHT // 2, WIDTH // 2].mean()) - 20) / 10))
        )
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.ellipse(mask, (80, 60), (14, 7), 0, 0, 360, 1, -1)
        return [mask], [0.9]


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


def make_args(**overrides):
    args = Namespace(
        backend="detectron2",
        prompt=None,
        weights="model_final.pth",
        weights_sha256="0" * 64,
        score_threshold=0.7,
        max_instances=1,
        min_component=0,
        dilate=0,
        dedup_iou=0.7,
        min_area=1.0,
        on_overlap="merge",
        merge_overlap=0.15,
        limit=0,
        local_cache=None,
        keep_cache=False,
        progress_every=1000,
        intervals=None,
        classes=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def export(video, output, **overrides):
    predictor = FakePredictor()
    args = make_args(**overrides)
    stats = export_video(
        video,
        output,
        args,
        predictor,
        "fake",
        lambda f: f,
        {"enhance": False},
        write_contours,
    )
    return predictor, stats, args


def test_only_the_tracked_frames_are_run(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    output = tmp_path / "clip.h5"
    predictor, stats, _ = export(video, output, intervals=[[3, 6], [9, 11]])

    assert predictor.frames_seen == [3, 4, 5, 9, 10]
    contours = ExternalContours(output)
    # still one slot per frame of the video, so frame numbers line up
    assert contours.n_frames == N_FRAMES
    for frame in range(N_FRAMES):
        expected = 1 if frame in (3, 4, 5, 9, 10) else 0
        assert len(contours.contours_in_frame(frame)) == expected
    assert sm.exported_intervals_of(contours.attrs) == [[3, 6], [9, 11]]
    contours.close()
    assert stats["processed_frames"] == 5 and stats["frames"] == N_FRAMES


def test_a_whole_export_records_no_intervals(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    output = tmp_path / "clip.h5"
    predictor, _, _ = export(video, output)
    assert predictor.frames_seen == list(range(N_FRAMES))
    with h5py.File(output) as file:
        assert "exported_intervals" not in file.attrs


def test_a_partial_file_is_reused_only_for_frames_it_holds(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    output = tmp_path / "clip.h5"
    _, _, args = export(video, output, intervals=[[2, 8]])
    enhancement = {"enhance": False}

    # stale_reason needs the postprocessing and enhancement records to match
    with h5py.File(output, "a") as file:
        file.attrs["enhancement"] = json.dumps(enhancement)
    args.intervals = [[3, 7]]
    assert stale_reason(output, video, args, enhancement) is None
    args.intervals = [[0, 8]]
    assert "are needed now" in stale_reason(output, video, args, enhancement)
    args.intervals = None
    assert "part of the video" in stale_reason(output, video, args, enhancement)


def write_session(tmp_path, name, videos, **params) -> Path:
    import toml

    path = tmp_path / f"{name}.toml"
    path.write_text(
        toml.dumps({"video_paths": [str(v) for v in videos]} | params),
        encoding="utf-8",
    )
    return path


def test_session_jobs_read_the_table(tmp_path):
    a = make_video(tmp_path / "GX010001.avi")
    b = make_video(tmp_path / "GX020001.avi")
    session = write_session(
        tmp_path,
        "pen1",
        [a, b],
        number_of_animals=9,
        tracking_intervals=[[5, 20]],
        segmentation_model={"weights": "w.pth", "contours_dir": str(tmp_path / "c")},
    )
    jobs = session_jobs([session])
    assert [job["output"].name for job in jobs] == ["GX010001.h5", "GX020001.h5"]
    assert [job["intervals"] for job in jobs] == [[[5, 12]], [[0, 8]]]
    assert {job["max_instances"] for job in jobs} == {9}


def test_session_jobs_name_the_file_at_fault(tmp_path):
    a = make_video(tmp_path / "clip.avi")
    no_table = write_session(tmp_path, "plain", [a], number_of_animals=3)
    with pytest.raises(SystemExit, match="plain.toml: has no \\[segmentation_model\\]"):
        session_jobs([no_table])

    no_count = write_session(
        tmp_path,
        "nocount",
        [a],
        segmentation_model={"weights": "w.pth", "contours_dir": str(tmp_path)},
    )
    with pytest.raises(SystemExit, match="nocount.toml: .*number_of_animals"):
        session_jobs([no_count])


def test_a_clip_shared_by_two_sessions_is_exported_once(tmp_path):
    a = make_video(tmp_path / "clip.avi")
    table = {"weights": "w.pth", "contours_dir": str(tmp_path / "c")}
    s1 = write_session(
        tmp_path,
        "s1",
        [a],
        number_of_animals=2,
        tracking_intervals=[[0, 4]],
        segmentation_model=table,
    )
    write_session(
        tmp_path,
        "s2",
        [a],
        number_of_animals=2,
        tracking_intervals=[[8, 12]],
        segmentation_model=table,
    )
    jobs = session_jobs([tmp_path / "s*.toml"])
    assert len(jobs) == 1
    assert jobs[0]["intervals"] == [[0, 4], [8, 12]]

    s3 = write_session(
        tmp_path, "s3", [a], number_of_animals=5, segmentation_model=table
    )
    with pytest.raises(SystemExit, match="different"):
        session_jobs([s1, s3])


# --------------------------------------------------------- at tracking time


def tracking_session(tmp_path, video, **params) -> Session:
    session = Session()
    session.set_parameters(
        reset=True,
        video_paths=[str(video)],
        number_of_animals=1,
        output_dir=str(tmp_path / "out"),
        **params,
    )
    return session


def test_the_session_finds_its_contours_in_contours_dir(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    contours_dir = tmp_path / "contours"
    contours_dir.mkdir()
    export(video, contours_dir / "clip.h5", intervals=[[2, 10]])

    session = tracking_session(
        tmp_path,
        video,
        tracking_intervals=[[2, 10]],
        segmentation_model={
            "weights": "model_final.pth",
            "contours_dir": str(contours_dir),
        },
    )
    session.prepare_tracking()
    assert Path(session.external_contours).name == "clip.h5"


def test_missing_contours_say_how_to_make_them(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    session = tracking_session(
        tmp_path,
        video,
        segmentation_model={"weights": "w.pth", "contours_dir": str(tmp_path / "none")},
    )
    with pytest.raises(IdtrackeraiError, match="idtrackerai_d2_export --session"):
        session.prepare_tracking()


def test_tracking_outside_the_exported_frames_is_refused(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    output = tmp_path / "clip.h5"
    export(video, output, intervals=[[2, 6]])

    # also without a model table: an empty arena is wrong whoever made the file
    session = tracking_session(
        tmp_path, video, external_contours=str(output), tracking_intervals=[[2, 10]]
    )
    with pytest.raises(IdtrackeraiError, match="holds frames"):
        session.prepare_tracking()


def test_contours_from_another_setting_are_refused(tmp_path):
    video = make_video(tmp_path / "clip.avi")
    output = tmp_path / "clip.h5"
    export(video, output, max_instances=1)

    session = tracking_session(
        tmp_path,
        video,
        external_contours=str(output),
        segmentation_model={"weights": "model_final.pth", "score_threshold": 0.5},
    )
    with pytest.raises(IdtrackeraiError, match="score_threshold is 0.7"):
        session.prepare_tracking()


def test_export_then_track_from_the_same_toml(tmp_path, monkeypatch, capsys):
    """The whole route: one .toml drives the export and the tracking checks."""
    import sys

    from idtrackerai.extra_tools.detectron2_pipeline import inference

    video = make_video(tmp_path / "clip.avi")
    weights = tmp_path / "model_final.pth"
    weights.write_bytes(b"not really a model")
    contours_dir = tmp_path / "contours"
    toml_path = write_session(
        tmp_path,
        "pen1",
        [video],
        number_of_animals=1,
        tracking_intervals=[[3, 9]],
        segmentation_model={
            "weights": str(weights),
            "contours_dir": str(contours_dir),
            "min_component": 0,
            "dilate": 0,
        },
    )
    predictor = FakePredictor()
    monkeypatch.setattr(
        inference, "build_predictor", lambda args, meta: (predictor, "fake")
    )
    monkeypatch.setattr(
        sys, "argv", ["idtrackerai_d2_export", "--session", str(toml_path)]
    )
    inference.main()
    # the default enhancement changes the pixels, so count rather than decode
    assert len(predictor.frames_seen) == 6
    with h5py.File(contours_dir / "clip.h5") as file:
        assert sm.exported_intervals_of(dict(file.attrs)) == [[3, 9]]
    assert "0.0% /   0.0%" in capsys.readouterr().out

    # a second run finds nothing to redo
    predictor.frames_seen.clear()
    inference.main()
    assert predictor.frames_seen == []

    session = tracking_session(
        tmp_path,
        video,
        tracking_intervals=[[3, 9]],
        segmentation_model={
            "weights": str(weights),
            "contours_dir": str(contours_dir),
            "min_component": 0,
            "dilate": 0,
        },
    )
    session.prepare_tracking()  # same weights, same settings: accepted

    weights.write_bytes(b"retrained")
    session = tracking_session(
        tmp_path,
        video,
        segmentation_model={"weights": str(weights), "contours_dir": str(contours_dir)},
    )
    with pytest.raises(IdtrackeraiError, match="other weights"):
        session.prepare_tracking()


def test_sam3_tables():
    spec = sm.normalize_model(
        {"backend": "sam3", "weights": "sam3.pt", "prompt": "chicken"}
    )
    assert sm.effective_max_instances(spec, 9) == 9
    keep_all = sm.normalize_model(spec | {"max_instances": 0})
    assert sm.effective_max_instances(keep_all, 9) == 0

    attrs = {
        "backend": "sam3",
        "weights": "sam3.pt",
        "prompt": "fish",
        "score_threshold": 0.7,
        "max_instances": 9,
        "on_overlap": "merge",
    }
    assert sm.mismatches(attrs, spec, 9) == [
        "prompt is 'fish', the table says 'chicken'"
    ]
    detectron2 = sm.normalize_model({"weights": "sam3.pt"})
    assert any("backend" in p for p in sm.mismatches(attrs, detectron2, 9))
