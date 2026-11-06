"""Choosing annotation frames from folders of videos, subfolders included.

The sampler already spreads a total of N frames across however many videos it is
given. What these tests pin down is how a *folder* becomes that list of videos:
top level only by default, the whole tree on request, with macOS companion
files and hidden folders left out, in an order that does not change between
runs (the selection number is only reproducible if the order is).

They also cover what happens when two different clips share a file name, in
one run or across separate runs: each is named from its folders, and frames
that already exist are left alone. Real, tiny videos are used where frames are
actually read.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from idtrackerai.extra_tools.detectron2_pipeline import cli
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline.errors import SamplingError
from idtrackerai.extra_tools.detectron2_pipeline.preprocessing import DEFAULT_SETTINGS

TOOLTIPS = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/segmentation_app/tooltips.toml"
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def make_video(path: Path, frames: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(frames):
        writer.write(np.full((48, 64, 3), 30 + i, np.uint8))
    writer.release()
    return path


@pytest.fixture
def tree(tmp_path):
    """A small recording folder, with everything that should be ignored in it."""
    root = tmp_path / "experiment"
    return {
        "root": root,
        "top": [touch(root / "a.mp4"), touch(root / "B.MP4")],
        "nested": [
            touch(root / "day1" / "c.avi"),
            touch(root / "day2" / "tank" / "d.mov"),
        ],
        "ignored": [
            touch(root / "notes.txt"),
            touch(root / "._a.mp4"),  # macOS companion
            touch(root / ".hidden" / "e.mp4"),  # hidden folder
            touch(root / "day1" / ".DS_Store"),
            touch(root / "day1" / "._c.avi"),
        ],
    }


# ------------------------------------------------------------ finding videos
def test_a_folder_gives_its_own_videos_and_not_its_subfolders_by_default(tree):
    found = sampling_mod.resolve_videos([tree["root"]])
    assert sorted(found) == sorted(tree["top"])


def test_recursive_walks_the_whole_tree(tree):
    found = sampling_mod.resolve_videos([tree["root"]], recursive=True)
    assert sorted(found) == sorted(tree["top"] + tree["nested"])


def test_extensions_match_whatever_the_case(tree):
    names = {p.name for p in sampling_mod.resolve_videos([tree["root"]])}
    assert "B.MP4" in names


def test_companion_files_hidden_folders_and_non_videos_are_skipped(tree):
    found = set(sampling_mod.resolve_videos([tree["root"]], recursive=True))
    assert not found & set(tree["ignored"])


def test_the_order_does_not_depend_on_the_disk(tree):
    once = sampling_mod.resolve_videos([tree["root"]], recursive=True)
    again = sampling_mod.resolve_videos([tree["root"]], recursive=True)
    assert once == again == sorted(once)


def test_files_and_globs_behave_as_they_did(tree):
    a = tree["top"][0]
    found = sampling_mod.resolve_videos([a, tree["root"] / "day1" / "*.avi"])
    assert found == [a, tree["root"] / "day1" / "c.avi"]


def test_a_folder_plus_a_file_inside_it_does_not_list_the_file_twice(tree):
    found = sampling_mod.resolve_videos([tree["top"][0], tree["root"]])
    assert len(found) == len(set(found)) == len(tree["top"])


def test_a_path_that_is_not_there_is_dropped_not_fatal(tree, tmp_path):
    found = sampling_mod.resolve_videos([tmp_path / "nowhere", tree["root"]])
    assert sorted(found) == sorted(tree["top"])


def test_recursive_does_not_widen_a_plain_file(tree):
    a = tree["top"][0]
    assert sampling_mod.resolve_videos([a], recursive=True) == [a]


# --------------------------------------------- N in total, across the folders
def test_the_total_is_shared_across_videos_from_different_subfolders(tree):
    videos = sampling_mod.resolve_videos([tree["root"]], recursive=True)
    plans = sampling_mod.plan_sampling(videos, 30, counts=[100] * len(videos))
    assert sum(p.requested for p in plans) == 30
    assert all(p.requested > 0 for p in plans)


def test_the_same_selection_number_picks_the_same_frames(tree):
    videos = sampling_mod.resolve_videos([tree["root"]], recursive=True)
    counts = [200] * len(videos)
    one = sampling_mod.plan_sampling(videos, 20, seed=7, counts=counts)
    two = sampling_mod.plan_sampling(videos, 20, seed=7, counts=counts)
    assert [p.indices for p in one] == [p.indices for p in two]


# ------------------------------------------- same name, separate sampling runs
def sample(videos, out, n=5, seed=1):
    return sampling_mod.sample_frames(videos, out, n, dict(DEFAULT_SETTINGS), seed=seed)


def source_names(out: Path) -> set[str]:
    recorded = json.loads((out / "sampling_manifest.json").read_text("utf-8"))
    return {f["source_video"] for f in recorded["frames"]}


def test_same_named_clips_in_one_run_get_folder_names(tmp_path):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"

    result = sample([day1, day2], out)

    assert source_names(out) == {"day1-cam1", "day2-cam1"}
    assert list(out.glob("day1-cam1_f*.png")) and list(out.glob("day2-cam1_f*.png"))
    assert set(result.per_video) == {"day1-cam1", "day2-cam1"}
    assert sum(result.per_video.values()) == 5


def test_a_later_run_names_the_newcomer_and_leaves_the_earlier_frames_alone(tmp_path):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    sample([day1], out)
    before = {p.name: p.read_bytes() for p in out.glob("cam1_f*.png")}
    assert before

    sample([day2], out)

    assert source_names(out) == {"cam1", "day2-cam1"}
    assert {p.name: p.read_bytes() for p in out.glob("cam1_f*.png")} == before
    assert list(out.glob("day2-cam1_f*.png"))


def test_two_clips_already_recorded_under_one_name_stop_the_run_before_it_writes(
    tmp_path,
):
    """An older run of same-named clips can leave one name for two source paths."""
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    records = [
        {
            "file_name": "cam1_f000001.png",
            "source_video": "cam1",
            "source_path": str(day1),
        },
        {
            "file_name": "cam1_f000002.png",
            "source_video": "cam1",
            "source_path": str(day2),
        },
    ]
    for record in records:
        touch(out / record["file_name"])
    manifest = out / "sampling_manifest.json"
    manifest.write_text(json.dumps({"frames": records}), encoding="utf-8")
    files_before = sorted(p.name for p in out.iterdir())
    manifest_before = manifest.read_text(encoding="utf-8")

    with pytest.raises(SamplingError) as caught:
        sample([day1, day2], out)

    message = str(caught.value)
    assert str(day1) in message and str(day2) in message and "cam1" in message
    assert "new frames folder" in message and "delete the frames" in message
    # stopped before anything was written: no new image, profile or manifest change
    assert sorted(p.name for p in out.iterdir()) == files_before
    assert manifest.read_text(encoding="utf-8") == manifest_before


def test_sampling_the_same_clip_again_is_still_fine(tmp_path):
    clip = make_video(tmp_path / "day1" / "cam1.avi")
    out = tmp_path / "frames"
    sample([clip], out, seed=1)
    sample([clip], out, seed=2)  # a corpus grows by sampling again
    recorded = json.loads((out / "sampling_manifest.json").read_text("utf-8"))
    assert {f["source_path"] for f in recorded["frames"]} == {str(clip)}


def test_the_same_clip_reached_by_a_relative_path_is_not_a_clash(tmp_path, monkeypatch):
    clip = make_video(tmp_path / "day1" / "cam1.avi")
    out = tmp_path / "frames"
    sample([clip], out, seed=1)
    monkeypatch.chdir(tmp_path)
    sample([Path("day1") / "cam1.avi"], out, seed=2)  # no error


def test_one_file_listed_twice_is_sampled_once(tmp_path, monkeypatch):
    clip = make_video(tmp_path / "day1" / "cam1.avi")
    out = tmp_path / "frames"
    monkeypatch.chdir(tmp_path)

    result = sample([clip, Path("day1") / "cam1.avi"], out)

    assert source_names(out) == {"cam1"}
    assert len(list(out.glob("*.png"))) == 5
    assert result.per_video == {"cam1": 5}


def test_one_file_given_two_ways_is_resolved_once(tmp_path, monkeypatch):
    clip = make_video(tmp_path / "day1" / "cam1.avi")
    monkeypatch.chdir(tmp_path)
    assert sampling_mod.resolve_videos([clip, Path("day1") / "cam1.avi"]) == [clip]


def test_plan_sampling_carries_the_names(tmp_path):
    first, second = tmp_path / "a" / "cam1.avi", tmp_path / "b" / "cam1.avi"
    chosen = {first: "left-cam1", second: "right-cam1"}

    plans = sampling_mod.plan_sampling(
        [first, second], 10, counts=[100, 100], names=chosen
    )
    assert [p.name for p in plans] == ["left-cam1", "right-cam1"]

    # without names it assigns them, as for a folder with nothing recorded
    plans = sampling_mod.plan_sampling([first, second], 10, counts=[100, 100])
    assert [p.name for p in plans] == ["a-cam1", "b-cam1"]

    assert sampling_mod.VideoPlan(first, 10, 5).name == "cam1"


def test_different_names_share_a_folder_without_trouble(tmp_path):
    one = make_video(tmp_path / "day1" / "cam1.avi")
    two = make_video(tmp_path / "day2" / "cam2.avi")
    out = tmp_path / "frames"
    sample([one], out)
    sample([two], out)
    recorded = json.loads((out / "sampling_manifest.json").read_text("utf-8"))
    assert {f["source_video"] for f in recorded["frames"]} == {"cam1", "cam2"}


def test_frames_that_were_deleted_no_longer_hold_the_name(tmp_path):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    sample([day1], out)
    for image in out.glob("*.png"):
        image.unlink()  # the user threw that batch away
    sample([day2], out)  # nothing left to overwrite, so the name is free
    assert source_names(out) == {"cam1"}


# ------------------------------------------------------------------------ CLI
def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["idtrackerai_d2_sample", *map(str, args)])
    cli.sample_main()


def sources_in(out: Path) -> set[str]:
    recorded = json.loads((out / "sampling_manifest.json").read_text("utf-8"))
    return {f["source_path"] for f in recorded["frames"]}


@pytest.fixture
def real_tree(tmp_path):
    root = tmp_path / "experiment"
    return {
        "root": root,
        "top": make_video(root / "a.avi"),
        "day1": make_video(root / "day1" / "b.avi"),
        "day2": make_video(root / "day2" / "tank" / "c.avi"),
    }


def test_the_cli_takes_a_folder_and_reads_only_its_top_level(
    real_tree, tmp_path, monkeypatch
):
    out = tmp_path / "frames"
    run_cli(
        monkeypatch, "--videos", real_tree["root"], "--n-frames", 6, "--output", out
    )
    assert sources_in(out) == {str(real_tree["top"])}


def test_the_cli_recursive_flag_reaches_every_subfolder(
    real_tree, tmp_path, monkeypatch
):
    out = tmp_path / "frames"
    run_cli(
        monkeypatch,
        "--videos",
        real_tree["root"],
        "--recursive",
        "--n-frames",
        9,
        "--output",
        out,
    )
    assert sources_in(out) == {
        str(real_tree["top"]),
        str(real_tree["day1"]),
        str(real_tree["day2"]),
    }
    assert len(list(out.glob("*.png"))) == 9  # the total, not 9 per video


def test_the_cli_says_so_when_a_folder_holds_no_videos(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as caught:
        run_cli(
            monkeypatch, "--videos", empty, "--n-frames", 5, "--output", tmp_path / "o"
        )
    assert "No videos" in str(caught.value)


def test_the_cli_names_clips_that_share_a_file_name(tmp_path, monkeypatch, capsys):
    day1 = make_video(tmp_path / "other" / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "other" / "day2" / "cam1.avi")
    out = tmp_path / "frames"

    run_cli(
        monkeypatch,
        "--videos",
        tmp_path / "other",
        "--recursive",
        "--n-frames",
        6,
        "--output",
        out,
    )

    shown = capsys.readouterr().out
    assert "named by folder" in shown
    assert "day1-cam1" in shown and "day2-cam1" in shown
    assert sources_in(out) == {str(day1), str(day2)}
    assert list(out.glob("day1-cam1_f*.png")) and list(out.glob("day2-cam1_f*.png"))


# ------------------------------------------------------------- the app panel
@pytest.fixture
def d2_panel(real_tree):
    pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from idtrackerai.segmentation_app.widgets import Detectron2Panel, EnhancementWidget

    panel = Detectron2Panel(EnhancementWidget())
    panel.gpu_thread.wait(30000)
    yield panel
    panel.count_thread.wait(30000)
    panel.close()
    del app


def choose_folder(monkeypatch, folder: Path):
    from qtpy.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )


def test_add_folder_stays_top_level_unless_subfolders_are_asked_for(
    d2_panel, real_tree, monkeypatch
):
    choose_folder(monkeypatch, real_tree["root"])
    d2_panel.set_video_paths([])
    assert not d2_panel.include_subfolders.isChecked()
    d2_panel.add_video_folder()
    assert d2_panel.video_paths == [real_tree["top"]]


def test_add_folder_with_subfolders_takes_the_whole_tree(
    d2_panel, real_tree, monkeypatch
):
    choose_folder(monkeypatch, real_tree["root"])
    d2_panel.set_video_paths([])
    d2_panel.include_subfolders.setChecked(True)
    d2_panel.add_video_folder()
    assert sorted(d2_panel.video_paths) == sorted(
        [real_tree["top"], real_tree["day1"], real_tree["day2"]]
    )


def test_the_subfolder_choice_is_explained(d2_panel):
    """setToolTips skips missing keys silently, so check the key is really there."""
    import toml

    tips = toml.load(TOOLTIPS)
    assert "d2_include_subfolders" in tips
    d2_panel.setToolTips(tips)
    assert d2_panel.include_subfolders.toolTip()
    assert "subfolder" in tips["d2_add_folder"].lower()
    assert "same file name" not in tips["d2_include_subfolders"]


def listed(panel) -> list[str]:
    return [panel.video_list.item(i).text() for i in range(panel.video_list.count())]


def counted(panel) -> None:
    """Waits for the frame counts and lets the panel hear about them."""
    from qtpy.QtWidgets import QApplication

    panel.count_thread.wait(30000)
    QApplication.processEvents()


def test_clips_sharing_a_file_name_are_listed_by_their_folder_name(d2_panel, tmp_path):
    d2_panel.set_video_paths(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            make_video(tmp_path / "day2" / "cam1.avi"),
            make_video(tmp_path / "other" / "cam2.avi"),
        ]
    )
    assert listed(d2_panel) == [
        "day1-cam1  (cam1.avi)",
        "day2-cam1  (cam1.avi)",
        "cam2.avi",
    ]


def test_the_summary_counts_recordings_by_name(d2_panel, tmp_path):
    d2_panel.set_video_paths(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            make_video(tmp_path / "day2" / "cam1.avi"),
        ]
    )
    assert "2 recordings" in d2_panel.video_summary.text()


def test_the_preview_says_how_many_are_named_by_folder(d2_panel, tmp_path):
    d2_panel.set_video_paths(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            make_video(tmp_path / "day2" / "cam1.avi"),
        ]
    )
    counted(d2_panel)
    assert "2 clips will be named by folder" in d2_panel.sample_preview.text()


def test_the_output_folder_decides_what_is_already_taken(d2_panel, tmp_path):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    sample([day1], out)  # recorded as plain cam1, before day2 was in the picture

    d2_panel.set_video_paths([day1, day2])
    assert listed(d2_panel) == ["day1-cam1  (cam1.avi)", "day2-cam1  (cam1.avi)"]

    d2_panel.frames_dir.setText(str(out))

    assert listed(d2_panel) == ["cam1.avi", "day2-cam1  (cam1.avi)"]
    assert d2_panel.clip_names() == {day1: "cam1", day2: "day2-cam1"}

    d2_panel.frames_dir.setText("")  # nothing is on record without a folder

    assert listed(d2_panel) == ["day1-cam1  (cam1.avi)", "day2-cam1  (cam1.avi)"]


def test_a_list_edited_during_a_run_is_named_from_what_the_run_recorded(
    d2_panel, tmp_path
):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    d2_panel.frames_dir.setText(str(out))
    d2_panel.set_video_paths([day1])  # what the run starts with
    d2_panel.set_video_paths([day1, day2])  # added while it works
    assert listed(d2_panel) == ["day1-cam1  (cam1.avi)", "day2-cam1  (cam1.avi)"]

    # the run only knew day1, so it recorded plain cam1
    d2_panel.sampling_thread.result = sample([day1], out)
    d2_panel._sampling_finished()

    assert listed(d2_panel) == ["cam1.avi", "day2-cam1  (cam1.avi)"]
    assert d2_panel.clip_names() == {day1: "cam1", day2: "day2-cam1"}


def test_a_frames_folder_that_repeats_a_name_does_not_break_the_list(
    d2_panel, tmp_path
):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    out = tmp_path / "frames"
    records = []
    for index, clip in enumerate((day1, day2), start=1):
        file_name = f"cam1_f{index:06d}.png"
        touch(out / file_name)
        records.append(
            {"file_name": file_name, "source_video": "cam1", "source_path": str(clip)}
        )
    (out / "sampling_manifest.json").write_text(
        json.dumps({"frames": records}), encoding="utf-8"
    )
    d2_panel.set_video_paths([day1, day2])

    d2_panel.frames_dir.setText(str(out))  # a run would refuse; the list must not

    assert listed(d2_panel) == ["day1-cam1  (cam1.avi)", "day2-cam1  (cam1.avi)"]


def test_a_clip_that_cannot_be_found_is_marked_under_its_name(d2_panel, tmp_path):
    d2_panel.set_video_paths(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            tmp_path / "day2" / "cam1.avi",  # not there
        ]
    )
    assert listed(d2_panel) == [
        "day1-cam1  (cam1.avi)",
        "day2-cam1  (cam1.avi)  (missing)",
    ]


def test_without_an_output_folder_no_manifest_is_read(d2_panel, tmp_path, monkeypatch):
    day1 = make_video(tmp_path / "day1" / "cam1.avi")
    day2 = make_video(tmp_path / "day2" / "cam1.avi")
    elsewhere = tmp_path / "somewhere_else"
    touch(elsewhere / "cam1_f000001.png")
    records = [
        {
            "file_name": "cam1_f000001.png",
            "source_video": "cam1",
            "source_path": str(day1),
        }
    ]
    (elsewhere / "sampling_manifest.json").write_text(
        json.dumps({"frames": records}), encoding="utf-8"
    )
    monkeypatch.chdir(elsewhere)
    assert d2_panel.frames_dir.text() == ""

    d2_panel.set_video_paths([day2])

    assert listed(d2_panel) == ["cam1.avi"]
    assert d2_panel.clip_names() == {day2: "cam1"}


def test_the_grouping_editor_shows_the_names(d2_panel, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QDialog, QTableWidget

    d2_panel.set_video_paths(
        [
            make_video(tmp_path / "day1" / "cam1.avi"),
            make_video(tmp_path / "day2" / "cam1.avi"),
        ]
    )
    shown: list[str] = []

    def look(dialog):
        table = dialog.findChild(QTableWidget)
        shown.extend(table.item(row, 0).text() for row in range(table.rowCount()))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", look)
    d2_panel.edit_grouping()

    assert shown == ["day1-cam1", "day2-cam1"]
