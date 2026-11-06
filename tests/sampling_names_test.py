"""Naming sampled clips from their folders, so same-named clips stay apart.

``assign_names`` is a pure function of path strings and of what a frames
folder's manifest already records: no video is opened, so most of these tests
need no files at all. They pin the table in the design (``day1/cam1`` and
``day2/cam1`` become ``day1-cam1`` and ``day2-cam1``), the rule that a whole
folder is qualified together so the pieces of one recording stay in one
train/validation group, and the promise that frames that already exist are
never renamed.
"""

import json
import os
import random
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from idtrackerai.extra_tools.detectron2_pipeline.errors import SamplingError
from idtrackerai.extra_tools.detectron2_pipeline.sampling import (
    _folder_chain,
    assign_names,
    describe_renames,
    recorded_clips,
    repeated_stems,
)


def names_for(tmp_path, videos, recorded=()):
    """The names for ``videos`` (relative strings), in the order they were given."""
    paths = [tmp_path / video for video in videos]
    names = assign_names(paths, [(str(tmp_path / p), n) for p, n in recorded])
    return [names[path] for path in paths]


CASES = {
    "two days": (
        ["day1/cam1.mp4", "day2/cam1.mp4", "day1/cam2.mp4"],
        [],
        ["day1-cam1", "day2-cam1", "day1-cam2"],
    ),
    "same leaf folder": (
        ["a/x/cam1.mp4", "b/x/cam1.mp4", "c/y/cam1.mp4"],
        [],
        ["a-x-cam1", "b-x-cam1", "c-y-cam1"],
    ),
    "pieces of two recordings": (
        ["rec_A/part_1.mp4", "rec_A/part_2.mp4", "rec_B/part_1.mp4"],
        [],
        ["rec_A-part_1", "rec_A-part_2", "rec_B-part_1"],
    ),
    "no collision": (
        ["rec_A/part_1.mp4", "rec_A/part_2.mp4", "rec_C/other.mp4"],
        [],
        ["part_1", "part_2", "other"],
    ),
    "same stem, one folder": (["a/cam1.mp4", "a/cam1.avi"], [], ["cam1-2", "cam1"]),
    "case only": (["a/Cam1.mp4", "b/cam1.mp4"], [], ["a-Cam1", "b-cam1"]),
    "shared tail": (
        ["mnt/x/exp/cam1.mp4", "media/x/exp/cam1.mp4"],
        [],
        ["mnt-x-exp-cam1", "media-x-exp-cam1"],
    ),
    "literal name equals a qualified one": (
        ["day1/cam1.mp4", "day2/cam1.mp4", "other/day1-cam1.mp4"],
        [],
        ["day1-cam1", "day2-cam1", "day1-cam1-2"],
    ),
    "beside a plain recorded clip": (
        ["day2/cam1.mp4"],
        [("day1/cam1.mp4", "cam1")],
        ["day2-cam1"],
    ),
    "sibling of qualified recorded clips": (
        ["day3/cam1.mp4", "day1/cam2.mp4"],
        [("day1/cam1.mp4", "day1-cam1"), ("day2/cam1.mp4", "day2-cam1")],
        ["day3-cam1", "day1-cam2"],
    ),
}


@pytest.mark.parametrize(
    "videos, recorded, expected", CASES.values(), ids=list(CASES.keys())
)
def test_names_for_each_layout(tmp_path, videos, recorded, expected):
    assert names_for(tmp_path, videos, recorded) == expected


def test_names_do_not_depend_on_input_order(tmp_path):
    videos = [tmp_path / v for v in CASES["two days"][0]]
    shuffled = list(videos)
    random.Random(0).shuffle(shuffled)
    assert shuffled != videos
    assert assign_names(shuffled) == assign_names(videos)


def test_a_recorded_clip_keeps_its_name(tmp_path):
    names = names_for(
        tmp_path, ["day1/cam1.mp4", "day2/cam1.mp4"], [("day1/cam1.mp4", "weird")]
    )
    assert names == ["weird", "day2-cam1"]


def test_a_new_prefix_keeps_clear_of_one_already_recorded(tmp_path):
    """x/day1 must not borrow the prefix day1 was recorded with, even if no name
    clashes yet; one depth serves every folder, so z/y is lengthened too."""
    names = names_for(
        tmp_path, ["x/day1/cam1.mp4", "z/y/cam1.mp4"], [("day1/cam9.mp4", "day1-cam9")]
    )
    assert names == ["x-day1-cam1", "z-y-cam1"]


def test_a_prefix_is_deepened_rather_than_make_a_name_already_taken(tmp_path):
    names = names_for(
        tmp_path,
        ["a/day1/cam1.mp4", "b/day2/cam1.mp4"],
        [("lit/day2-cam1.mp4", "day2-cam1")],
    )
    assert names == ["a-day1-cam1", "b-day2-cam1"]


def test_a_file_name_equal_to_a_recorded_name_is_qualified(tmp_path):
    names = names_for(
        tmp_path, ["other/day1-cam1.mp4"], [("day1/cam1.mp4", "day1-cam1")]
    )
    assert names == ["other-day1-cam1"]


def test_a_clip_at_the_top_of_a_drive_has_no_prefix_to_add():
    root = Path(os.path.abspath(os.sep))
    clips = [root / "cam1.mp4", root / "day1" / "cam1.mp4"]
    assert [assign_names(clips)[c] for c in clips] == ["cam1", "day1-cam1"]


def test_illegal_characters_in_folders_become_underscores(tmp_path):
    names = names_for(tmp_path, ["a:b/cam1.mp4", "c?d/cam1.mp4"])
    assert names == ["a_b-cam1", "c_d-cam1"]
    # a stem is never altered, so a lone clip keeps the name it always had
    assert names_for(tmp_path, ["x:y.mp4"]) == ["x:y"]


def test_drive_letters_are_not_part_of_the_chain():
    assert _folder_chain(PureWindowsPath("D:/exp/day1/cam1.mp4")) == ["exp", "day1"]
    assert _folder_chain(PureWindowsPath("//server/share/exp/cam1.mp4")) == ["exp"]
    assert _folder_chain(PurePosixPath("/d/day1/cam1.mp4")) == ["d", "day1"]
    assert _folder_chain(PurePosixPath("/cam1.mp4")) == []


def test_one_file_spelled_two_ways_is_one_clip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "day1" / "cam1.mp4"
    relative = Path("day1/cam1.mp4")

    alone = assign_names([absolute, relative])
    assert alone[absolute] == alone[relative] == "cam1"

    other = tmp_path / "day2" / "cam1.mp4"
    beside = assign_names([absolute, other, relative])
    assert beside[absolute] == beside[relative] == "day1-cam1"
    assert beside[other] == "day2-cam1"


def test_one_file_reached_through_a_symlink_is_one_clip(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    try:
        os.symlink(real, tmp_path / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    direct, through = real / "cam1.mp4", tmp_path / "link" / "cam1.mp4"
    names = assign_names([through, direct])
    assert names[direct] == names[through] == "cam1"


def test_twenty_thousand_clips_are_named_quickly():
    videos = [Path(f"/data/exp{i % 400}/cam{i // 400}.mp4") for i in range(20_000)]
    started = time.perf_counter()
    names = assign_names(videos)
    elapsed = time.perf_counter() - started
    assert elapsed < 5
    assert len(names) == 20_000
    assert len({name.casefold() for name in names.values()}) == 20_000
    assert names[Path("/data/exp0/cam0.mp4")] == "exp0-cam0"


# ---------------------------------------------------------- recorded_clips
def write_manifest(out: Path, records):
    """Touches each frame file the records name, then writes the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("file_name"), str):
            (out / record["file_name"]).write_bytes(b"")
    (out / "sampling_manifest.json").write_text(
        json.dumps({"frames": records}), encoding="utf-8"
    )


def frame(path, name, index=0, **extra):
    record = {
        "file_name": f"{name}_f{index:06d}.png",
        "source_video": name,
        "source_path": str(path),
    }
    record.update(extra)
    return record


def test_no_manifest_records_nothing(tmp_path):
    assert recorded_clips(tmp_path) == []
    assert recorded_clips(tmp_path / "not-created-yet") == []


def test_an_unreadable_manifest_records_nothing(tmp_path):
    (tmp_path / "sampling_manifest.json").write_text("{not json", encoding="utf-8")
    assert recorded_clips(tmp_path) == []


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"frames": 5},
        {"frames": {"a": 1}},
        {
            "frames": [
                1,
                "x",
                None,
                {"file_name": 3, "source_path": [], "source_video": {}},
                {"file_name": "a.png", "source_path": "/d/a.mp4", "source_video": 7},
            ]
        },
    ],
    ids=["a list", "frames not a list", "frames a dict", "records of the wrong type"],
)
def test_a_manifest_of_the_wrong_shape_records_nothing(tmp_path, manifest):
    (tmp_path / "sampling_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert recorded_clips(tmp_path) == []


def test_records_without_a_source_path_are_ignored(tmp_path):
    old = {"file_name": "old_f000001.png", "source_video": "old"}
    write_manifest(tmp_path, [old, frame("/d/new.mp4", "new")])
    assert recorded_clips(tmp_path) == [("/d/new.mp4", "new")]


def test_a_source_video_with_an_extension_is_normalised(tmp_path):
    write_manifest(tmp_path, [frame("/d/cam1.mp4", "cam1.mp4")])
    assert recorded_clips(tmp_path) == [("/d/cam1.mp4", "cam1")]


def test_a_clip_whose_frames_were_all_deleted_holds_nothing(tmp_path):
    write_manifest(tmp_path, [frame("/d/cam1.mp4", "cam1", i) for i in range(3)])
    for png in tmp_path.glob("*.png"):
        png.unlink()
    assert recorded_clips(tmp_path) == []


def test_a_clip_with_one_frame_left_still_holds_its_name(tmp_path):
    write_manifest(tmp_path, [frame("/d/cam1.mp4", "cam1", i) for i in range(3)])
    (tmp_path / "cam1_f000000.png").unlink()
    (tmp_path / "cam1_f000001.png").unlink()
    assert recorded_clips(tmp_path) == [("/d/cam1.mp4", "cam1")]


def test_each_clip_is_listed_once_however_many_frames_it_has(tmp_path):
    records = [frame("/d/a.mp4", "a", i) for i in range(4)]
    records += [frame("/d/b.mp4", "b", i) for i in range(2)]
    write_manifest(tmp_path, records)
    assert recorded_clips(tmp_path) == [("/d/a.mp4", "a"), ("/d/b.mp4", "b")]


def test_paths_come_back_as_recorded(tmp_path):
    spelled = "/data/./exp/../exp/day1/cam1.mp4"
    write_manifest(tmp_path, [frame(spelled, "day1-cam1")])
    assert recorded_clips(tmp_path) == [(spelled, "day1-cam1")]


# -------------------------------------------------------- describe_renames
def test_a_contested_clip_names_the_one_it_shares_with(tmp_path):
    day1, day2 = tmp_path / "day1" / "cam1.mp4", tmp_path / "day2" / "cam1.mp4"
    names = assign_names([day1, day2])
    assert describe_renames([day1, day2], names) == [
        f"{day1} -> day1-cam1  (shares a file name with {day2})",
        f"{day2} -> day2-cam1  (shares a file name with {day1})",
    ]


def test_a_folder_mate_says_why(tmp_path):
    cam1, other = tmp_path / "day1" / "cam1.mp4", tmp_path / "day2" / "cam1.mp4"
    cam2 = tmp_path / "day1" / "cam2.mp4"
    videos = [cam1, other, cam2]
    lines = describe_renames(videos, assign_names(videos))
    assert lines[-1] == (
        f"{cam2} -> day1-cam2  (same folder as {cam1}, which shares a file name)"
    )


def test_a_recorded_clip_can_be_the_one_shared_with(tmp_path):
    earlier, new = tmp_path / "day1" / "cam1.mp4", tmp_path / "day2" / "cam1.mp4"
    recorded = [(str(earlier), "cam1")]
    names = assign_names([new], recorded)
    assert describe_renames([new], names, recorded) == [
        f"{new} -> day2-cam1  (shares a file name with {earlier})"
    ]


def test_a_numbered_clip_names_its_twin_in_the_folder(tmp_path):
    mp4, avi = tmp_path / "a" / "cam1.mp4", tmp_path / "a" / "cam1.avi"
    lines = describe_renames([mp4, avi], assign_names([mp4, avi]))
    assert lines == [f"{mp4} -> cam1-2  (shares a file name with {avi})"]


def test_the_clip_in_another_folder_is_named_before_a_twin_in_the_same_one(tmp_path):
    avi, mp4 = tmp_path / "day1" / "cam1.avi", tmp_path / "day1" / "cam1.mp4"
    elsewhere = tmp_path / "day2" / "cam1.mp4"
    videos = [avi, mp4, elsewhere]
    lines = describe_renames(videos, assign_names(videos))
    assert lines[0] == f"{avi} -> day1-cam1  (shares a file name with {elsewhere})"


def test_one_file_listed_twice_is_described_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute, relative = tmp_path / "day1" / "cam1.mp4", Path("day1/cam1.mp4")
    other = tmp_path / "day2" / "cam1.mp4"
    videos = [absolute, relative, other]
    lines = describe_renames(videos, assign_names(videos))
    assert [line.split(" -> ")[1].split("  ")[0] for line in lines] == [
        "day1-cam1",
        "day2-cam1",
    ]


def test_nothing_renamed_gives_no_lines(tmp_path):
    videos = [tmp_path / v for v in CASES["no collision"][0]]
    assert describe_renames(videos, assign_names(videos)) == []


# --------------------------------------------------------------- the safety net
def test_two_recorded_clips_that_already_share_a_name_are_reported(tmp_path):
    """A manifest from before names were unique can hold the same name twice."""
    videos = [tmp_path / "a" / "cam1.mp4", tmp_path / "b" / "cam1.mp4"]
    recorded = [(str(video), "cam1") for video in videos]
    with pytest.raises(SamplingError, match="cam1"):
        assign_names(videos, recorded)


# ------------------------------------------------------- contour files by stem
def test_repeated_stems_finds_only_shared_stems():
    a1, b1, a2 = Path("a/cam1.mp4"), Path("b/cam1.mp4"), Path("a/cam2.mp4")
    assert repeated_stems([a1, b1, a2]) == {"cam1": [a1, b1]}
    assert repeated_stems([a1, a2]) == {}
