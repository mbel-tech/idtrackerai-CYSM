# Path-based Clip Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let annotation frames be sampled from clips that share a file name by naming them from their folders (`day1-cam1`, `day2-cam1`), instead of refusing the run.

**Architecture:** A pure function `assign_names` in `sampling.py` turns the clip list plus what the output folder's manifest already records into one unique name per clip. Sampling, the CLI, the app panel and the Colab notebook's split preview all read those names; `dataset.py` already groups on the manifest's `source_video`, so the split logic does not change.

**Tech Stack:** Python 3.11/3.13, OpenCV, PyQt6 via qtpy, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-11-06-path-based-clip-names-design.md`

## Global Constraints

- Separator is a hyphen `-`; leftover duplicates get `-2`, `-3`, ... in sorted-location order, the first keeping the bare name.
- A folder is a clip's immediate parent directory. Qualification is per folder: every new clip in a qualified folder gets the folder's prefix, contested or not.
- Prefix = last `d` folder names joined by `-`, `d` the smallest depth that makes qualified folders' prefixes distinct from each other and from recorded prefixes and makes no name taken. One `d` for all. No cap on depth.
- Frames that already exist are never renamed; the manifest decides. A folder with recorded qualified clips reuses the prefix it was recorded with (recorded name minus `-<stem>`).
- All name comparisons are case-folded; display keeps the original case. A stem is never altered; folder parts have `<>:"/\|?*` and control characters replaced by `_`.
- Path parts come from `os.path.abspath` with the drive/root dropped (`PurePath.parts[1:]` of the parent), not from symlink targets. Locations compare via `os.path.realpath` (`_location`, already in `sampling.py`).
- Frame file name stays `<name>_f<index:06d>.<fmt>`; manifest format is unchanged; `source_path` stays absolute.
- `sampling.py` and `dataset.py` are also loaded as flat files on Colab: add no package-relative imports to either, and nothing outside the standard library to `dataset.py`.
- Tests run with `QT_QPA_PLATFORM=offscreen python -m pytest <paths> -q -p no:cacheprovider -o log_file=/dev/null` (the `log_file` override keeps `pytest.log` out of the tree). Full suite: `tests -q --ignore=tests/smoke_test.py`.
- `sampling.py` and new test files are black-clean (`black --check`, `isort --check --profile black`); `cli.py` already is not, so do not reformat it.

## Review Focus

- The same file listed twice under different spellings (relative and absolute): one clip, not `cam1` and `cam1-2` (Task 1 and Task 2).
- A whole drive: 20,000 clips must be named in seconds, not minutes (Task 1).
- Manifests from before source paths were recorded, or whose `source_video` carries an extension: they must neither object nor crash (Task 1).
- Windows drive letters and UNC anchors are not part of a prefix; illegal characters in folder names are replaced (Task 1).
- No output folder chosen yet: the panel must not read a `sampling_manifest.json` from the working directory (Task 4).

## File Structure

- `src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py`: all naming logic (Tasks 1, 2, 5).
- `.../cli.py`: prints renamed clips (Task 2).
- `.../dataset.py`: `sources_in_folder` (Task 3).
- `.../notebooks/colab_detectron2_pipeline.ipynb`: cell 13 only (Task 3).
- `src/idtrackerai/segmentation_app/widgets/detectron2_panel.py`, `sam3_panel.py`, `.../segmentation_app/tooltips.toml` (Tasks 4, 5).
- Tests: new `tests/sampling_names_test.py` (Tasks 1, 5) and `tests/clip_names_dataset_test.py` (Task 3); existing `tests/sampling_folders_test.py` (Tasks 2, 4) and `tests/sam3_panel_test.py` (Task 5).
- Docs: `docs/detectron2-pipeline.md`, `USER_GUIDE.md`, `docs/source/user_guide/changelog.rst` (Task 6).

---

### Task 1: Name assignment as pure functions

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py` (add after `check_against_folder`)
- Create: `tests/sampling_names_test.py`

**Interfaces:**
- Produces:
  - `recorded_clips(output: Path) -> list[tuple[str, str]]`: `(source_path, name)` for each clip with at least one frame still in `output`.
  - `assign_names(videos: Sequence[Path], recorded: Sequence[tuple[str, str]] = ()) -> dict[Path, str]`: keyed by the given `Path` objects.
  - `describe_renames(videos: Sequence[Path], names: Mapping[Path, str], recorded: Sequence[tuple[str, str]] = ()) -> list[str]`: one line per clip whose name differs from `video.stem`; `[]` if none.
  - `_folder_chain(path: PurePath) -> list[str]`: parent folder names, drive/root excluded.

- [ ] **Step 1: Write the failing tests** in `tests/sampling_names_test.py`. A `names_for(tmp_path, videos, recorded=())` helper joins relative strings onto `tmp_path` and returns the names in input order. Tests:
  - `test_names_for_each_layout` parametrized over this table (relative paths; expected names in input order):

```python
CASES = {
    "two days": (["day1/cam1.mp4", "day2/cam1.mp4", "day1/cam2.mp4"], [],
                 ["day1-cam1", "day2-cam1", "day1-cam2"]),
    "same leaf folder": (["a/x/cam1.mp4", "b/x/cam1.mp4", "c/y/cam1.mp4"], [],
                         ["a-x-cam1", "b-x-cam1", "c-y-cam1"]),
    "pieces of two recordings": (["rec_A/part_1.mp4", "rec_A/part_2.mp4", "rec_B/part_1.mp4"], [],
                                 ["rec_A-part_1", "rec_A-part_2", "rec_B-part_1"]),
    "no collision": (["rec_A/part_1.mp4", "rec_A/part_2.mp4", "rec_C/other.mp4"], [],
                     ["part_1", "part_2", "other"]),
    "same stem, one folder": (["a/cam1.mp4", "a/cam1.avi"], [], ["cam1-2", "cam1"]),
    "case only": (["a/Cam1.mp4", "b/cam1.mp4"], [], ["a-Cam1", "b-cam1"]),
    "shared tail": (["mnt/x/exp/cam1.mp4", "media/x/exp/cam1.mp4"], [],
                    ["mnt-x-exp-cam1", "media-x-exp-cam1"]),
    "literal name equals a qualified one": (
        ["day1/cam1.mp4", "day2/cam1.mp4", "other/day1-cam1.mp4"], [],
        ["day1-cam1", "day2-cam1", "day1-cam1-2"]),
    "beside a plain recorded clip": (["day2/cam1.mp4"], [("day1/cam1.mp4", "cam1")], ["day2-cam1"]),
    "sibling of qualified recorded clips": (
        ["day3/cam1.mp4", "day1/cam2.mp4"],
        [("day1/cam1.mp4", "day1-cam1"), ("day2/cam1.mp4", "day2-cam1")],
        ["day3-cam1", "day1-cam2"]),
}
```

  - `test_names_do_not_depend_on_input_order`: shuffle the "two days" videos with `random.Random(0)`; the `{path: name}` mapping is identical.
  - `test_a_recorded_clip_keeps_its_name`: `recorded=[("day1/cam1.mp4", "weird")]`, video list includes that clip: its name is `"weird"`.
  - `test_illegal_characters_in_folders_become_underscores`: `["a:b/cam1.mp4", "c?d/cam1.mp4"]` gives `["a_b-cam1", "c_d-cam1"]`; a lone `x:y.mp4` keeps stem `"x:y"`.
  - `test_drive_letters_are_not_part_of_the_chain`: `_folder_chain(PureWindowsPath("D:/exp/day1/cam1.mp4")) == ["exp", "day1"]`; `_folder_chain(PurePosixPath("/d/day1/cam1.mp4")) == ["d", "day1"]`.
  - `test_one_file_spelled_two_ways_is_one_clip`: absolute path and the same path relative after `monkeypatch.chdir(tmp_path)`; both keys map to the same name and neither carries a `-2`.
  - `test_twenty_thousand_clips_are_named_quickly`: `Path(f"/data/exp{i % 400}/cam{i // 400}.mp4")` for 20,000 `i`; finishes in under 5 seconds and all names are unique case-folded.
  - `recorded_clips`: `test_no_manifest_records_nothing`; `test_an_unreadable_manifest_records_nothing` (`"{not json"`); `test_records_without_a_source_path_are_ignored`; `test_a_source_video_with_an_extension_is_normalised` (`"cam1.mp4"` returns name `"cam1"`); `test_a_clip_whose_frames_were_all_deleted_holds_nothing`; `test_a_clip_with_one_frame_left_still_holds_its_name`; `test_paths_come_back_as_recorded`. A `write_manifest(out, records)` helper touches each frame file and writes `{"frames": records}`.
  - `describe_renames`: `test_a_contested_clip_names_the_one_it_shares_with` (line contains the new name, `"shares a file name with"` and the other clip's path); `test_a_folder_mate_says_why` (`day1/cam2` line contains `"same folder as"` and `day1/cam1`'s path); `test_nothing_renamed_gives_no_lines`.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_names_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: collection error, `ImportError` for `assign_names` / `recorded_clips` / `_folder_chain`.

- [ ] **Step 3: Implement `recorded_clips` and the helpers `_folder_chain` and `_sanitise(part: str) -> str`** in `sampling.py`. Group manifest records by `(source_path, source_video)`; a clip counts if any of its frame files is still in `output` (stop at the first, so a large manifest costs about one stat per clip). Strip a trailing video suffix from `source_video` using `VIDEO_SUFFIXES`. Skip records without `source_path` or `source_video`, and return `[]` for a missing, unreadable or malformed manifest. Run the `recorded_clips`, `_folder_chain` and sanitising tests: pass.

- [ ] **Step 4: Implement `assign_names`** following spec section "Naming rule", rules 1 to 8, which the table above pins. Index `stem -> set of folders` once so naming is O(n); an O(n²) scan fails the 20,000-clip test. Clips whose `_location` is equal are one clip: they share one name and count once. Raise `SamplingError` if the final names are not unique case-folded. Run the whole file except `describe_renames`: pass.

- [ ] **Step 5: Implement `describe_renames`.** Line format `f"{video} -> {name}  (shares a file name with {other})"`, where `other` is the first clip, sorted by path, among `videos` and the recorded paths with the same case-folded stem and a different location; for a renamed clip with no such partner, `f"{video} -> {name}  (same folder as {mate}, which shares a file name)"`, where `mate` is the first clip in its folder that has a partner. Reuse the stem index from `assign_names` rather than restating the contest test. Run `tests/sampling_names_test.py`: all pass.

- [ ] **Step 6: Format and commit**

```bash
black src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py tests/sampling_names_test.py
isort --profile black tests/sampling_names_test.py
git add src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py tests/sampling_names_test.py
git commit -m "Assign unique clip names from folders, taking the manifest into account"
```

---

### Task 2: Use the names when sampling, and in the CLI

**Files:**
- Modify: `.../detectron2_pipeline/sampling.py` (`VideoPlan`, `plan_sampling`, `sample_frames`; delete `check_distinct_stems` and `check_against_folder`)
- Modify: `.../detectron2_pipeline/cli.py` (`sample_main`)
- Test: `tests/sampling_folders_test.py` (rewrite the refusal tests), `tests/sampling_names_test.py`

**Interfaces:**
- Consumes: `assign_names`, `recorded_clips`, `describe_renames`, `_location` (Task 1 and #9).
- Produces: `VideoPlan.name: str` (last field, defaults to `video.stem` via `__post_init__`); `plan_sampling(..., names: Mapping[Path, str] | None = None)`, which assigns names with no recorded clips when `names` is `None`; `SamplingResult.per_video` keyed by clip name.

- [ ] **Step 1: Write the failing tests.** In `tests/sampling_folders_test.py` replace `test_a_clip_with_the_same_name_from_elsewhere_is_refused` and delete the two tests that call `check_against_folder` (their cases moved to Task 1). Add, using the existing `make_video`, `sample` and manifest-reading helpers:
  - `test_same_named_clips_in_one_run_get_folder_names`: `day1/cam1.avi` and `day2/cam1.avi` in one run; `{f["source_video"]}` is `{"day1-cam1", "day2-cam1"}`, files `day1-cam1_f*.png` and `day2-cam1_f*.png` both exist, and `result.per_video` has both keys.
  - `test_a_later_run_names_the_newcomer_and_leaves_the_earlier_frames_alone`: sample day1 then day2 into one folder; source names are `{"cam1", "day2-cam1"}`; the day1 frame files are byte-identical before and after.
  - `test_frames_that_were_deleted_no_longer_hold_the_name` (existing): add the assertion that the day2 clip's `source_video` is `"cam1"`.
  - `test_one_file_listed_twice_is_sampled_once`: `sample([clip, Path("day1") / "cam1.avi"])` after `monkeypatch.chdir(tmp_path)`; one source name `"cam1"`, and exactly `n` frames written.
  - `test_plan_sampling_carries_the_names`: `plan_sampling(videos, 10, counts=[100, 100], names={...})` sets `plan.name`; a hand-built `VideoPlan(video, 10, 5)` has `name == video.stem`.
  - `test_the_cli_names_clips_that_share_a_file_name`: real videos `other/day1/cam1.avi` and `other/day2/cam1.avi`, `--recursive`; the run succeeds, `capsys` output contains `"named by folder"`, `day1-cam1` and `day2-cam1`, and the frames exist.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_folders_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: the new tests FAIL (refusal raised, or `names` unexpected).

- [ ] **Step 3: Implement `VideoPlan.name` and `plan_sampling(names=...)`.** The plan carries the name; allocation and picking are untouched.

- [ ] **Step 4: Change `sample_frames`.** Drop videos whose `_location` repeats, keeping the first spelling. Compute `names = assign_names(videos, recorded_clips(output))` and pass it to `plan_sampling`. Write `f"{plan.name}_f{index:06d}.{image_format}"`, record `"source_video": plan.name`, key `per_video` by `plan.name`, and use `plan.name` in the `unreadable` entries. Delete `check_distinct_stems` and `check_against_folder` and their calls.

- [ ] **Step 5: CLI.** In `sample_main`, compute `recorded = recorded_clips(args.output)` and `names = assign_names(videos, recorded)`, pass `names=names` to `plan_sampling`, and after the per-clip plan print `"\nClips named by folder, because their file names are shared:"` followed by each line of `describe_renames(videos, names, recorded)` indented two spaces; print nothing when there are none.

- [ ] **Step 6: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_folders_test.py tests/sampling_names_test.py tests/metadata_sidecars_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
black src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py tests/sampling_folders_test.py
git add -u && git commit -m "Name sampled frames from the clip's folders when file names are shared"
```

---

### Task 3: Training-split preview reads the recorded names

**Files:**
- Modify: `.../detectron2_pipeline/dataset.py` (add `sources_in_folder` near `source_video_of`)
- Modify: `.../notebooks/colab_detectron2_pipeline.ipynb` (cell 13 only)
- Create: `tests/clip_names_dataset_test.py`

**Interfaces:**
- Consumes: `sample_frames` naming (Task 2); `labelme_document(image_path, width, height, polygons, label)` from `prelabel.py` to write annotations in tests.
- Produces: `sources_in_folder(input_dir: Path) -> list[str]`: sorted unique source names of the images (`IMAGE_SUFFIXES`) in the folder, from `sampling_manifest.json` when it names the frame, else from the file name via `source_video_of`. Never raises on a malformed manifest.

- [ ] **Step 1: Write the failing tests** in `tests/clip_names_dataset_test.py`:
  - `test_sources_come_from_the_manifest`: frames written by `sample_frames` from `day1/cam1.avi` and `day2/cam1.avi` give `["day1-cam1", "day2-cam1"]`.
  - `test_without_a_manifest_the_file_name_gives_the_source`: `day1-cam1_f000012.png` alone gives `["day1-cam1"]`.
  - `test_a_malformed_manifest_falls_back_to_file_names` and `test_non_image_files_are_ignored`.
  - `test_pieces_of_one_qualified_recording_stay_together`: sample four tiny videos `day1/cam1_segment_1.avi`, `day1/cam1_segment_2.avi`, `day2/cam1_segment_1.avi`, `day2/cam1_segment_2.avi`; `group_videos(sources_in_folder(folder))` has exactly the groups `day1-cam1` and `day2-cam1`, two members each.
  - `test_the_split_keeps_each_recording_whole`: for the same frames write a LabelMe file per frame (one 4-point polygon, label `fish`); `build_coco_dataset(DatasetRequest(input_dir=..., output_dir=..., val_fraction=0.5, single_class="fish"))` gives `train_videos` and `val_videos` disjoint, one group each, together `{"day1-cam1", "day2-cam1"}`.
  - `test_the_notebook_previews_the_recorded_names`: load the notebook JSON from `Path(__file__).resolve().parents[1] / "src/idtrackerai/extra_tools/detectron2_pipeline/notebooks/colab_detectron2_pipeline.ipynb"`; the cell whose source contains `"before it is acted on"` contains `sources_in_folder(FRAMES)` and no `v.stem`.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/clip_names_dataset_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: FAIL with `ImportError`/`AttributeError` for `sources_in_folder`, and the notebook test failing on `v.stem`.

- [ ] **Step 3: Implement `sources_in_folder` in `dataset.py`.** Standard library only, so it works when `dataset.py` is loaded flat on Colab.

- [ ] **Step 4: Edit cell 13** with the Edit tool on the raw JSON (the file has non-ASCII characters, so do not round-trip it through `json.dump`). Replace the comment, `groups = ...` and first `print` lines so the cell computes `sources = ds.sources_in_folder(FRAMES)`, then `groups = ds.group_videos(sources)`, and prints `len(sources)` instead of `len(videos)`. Keep the cell's other lines. Confirm `git diff` shows only cell 13.

- [ ] **Step 5: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/clip_names_dataset_test.py tests/bundle_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
black tests/clip_names_dataset_test.py
git add tests/clip_names_dataset_test.py src/idtrackerai/extra_tools/detectron2_pipeline/dataset.py src/idtrackerai/extra_tools/detectron2_pipeline/notebooks/colab_detectron2_pipeline.ipynb
git commit -m "Preview the training split from the names recorded with the frames"
```

---

### Task 4: The Detectron2 panel shows and uses the names

**Files:**
- Modify: `src/idtrackerai/segmentation_app/widgets/detectron2_panel.py`
- Modify: `src/idtrackerai/segmentation_app/tooltips.toml`
- Test: `tests/sampling_folders_test.py` (panel section; reuse the `d2_panel` fixture)

**Interfaces:**
- Consumes: `assign_names`, `recorded_clips`, `VideoPlan.name`, `plan_sampling(names=...)` (Tasks 1 and 2).
- Produces: `Detectron2Panel.clip_names() -> dict[Path, str]` (the cached result of `_refresh_names()`).

- [ ] **Step 1: Write the failing tests** (real tiny videos via `make_video`):
  - `test_clips_sharing_a_file_name_are_listed_by_their_folder_name`: after `set_video_paths([day1/cam1.avi, day2/cam1.avi, other/cam2.avi])` the list item texts are `"day1-cam1  (cam1.avi)"`, `"day2-cam1  (cam1.avi)"`, `"cam2.avi"`.
  - `test_the_summary_counts_recordings_by_name`: the summary text contains `"2 recordings"` for the two colliding clips.
  - `test_the_preview_says_how_many_are_named_by_folder`: after counts are ready, `sample_preview` text contains `"2 clips will be named by folder"`.
  - `test_the_output_folder_decides_what_is_already_taken`: sample `day1/cam1.avi` into `out` with `sample_frames`, then list day1 and day2 and `frames_dir.setText(str(out))`; the items read `"cam1.avi"` and `"day2-cam1  (cam1.avi)"`.
  - `test_without_an_output_folder_no_manifest_is_read` (Review Focus): `monkeypatch.chdir` into a folder holding a `sampling_manifest.json` that records `day1/cam1.avi` as `cam1`; with `frames_dir` empty, listing day2's `cam1.avi` alone gives plain `"cam1.avi"`.
  - `test_the_grouping_editor_shows_the_names`: monkeypatch `QDialog.exec` to a function that finds the dialog's `QTableWidget`, records its first-column texts and returns `QDialog.DialogCode.Rejected`; they are `["day1-cam1", "day2-cam1"]`.
  - In `test_the_subfolder_choice_is_explained`, also assert `"same file name" not in tips["d2_include_subfolders"]`.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_folders_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: the new panel tests FAIL.

- [ ] **Step 3: Implement `_refresh_names()` and `clip_names()`.** `_refresh_names` recomputes the cache from `video_paths` and `Path(self.frames_dir.text())`, treating an empty text as "no recorded clips". Call it from `set_video_paths` before the list is built, from a `textChanged` handler on `frames_dir` (connect it at the end of `__init__`, since `setText` is called early) that also relabels the items and refreshes the summary, grouping status and preview, and at the top of `_sampling_finished`.

- [ ] **Step 4: Use the names.** List item label is `name` followed by two spaces and `(path.name)` in parentheses when `name != path.stem`, otherwise `path.name`. In the group editor (both `video.stem` sites), `_update_grouping_status` and `_update_video_summary`, replace stems with `self.clip_names()[video]`. Pass `names=self.clip_names()` to `plan_sampling` in `update_preview`, and append `f" {n} {'clip' if n == 1 else 'clips'} will be named by folder, because file names are shared."` when `n = sum(1 for p in plans if p.name != p.video.stem)` is positive.

- [ ] **Step 5: Tooltips.** In `tooltips.toml`, replace the last sentence of `d2_include_subfolders` (the one saying same-named clips cannot share frames) with: `Clips that share a file name are told apart by their folder names, for example day1-cam1 and day2-cam1.`

- [ ] **Step 6: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_folders_test.py tests/segmentation_model_ui_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -u && git commit -m "Show and use folder-based clip names in the Detectron2 panel"
```

---

### Task 5: Export refuses repeated file names everywhere

**Files:**
- Modify: `sampling.py` (add `repeated_stems`), `sam3_panel.py` (`run_export`), `detectron2_panel.py` (`_update_export_command`)
- Test: `tests/sampling_names_test.py`, `tests/sam3_panel_test.py`, `tests/sampling_folders_test.py`

**Interfaces:**
- Produces: `repeated_stems(videos: Sequence[Path]) -> dict[str, list[Path]]`: each exact `stem` shared by two or more videos, mapped to those paths (exact comparison, as `inference.py` does for its outputs).

- [ ] **Step 1: Write the failing tests.**
  - `tests/sampling_names_test.py::test_repeated_stems_finds_only_shared_stems`: `["a/cam1.mp4", "b/cam1.mp4", "a/cam2.mp4"]` gives `{"cam1": [a/cam1.mp4, b/cam1.mp4]}`; unique stems give `{}`.
  - `tests/sam3_panel_test.py::test_export_refuses_clips_that_would_share_a_contour_file`: with the `panel` fixture, a CUDA report, `panel.weights` set and `video_paths` of `a/cam1.mp4` and `b/cam1.mp4`, patch `panel_mod.QMessageBox.warning` to record its text and `panel.export_thread.start` to record calls; `run_export()` calls `warning` once with text containing `"Two videos would write the same contour file (same file name in different folders). Export them separately."` and both paths, never starts the thread, and does not create the output directory.
  - `tests/sampling_folders_test.py::test_the_export_step_warns_before_the_command_is_copied`: after `set_video_paths` with the two colliding clips, `d2_panel._update_export_command()` leaves `export_status` text beginning `"2 clips share a file name"` and containing `"will refuse"`; with no collision the text does not contain it.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sampling_names_test.py tests/sam3_panel_test.py tests/sampling_folders_test.py -q -p no:cacheprovider -o log_file=/dev/null`
Expected: the three new tests FAIL.

- [ ] **Step 3: Implement `repeated_stems`** in `sampling.py`.

- [ ] **Step 4: Guard `Sam3Panel.run_export`** before it creates `output_dir`: when `repeated_stems(self.video_paths)` is non-empty, show `QMessageBox.warning(self, "Clips share a file name", text)` where `text` is the sentence from the test followed by one `stem: path, path` line per shared stem, and return.

- [ ] **Step 5: Warn in `Detectron2Panel._update_export_command`:** using `repeated_stems(self.video_paths)`, prefix `export_status` with `f"{n} clips share a file name, so the exporter will refuse this list; export them in separate runs. "` where `n` is the number of clips involved (the sum over shared stems).

- [ ] **Step 6: Run to verify pass** (same command as Step 2). Expected: all pass.

- [ ] **Step 7: Commit**

```bash
black src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py tests/sampling_names_test.py
git add -u && git commit -m "Refuse repeated file names in the in-app SAM 3 export and warn in the Detectron2 export step"
```

---

### Task 6: Documentation and final check

**Files:**
- Modify: `docs/detectron2-pipeline.md`, `USER_GUIDE.md`, `docs/source/user_guide/changelog.rst`

- [ ] **Step 1: `docs/detectron2-pipeline.md`.** Replace the paragraph that says same-named clips "cannot share one frames folder yet" and are refused with one that says: clips with the same file name are named from their folders (`day1-cam1`, `day2-cam1`), a folder's clips all get its prefix, frames already in the folder are never renamed, the plan lists the renamed clips before anything is written, and a clip moved to a new location is treated as a new clip.

- [ ] **Step 2: `USER_GUIDE.md`.** After the sentence about **Include subfolders**, add one sentence that clips sharing a file name are named by folder and shown that way in the list.

- [ ] **Step 3: `changelog.rst`.** Reword the bullet "Fixed sampling a clip into a folder that already held frames from a *different* clip..." so it ends with the clip being given its own name built from its folders, not refused; keep the sentence that the manifest records absolute source paths. Add a bullet for the SAM 3 export refusing repeated file names.

- [ ] **Step 4: Full verification**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests -q --ignore=tests/smoke_test.py -p no:cacheprovider -o log_file=/dev/null`
Expected: all pass (257 before this work, plus the new tests).
Run: `black --check src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py tests/sampling_names_test.py tests/clip_names_dataset_test.py && isort --check --profile black tests/sampling_names_test.py tests/clip_names_dataset_test.py`
Expected: no changes needed.

- [ ] **Step 5: Run the real CLI once** on a temporary tree with `day1/cam1.avi` and `day2/cam1.avi` (any tiny MJPG videos): `idtrackerai_d2_sample --videos <tree> --recursive --n-frames 6 --output <out>` prints the "named by folder" block and writes `day1-cam1_f*.png` and `day2-cam1_f*.png`.

- [ ] **Step 6: Commit and push**

```bash
git add -u && git commit -m "Document folder-based clip names"
git push -u origin fatina-del-codice/inspiring-albattani-89oy5q
```
