# Path-based names for same-named clips

Status: draft for review. Follows [mbel-tech/idtrackerai-CYSM#9](https://github.com/mbel-tech/idtrackerai-CYSM/pull/9),
merged, which added folder and subfolder selection and made same-named clips a
hard error.

## Problem

Annotation frames are named after the video file (`cam1_f000123.png`) and the
manifest records that name as the frame's `source_video`. The train/validation
split groups on `source_video`. Two different clips with the same file name in
different folders (`day1/cam1.mp4`, `day2/cam1.mp4`) would therefore overwrite
each other's frames and be treated as one source.

Choosing frames from a folder tree makes this common, because experiments are
often stored as `<day>/<camera>.mp4`. #9 stops it by refusing the run. This
change replaces the refusal with a name that is unique and still readable, taken
from the clip's path.

## What the user sees

Names below were produced by a throwaway prototype of the rule in this document
and then checked against `dataset.group_key`.

| Clips (all under `/d`) | Names used for frames and grouping | Split groups |
| --- | --- | --- |
| `day1/cam1`, `day2/cam1`, `day1/cam2` | `day1-cam1`, `day2-cam1`, `day1-cam2` | one each |
| `a/x/cam1`, `b/x/cam1`, `c/y/cam1` | `a-x-cam1`, `b-x-cam1`, `c-y-cam1` | one each |
| `rec_A/part_1`, `rec_A/part_2`, `rec_B/part_1` | `rec_A-part_1`, `rec_A-part_2`, `rec_B-part_1` | `rec_A-part` (both pieces), `rec_B-part` |
| `rec_A/part_1`, `rec_A/part_2`, `rec_C/other` | unchanged: `part_1`, `part_2`, `other` | as today |
| `a/cam1.mp4`, `a/cam1.avi` (one folder) | `cam1-2`, `cam1` (numbered in sorted path order) | one each |

- A clip keeps its plain file name unless its **folder** is involved in a name
  collision. Then every clip from that folder gets the folder's prefix, not just
  the one that collided. Row 1 shows the consequence: `day1/cam2` is renamed
  though nothing else is called `cam2`. This is deliberate; see decision 3.
- The plan shown before sampling lists every renamed clip and why, in the
  command-line output and in the app's preview line. Nothing is written before the
  user can see it.
- In the app's clip list a renamed clip reads `day2-cam1  (cam1.mp4)`; the tooltip
  still holds the full path.
- The frame total is still a total, shared across clips by length. Selection
  numbers still reproduce the same frames.

Across runs, using the same output folder:

| Already recorded | New clips | Result for the new clips |
| --- | --- | --- |
| `day1/cam1` as `cam1` | `day2/cam1` | `day2-cam1`; day1 stays `cam1` |
| `day1/cam1` as `day1-cam1`, `day2/cam1` as `day2-cam1` | `day3/cam1`, `day1/cam2` | `day3-cam1`, `day1-cam2` (day1 reuses its recorded prefix) |

## Decisions already made

1. **Names are built from path information**, by the program, not by the user.
2. **Frames that already exist are never renamed.** The manifest is the source of
   truth. A clip sampled earlier under a plain name keeps it, so with
   `day1/cam1.mp4` sampled first, a later `day2/cam1.mp4` becomes `day2-cam1`
   while day1 stays `cam1`. Renaming existing frames would orphan annotations
   that may already be drawn.
3. **A folder is qualified as a whole.** Found while reviewing this spec: renaming
   only the colliding clip would turn `rec_A/part_1` into `rec_A-part_1` and leave
   `rec_A/part_2` as `part_2`. Those are pieces of one recording, but
   `dataset.group_key` would put them in different groups, and one recording could
   end up on both sides of the split. Giving every clip from a folder the same
   prefix keeps them together.
4. **The separator is a hyphen.** `group_key` splits names on underscores to fold
   the pieces of one recording together, and a hyphen does not disturb that:
   `rec_A-part_1` and `_2` group as `rec_A-part`, and `day1-cam1_segment_1` and
   `_2` group as `day1-cam1`.
5. **The refusal from #9 is replaced, not kept alongside.**

## Naming rule

`assign_names(videos, recorded)` in `sampling.py` returns one unique name per
video. `recorded` is a list of `(source path, name)` pairs read from the output
folder's manifest, counting only frames still on disk (a deleted batch holds no
name, as in #9). A clip's *folder* is its immediate parent directory.

1. **Recorded clips keep their recorded name**, matched by location (the
   `realpath` comparison #9 introduced, so a relative path and an absolute path to
   the same file agree). A folder that already has qualified clips on record keeps
   the prefix they were recorded with: the prefix is the recorded name with
   `-<stem>` removed.
2. **Every other clip is new.** A new clip is *contested* if its stem, compared
   case-folded, equals the stem of any clip in a different folder (another new
   clip, or a recorded one) or equals a name already recorded for another clip.
3. **Folders to qualify** are the parent folders of the contested new clips. Every
   new clip in one of those folders, contested or not, takes its folder's prefix.
   New clips in a folder that has a recorded prefix reuse it. All other new clips
   keep their plain stem.
4. **A folder's prefix** is its last `d` folder names joined by `-`, with `d` the
   smallest value at which the prefixes of all folders being qualified are
   distinct from each other and from recorded prefixes, and no resulting name is
   already taken. One `d` is used for all of them, so a deep collision elsewhere
   in the tree can lengthen an unrelated pair's prefix; that is accepted for the
   sake of a rule that is easy to state. If no depth separates them, the deepest is
   used and rule 5 finishes the job.
5. **Leftovers are numbered.** Any name still taken, whether from folders with
   identical chains (the same path on two Windows drives, whose drive letter is not
   part of the chain), the same stem with two extensions in one folder, or a literal
   file name that equals a qualified one, gets `-2`, `-3`, ... in sorted-location
   order. The first keeps the bare name.
6. **Folder parts are sanitised**: characters illegal in Windows file names
   (`<>:"/\|?*` and control characters) become `_`. A stem is never altered, so
   existing names are unchanged. Uniqueness is checked after sanitising.
7. **Case-insensitive collisions count.** `Cam1` and `cam1` collide on Windows and
   macOS defaults, so every comparison is case-folded. Display keeps the original
   case.
8. **Deterministic.** Clips are processed in location order, so the same inputs
   always give the same names regardless of the order they were given in.

Path parts come from `os.path.abspath` (the folder names the user chose, not
symlink targets). A final check raises `SamplingError` if the result is not
unique; it should be unreachable and exists to make a bug loud instead of silent.

## Where it plugs in

### `sampling.py`
- `assign_names(videos, recorded)` and `recorded_clips(output)` as above.
- `VideoPlan` gains `name: str`, defaulting to `video.stem`. `plan_sampling` takes
  an optional `names` mapping so the app's preview can show the names a run would
  use; without it, it assigns names with no recorded clips.
- `sample_frames` assigns names against the output folder's manifest, then writes
  `<name>_f<index>.<fmt>` and records `source_video = name`. `source_path` stays
  absolute, as in #9.
- `check_distinct_stems` and `check_against_folder` are removed; their job is now
  done by `assign_names`.
- The manifest format does not change, so manifests written before this change
  remain valid and older readers can read new ones.

### `idtrackerai_d2_sample`
- Prints the clips that were renamed, under the existing per-clip plan.

### `detectron2_panel.py`
- A `clip_names()` helper gives `{path: name}` for the current list and output
  folder, recomputed when either changes.
- The group editor, the grouping status line, the video summary and
  `group_overrides` use the names instead of `video.stem` (four sites, at lines
  397, 426, 452 and 921 on this branch). Overrides already stored under plain stems stay valid,
  because clips outside a collision keep their plain names.
- The clip list shows the qualified name when it differs from the file name.
- The preview line says how many clips will be renamed.
- The warning in the `d2_include_subfolders` and `d2_add_folder` tooltips that
  same-named clips cannot share a frames folder is removed.

### `dataset.py`
- Split logic is unchanged: `source_video_of` already reads `source_video` from
  the manifest, and its fallback from a file name (`rsplit("_f", 1)`) returns the
  qualified name from `day1-cam1_f000012.png`.
- New `sources_in_folder(input_dir)` returns the source name of every frame in a
  folder, so the notebook can preview the grouping the split will really use.

### Colab notebook
- Cell 13 previews the training split from `v.stem` of the Drive videos. With
  qualified names that would no longer match what `build_coco_dataset` does, so
  it switches to `ds.group_videos(ds.sources_in_folder(FRAMES))`. `dataset.py`
  already travels in the bundle, so the helper reaches Colab with no bundle
  change.
- Cell 32 builds one tracking file per recording from stems and pairs
  `<stem>.h5` files. That is the export side and stays on stems.

### Export guard
The exporter already refuses same-named clips: `inference.py` stops when two
videos would write one contour file, and `segmentation_model.py` stops for a
session with repeated file names. Two gaps remain:
- The in-app SAM 3 export (`Sam3Panel.run_export` driving the export thread in
  `pipeline_threads.py`) writes `<stem>.h5` and treats an existing file as already
  done, with no collision check that I could find. `run_export` refuses a list
  with repeated stems, with the same wording as `inference.py`.
- The Detectron2 panel's export step says up front, in `export_status`, that
  clips share a file name and the exporter will refuse them, instead of leaving
  that to be found after the command is copied.

## Out of scope

- Naming of contour files (`<stem>.h5`), their pairing with videos, and tracking
  session names. They have guards and are a separate problem from naming frames.
- The notebook's video discovery, which assumes one flat folder and one glob.
- Moving or renaming a clip's frames after they exist.

## Edge cases and risks

- **A clip that moved is treated as a new clip.** Resampling `/old/day1/cam1.mp4`
  from `/new/day1/cam1.mp4` cannot be told apart from a different video with the
  same name, so it gets a new name and its frames are a second source. If it
  lands in the other half of the split, that is the near-duplicate leak the
  grouping exists to prevent. Mitigation: the plan lists every renamed clip with
  the path it was distinguished from, before anything is written.
- **Asymmetric names across runs** (decision 2) can split a recording's pieces
  across groups when some were sampled before the collision existed and some
  after. Deliberate, visible in the plan, and the group editor is the correction.
- **Long names.** Qualification only goes as deep as needed, so one or two
  folders is typical, but a deep tree with identical leaf folders can produce
  long frame names, which matters against the Windows path limit. There is no cap,
  because a cap would trade informative names for numbered ones.
- **Folder names can influence grouping**, since the name now contains them.
  `group_key` behaves sensibly for the layouts above; the group editor remains the
  correction for any other.
- **Unreachable clips.** Names are computed from path strings and the manifest
  only, so an unplugged drive does not change or block naming.

## Tests

- `assign_names` as a pure function, no video files: every row of both tables
  above; a folder keeping its recorded prefix; the sibling case; case folding;
  identical folder chains and the `-2` fallback; a literal name equal to a
  qualified one; sanitising; the same names under shuffled input.
- `sample_frames` on real tiny videos: two same-named clips in one run and across
  two runs both get distinct frames and distinct `source_video` values; frames
  from the first run are untouched; a deleted batch frees its name. The tests in
  `tests/sampling_folders_test.py` that assert refusal are rewritten to assert
  this.
- Dataset: qualified names land in separate train/validation groups; the pieces
  of one qualified recording stay together; the fallback without a manifest
  returns the qualified name; `sources_in_folder`.
- Panel: list labels, group editor keys, preview line, tooltip text, and
  `group_overrides` surviving unchanged.
- Export guard: SAM 3 `run_export` refuses repeated stems; the Detectron2 export
  status warns.
- Notebook: the bundle test still passes and cell 13 uses `sources_in_folder`.

## Documentation

`docs/detectron2-pipeline.md` (replace the paragraph that says same-named clips
are refused), `USER_GUIDE.md`, the changelog entry from #9 (reworded, since the
refusal it describes goes away), and the two tooltips.

## Delivery

This spec is delivered on its own, as a docs-only pull request, after #9 was
merged without it. The implementation follows the written plan and lands as a
separate change.
