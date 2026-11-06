# Validator modifications

What this fork changes in idtracker.ai's Validator, and why. For anyone
validating trajectories with it, and for anyone re-applying these changes after
a merge from upstream.

**Nothing about tracking changes here.** Segmentation, crossing detection,
fragmentation, identification and gap closing are untouched by everything on
this page. These are changes to the tool you use *after* tracking, to correct
its output by hand — plus the small pieces of `Blob`, `ListOfBlobs` and
`Session` that tool drives.

The starting point is that validating a long recording is hours of manual,
unrepeatable work, and the upstream Validator has no undo, saves only when
asked, and loses everything if a save fails. Most of what follows is about
that.

## At a glance

| | What it does |
|---|---|
| **Ctrl+Z** | Undoes the last edit. Twenty deep, in memory, gone when you close the Validator. |
| **Swap with…** | Exchanges two identities along a fragment, instead of renaming one. |
| **Extend…** | Repeats the clicked centroid over the next N frames. |
| **Merge with…** | Rebuilds one blob from two, for an animal segmentation split in two. |
| **Autosave** | Saves by itself every five minutes, with no dialogs. |
| **Backup on failure** | A save that fails writes to `~/idtrackerai_backups/<session>/` instead of losing the work. |
| **Set Presence Interval** | Right-click an identity's colour. Frames outside the interval stop counting as "Miss id" errors. |
| **Min duration** | Hides errors shorter than N frames. |
| **Filter pair** | Shows only errors belonging to two chosen identities. |
| **N** | Selects the next error in the list. |
| **Flag unambiguous** | Ticks every "Miss id" gap that no other animal comes near (the **Vicinity** distance). |
| **Interpolate all flagged** | Fills every ticked gap with a straight line, as one Ctrl+Z step. |
| **I** | Starts interpolating the selected error, or the selected identity. |
| **PCHIP** | An interpolation mode that cannot overshoot between fixed points. |
| **Hold A / D / ← / →** | Scrubs through frames instead of stepping once. |
| **Video Info tab** | Frame rate, resolution, frame count, animal count, video files. |

## Saving

**A failed save used to look like a successful one.** `SaveSessionObjects` runs
on a worker thread and let any exception escape, where Qt logged it and nothing
else happened: the progress dialog closed, the Validator carried on, and
nothing had been written. Session folders live on network drives and synced
folders that disconnect without warning, so this is not hypothetical.

The save is now wrapped. If it fails, the session and the blobs are written to
`~/idtrackerai_backups/<session name>/` instead, and you are told what failed
and where the work went. If the backup fails too you are told that as well,
and told not to close the Validator, because the work is still in memory.

To resume from a backup, copy `session.json` and `list_of_blobs.pickle` back
into the original session folder once the drive is available again.

**The Validator saves by itself every five minutes.** Autosaves are silent: no
progress dialogs, no focus stolen, no progress bars in the terminal. That is
the whole reason `ListOfBlobs.save` gained a `verbose` argument. An autosave
is skipped while another save is still running, and does nothing when there is
nothing unsaved.

Note that a save — automatic or not — also recomputes and rewrites the
trajectory files. On a long session that is not instant.

## Undo

**Ctrl+Z reverses the last edit.** A propagated identity change that went the
wrong way used to mean reloading the session and losing everything done since
the last save.

Two kinds of snapshot are taken, depending on the edit. Identity edits record
the `user_generated_identities` and `user_generated_centroids` of the blobs
that could be touched — for a propagated edit, that is the whole fragment,
since propagation stops at fragment boundaries. Edits that add or remove blobs
(Merge, Extend, Interpolate) also record the blob lists of the frames involved.

Three things worth knowing:

- **It is in memory and per-session.** Closing the Validator discards the
  stack. It is not a substitute for saving.
- **Twenty edits deep.** Snapshots of a long fragment are not small.
- **Undoing recomputes the whole video.** An edit can have propagated anywhere
  along a fragment and the snapshot does not record how far, so the safe thing
  is to recompute everything. On a long session there is a pause.

## The three new edits

All three are on the dialog that opens when you double-click a blob, next to
the existing **Change id**, **Interpolate**, **Reset** and **Remove**.

### Swap with…

**Exchanges two identities, rather than renaming one.** After a crossing that
the identification network resolved the wrong way round, both animals carry
each other's identity. **Change id** can only rename one of them, which leaves
the other duplicated for the rest of the fragment and turns one error into two.

Swap moves both at once, and propagates along the fragment the way a normal
identity change does — same stack walk, same fragment boundary. A blob that
carries only one of the two identities is renamed rather than skipped, which is
what makes it safe to run along a fragment where only some blobs are crossings.

Swapping the same pair twice puts everything back, so it is its own undo.

### Extend…

**Repeats the clicked centroid over the next N frames.** An animal that stops
moving can stop being segmented — it blends into the background, or settles
against a wall. Interpolating that gap draws a straight line between two points
that are in the same place anyway; stamping the centroid says the same thing in
one step.

It uses the same `ListOfBlobs.add_centroid` as everything else, so a frame that
already has a blob there gets a centroid added to it, and a frame with no blobs
gets a new user-added one.

### Merge with…

**Rebuilds one blob from two.** Segmentation splits an animal often enough — a
fish whose tail falls below the intensity threshold, one partly behind a
structure — and no amount of identity editing fixes it, because the geometry is
wrong: the tracker sees two objects where there is one.

The merged blob is the convex hull of the combined contours. Everything
downstream (centroid, area, bounding box, orientation) is derived from the
contour, so the merged blob is a normal blob. It is marked `added_by_user`, like
blobs you add by hand, so the Validator's reset path removes it rather than
trying to restore identities it never had.

Sharp edges, because this one is structural rather than an identity edit:

- **It finds blobs by identity, not by object.** The blob you clicked does not
  exist as the same object in later frames; its identity does. Two blobs in a
  frame carrying either of the two identities are the ones merged.
- **A convex hull is not the animal's outline.** For a fish bent into a C, the
  hull spans the gap. Area and orientation of a merged blob are approximations.
- **Propagation is bounded.** There is no fragment to stop at, so it stops when
  the two parts stop being separate, and in any case after
  `MERGE_PROPAGATION_FRAMES` (100) frames. Without a bound it would run to the
  end of the video, merging blobs long after the two parts stopped belonging to
  the same animal.

## The error list

Upstream reports four kinds of error and shows all of them. On a long recording
that is a list nobody can work through. Three filters narrow it, and none of
them changes the data — they are display filters.

### Presence intervals

**Says when each animal is actually in the video.** An animal that enters the
arena at frame 4000, or is netted out before the end, has no trajectory outside
the stretch it was there for, and every one of those frames is reported as a
"Miss id". On a long recording that is thousands of rows hiding the handful
that matter.

Right-click an identity's colour swatch in the **Labels** tab and choose **Set
Presence Interval**, then give a frame range as `start-end`, inclusive at both
ends. "Miss id" errors outside it stop being reported. Nothing else is
affected: duplicates, jumps and unidentified blobs are still flagged, and the
trajectories themselves are untouched.

An identity with no interval set is assumed present throughout, so a session
that never uses this behaves exactly as before.

Intervals are stored per session, in `session.json`, under
`presence_intervals`, keyed by the identity number as a string. They survive
save and reload.

### Min duration and Filter pair

**Min duration** hides errors shorter than N frames. A one-frame gap in an
otherwise clean fragment is rarely worth a correction, and there are usually
many of them.

**Filter pair** shows only errors belonging to two chosen identities. Two
animals that keep swapping with each other are worked on as a pair, and
everything else is noise while that is happening.

One catch: **Filter pair also hides every "No id" error**, because those carry
no identity (internally, `-1`). Turn it off to see them again.

### Next [N]

Selects the next error in the list and jumps to it, wrapping round at the end,
so stepping through the list does not need the mouse.

### Flag unambiguous and Interpolate all flagged

**Closes the "Miss id" gaps that nobody could be confused about, all at once.**
Most of a long list of "Miss id" errors is an animal that vanished for a few
frames with nobody else around, and the correction is the same every time: a
straight line. Confirming that, one error at a time, is the slow part.

Each "Miss id" row has a **Flag** checkbox. **Flag unambiguous** ticks it for
every gap where no other animal comes closer than **Vicinity** (default 50 px)
to the straight line across the gap, in any frame of the gap. Other animals
that have no position in a frame are ignored in that frame. Tick or untick any
box by hand afterwards; the button only ever ticks.

**Interpolate all flagged** then fills every ticked gap with that same line, so
what was judged is what is drawn. It is always linear, whatever mode the
interpolation dialog is set to, and the whole batch is one Ctrl+Z step.

Things to know:

- **The flags are lost when the list is rebuilt**, which happens after the fill,
  and whenever a filter or the jumps threshold changes.
- **A gap needs a position on both sides.** One at the very start or end of the
  video has no line to draw. It is never ticked by the button, and if ticked by
  hand it is skipped, and the summary under the buttons says so.
- **"Unambiguous" only looks at where the other animals are.** It does not know
  about walls, structures or a path a fish could not have swum. Skim the result
  before moving on, and use Ctrl+Z if it is wrong.
- **Recomputing is not instant.** The trajectories are recomputed from the
  first to the last frame of the batch, as with any edit.

## The interpolator

**PCHIP, a mode that cannot overshoot.** A cubic or fifth-order spline through
widely spaced fixed points overshoots between them. For a fish that means an
interpolated path that leaves the tank and comes back, which then has to be
corrected by hand. PCHIP is monotone between consecutive points, so it cannot.
It has no spline order, which is why it sits outside the list of orders rather
than in it.

**The default is now Linear, not Cubic.** Filling a short gap, a straight line
between the two fixed points is the claim the data actually supports. Anything
higher-order is inventing motion between them.

Three changes that each save a keystroke or two per correction, which adds up
over hundreds:

- **Enter applies.** Apply is the dialog's default button. Ctrl+A still works.
- **Applying jumps to the end of the gap**, which is where the next error
  usually is.
- **Left-clicking a fixed point jumps to its frame.** The points are drawn on
  the canvas already; previously reaching one meant reading its frame number
  off the drawing and typing it into the player. Left click did nothing in the
  interpolator before, so nothing is displaced — right click still adds and
  moves centroids.

**I** starts an interpolation without going through the double-click dialog:
for the error selected in the list if there is one, otherwise for the selected
identity at the current frame.

## Length calibration

**The line follows the cursor.** Placing a calibration meant clicking two
points blind and only seeing the line afterwards, so one that landed off the
mark had to be redone. The second click now confirms what is already on screen.

This is why `Canvas` grew a `move_event` signal and `setMouseTracking(True)`:
Qt only reports mouse moves with a button held unless you ask for the rest.

Two bugs went with it:

- **Deleting a calibration deleted the wrong one.** Removal matched
  calibrations by their text, so two with the same points and distance were
  indistinguishable and deleting either row removed the first. `CustomList` now
  also reports the row index, and the calibrator deletes by row.
- **A reloaded calibration crashed.** `LengthCalibration` is a `slots=True`
  dataclass whose `from_dict` never set `color`, so the slot did not exist on a
  reloaded object and every later read of it — painting the line, and `asdict()`
  on the next save — raised `AttributeError`.

The distance prompt is now a numeric field with a positive minimum rather than
a text field with a retry loop. The distance is a denominator, so zero was a
division by zero waiting to happen.

## Video player and Video Info

**Holding a frame-step key scrubs.** Every auto-repeated key used to be
dropped, so holding A or D stepped one frame and then stopped. Moving through a
video a frame at a time meant tapping a key several hundred times. Auto-repeat
is let through for A, D, ← and → only; the other shortcuts toggle things, and
repeating a toggle is never what you want.

**The Video Info tab** shows frame rate, resolution, frame count, animal count
and the video files the session was built from. Frame rate and resolution
decide whether a jump is impossible or merely fast, and a session assembled
from several files needs its file list checked before any of its frame numbers
mean anything. It is all in `session.json`, but reading that mid-validation
means leaving the Validator.

## Where the code is

| File | What changed |
|---|---|
| `extra_tools/validator/validation_GUI.py` | Undo stack, Swap/Extend/Merge, resilient save, autosave, presence-interval dialog, `I` and Ctrl+Z |
| `extra_tools/validator/widgets/errors_explorer.py` | Min duration, pair filter, `Next [N]`, presence-interval masking, Flag column, Vicinity, Flag unambiguous, Interpolate all flagged |
| `extra_tools/validator/widgets/gaps.py` | New — the straight line across a gap and the "nobody is near" test, in plain NumPy |
| `extra_tools/validator/widgets/interpolator.py` | PCHIP, Linear default, Enter to apply, click-to-jump, `batch_interpolate` |
| `extra_tools/validator/widgets/length_calibrator.py` | Preview line, delete-by-row, numeric distance prompt |
| `extra_tools/validator/widgets/video_info.py` | New — the Video Info tab |
| `blob.py` | `swap_identity`, `propagate_swap_identity` |
| `list_of_blobs.py` | `merge_blobs`, `save(verbose=…)` |
| `session.py` | `presence_intervals` |
| `GUI_tools/widgets_utils/canvas.py` | `move_event` signal, mouse tracking |
| `GUI_tools/widgets_utils/custom_list.py` | `removedItemIndex` signal |
| `GUI_tools/widgets_utils/id_labels.py` | "Set Presence Interval" context-menu entry |
| `GUI_tools/widgets_utils/video_player.py` | Auto-repeat on the frame-step keys |

All paths are under `src/idtrackerai/`. Each change is a separate commit with
its reasoning in the message; `git log --oneline -- src/idtrackerai/extra_tools/validator`
is the short version.

## Re-applying these after an upstream merge

These changes touch upstream files, so a merge from upstream can conflict with
them. To see exactly what this fork changed in the Validator, diff against the
upstream release the changes were written for:

```bash
git archive <upstream-tag> src/idtrackerai | tar -x -C /tmp/upstream
diff -ruN --strip-trailing-cr --exclude=__pycache__ /tmp/upstream/src/idtrackerai src/idtrackerai
```

The same recipe reconstructs the patch set from any later upstream tag, which is
how these changes were extracted in the first place.

`NOTICE` lists every modified and added file, as GPL-3.0 section 5(a) requires.

## Provenance

These began as hand edits to an installed copy of idtracker.ai 6.0.13 made
while validating fish recordings, kept outside version control. They were
isolated by diffing that copy against a pristine 6.0.13 and brought into the
fork here.

Defects were fixed on the way in rather than carried over: a crash when
swapping on a blob with no fragment, dead code, a `LengthCalibration` load
path that raised, an interpolation mode declared with a spline order it does
not have, and a length-calibration change that had accidentally dropped the
explanatory popup and the filter that keeps incomplete calibrations out of the
saved session.

**Flag unambiguous and Interpolate all flagged came later.** They were in a
second hand-patched copy that the first extraction did not look at, and were
ported separately from it. The gap test is a vectorised rewrite of the original
per-frame loops, and batch interpolation became one undo step. The original also
told the error list that a Jump had been corrected after filling Miss id gaps,
which would have marked a selected Jump as accepted; that is not carried over.

**Most of this is not tested automatically.** There is no `pytest-qt` in this
project, so the GUI cannot be driven by clicks. The logic underneath it — the
identity swap, the merge, the presence-interval masking — was verified
directly; everything above that was checked by hand in the running Validator.
The exception is the flagged-gap tools: `tests/validator_flagged_gaps_test.py`
calls their buttons on a real Validator window without a screen, fills gaps in
real blobs and undoes them.
