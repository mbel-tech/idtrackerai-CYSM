# Detectron2 segmentation for idtracker.ai

Replaces idtracker.ai's intensity thresholding with a fine-tuned Mask R-CNN,
while leaving identity tracking — crossing detection, fragmentation, the
identification network, gap closing — exactly as it was.

Contours go straight from the detector into `Blob`. Nothing is painted back into
a video and re-thresholded, so there are no per-clip intensity or area
thresholds to tune.

## The pipeline

```
  LOCAL (the Segmentation App, or these scripts)
    1  enhancement        choose it against your own footage
    2  sample frames      from every clip you list, across the whole of each
    3  annotate           LabelMe, one polygon per animal
    4  build dataset      validate, convert, split train/val by recording
                    |
  NEEDS A CUDA GPU (this machine if it has one, otherwise Colab)
    5  train              fine-tune Mask R-CNN, record the settings used
    6  export contours    run the model over each clip -> one .h5 per clip
                    |
  LOCAL
    7  track              external_contours = one .h5 per clip of the session
```

There are two ways through this, and which is better depends on one thing: do
you have a CUDA card?

**With one**, the Segmentation App does steps 1–4 and runs training itself; it
checks for the card rather than assuming, and hands over the export as a
command, since a collection takes days and that does not belong behind a window
you cannot close. Nothing leaves the machine.

**Without one** — which is the common case for people tracking animals — the
app still does steps 1 to 3, and the Colab notebook takes over from your
annotated frames. The **Get Colab bundle...** button hands you the notebook at
exactly that point. Uploading the clips is the price, and it buys a GPU you do
not have.

**A recording saved in several clips is tracked as one session.** Give
`external_contours` one file per video path, in the same order; the Segmentation
App pairs them by name, so the order is right by construction. Identities are
then carried across the joins instead of restarting at each segment.

> Step 3 is the slow one. [SAM 3](sam3-pipeline.md) can draft the outlines
> first, so it becomes correcting rather than drawing. SAM 3 can also replace
> steps 1–6 entirely, at the cost of a model that has never seen your setup.

The enhancement settings chosen in step 1 travel all the way to step 6. That is
the point of the profile file: the model must see the same kind of image at
inference that it was annotated on, and training on CLAHE-enhanced frames while
predicting on raw ones costs accuracy without raising an error. The settings are
written beside the frames, copied into the dataset, recorded with the weights,
and adopted by the exporter unless you override them.

## The Colab notebook, for a machine without a CUDA card

Steps 1 to 3 happen in the Segmentation App, because each wants your eyes on
the footage: judging the enhancement, seeing how the frames are spread across
the clips, and drawing the polygons. Step 3 is also the one step that could
never move -- LabelMe is a desktop application and a runtime has no display.

When you have finished annotating, the app's **Get Colab bundle...** button on
the Annotate step writes `colab_bundle.zip`, and tells you what else to put on
Drive. The notebook inside it is **steps 4 to 12**:

```
  app     1-3   enhancement, sample frames, annotate in LabelMe
                        |
  Colab   4-5   your annotated frames -> the COCO dataset
          6-8   install Detectron2, check the dataset, train
          9-10  export one contour file per clip, check them
         11-12  one parameter file per recording, then track
```

The two *Setup* sections before step 4 are not pipeline steps: they mount
Drive, unpack the bundle and locate the videos, and you run them at the start
of each session. Everything between sittings lives on Drive, so closing the tab
costs nothing.

The parameter files it writes for tracking **inherit whatever the Segmentation
App saved** - the file from *Save parameters*, the same one a legacy
thresholding run would use - so the number of animals, the area thresholds, the
region of interest and the tracking interval carry over instead of being
retyped. One file per recording, with its clips and their contour files in
matching order.

Tracking on the Colab runtime is offered but **has not been run there**; the
notebook says so where it matters, and the same parameter files work unchanged
on your own machine.

## From the Segmentation App

The shortest route. Open a video, then set **Segmentation → Detectron2
pipeline**:

1. **Enhancement** — a preset (None / Gentle / Standard / Strong), or Custom for
   the individual light edits (exposure, brightness, contrast, gamma, shadows,
   highlights, blacks, whites, sharpness, denoise) alongside local contrast and
   lighting evenness, previewed live on the frame you are looking at. Drag the
   raw↔enhanced slider to compare. Save it as this setup's profile.
2. **Sample frames** — the videos to draw from, how many frames, the selection
   number, the output folder. Runs in the background with a cancellable progress
   bar; a cancelled run keeps the frames it wrote.

   The video list is the panel's own, not the tracking session's. It starts
   from whatever is open, and **Add videos...** / **Add folder...** extend it.
   That separation is the point: a model should see every recording from a
   setup, while a session is the one recording being tracked. Opening fifty
   clips just to annotate them would make one absurd concatenated session out
   of fifty separate experiments.

   **Add folder...** takes the videos directly inside the folder you choose.
   Tick **Include subfolders** first to take every video anywhere beneath it,
   which is how you gather a whole experiment in one click. The number of
   frames is a total for the whole list, shared out in proportion to each
   video's length -- 600 frames from a folder of 40 clips is not 600 per clip --
   and the selection number makes the same choice again. Hidden folders and
   macOS `._` companion files are skipped.

   Frames are named after the video file (`cam1.mp4` gives `cam1_f000123.png`),
   and the train/validation split treats each name as one source. Clips with the
   same file name in different folders (`day1/cam1.mp4`, `day2/cam1.mp4`) would
   collide, so they are named from their folders instead: `day1-cam1` and
   `day2-cam1`. The prefix is the last folder names, joined by a hyphen, and as
   few of them as make the prefixes distinct. A clip keeps its plain file name
   unless its *folder* is involved in a collision; then every clip from that
   folder gets the folder's prefix, so `day1/cam1`, `day2/cam1` and `day1/cam2`
   become `day1-cam1`, `day2-cam1` and `day1-cam2`. That is deliberate: it keeps
   the pieces of one recording (`part_1`, `part_2`) in one train/validation
   group, which renaming only the clip that collided would not. Two files in one
   folder that differ only in their extension (`cam1.mp4`, `cam1.avi`) are
   numbered `cam1` and `cam1-2`.

   Grouping corrections made in the group editor are stored under the clip's
   name, so a correction stored under a plain file name stops applying once a
   collision renames that clip, and nothing tells you.

   Frames already in the output folder are never renamed; the folder's manifest
   decides. A clip sampled earlier under its plain name keeps it, and only later
   arrivals get a prefix: with `day1/cam1.mp4` sampled first, a later
   `day2/cam1.mp4` becomes `day2-cam1` and day1 stays `cam1`. The plan is shown
   before anything is written: `idtrackerai_d2_sample` prints a block headed
   "Clips named by folder, because their file names are shared:", and the app's
   line above the button says how many clips will be named by folder, while the
   clip list shows such a clip as `day2-cam1  (cam1.avi)`.

   A clip that has moved to a new location cannot be told apart from a different
   video with the same name, so it is treated as a new clip with a new name, and
   its frames become a second source in the split. If the earlier frames end up
   on the other side, that is the near-duplicate leak the grouping exists to
   prevent, so read the plan when you re-sample from a moved folder.

   If an older run already left two different clips recorded under one name in
   a folder's manifest, sampling both stops before writing anything, names the
   clips, and says how to recover: sample into a new frames folder, or delete
   the frames of one of the clips from this one.

   The export side is unchanged. Contour files are still named
   `<video file name>.h5`, so clips with the same file name cannot be exported
   together: the command-line exporter and the session loader always refused
   them, the in-app SAM 3 export now does too, and the Detectron2 export step
   warns before the command is copied. Export such clips in separate runs, each
   to its own output folder.

   Above the button, a line says what the run would do -- how many frames from
   how many videos, and the smallest and largest share. If the count is lower
   than the number of videos, it says how many would get nothing at all, which
   is the failure that most defeats the point of a collection.
3. **Annotate** — opens LabelMe on the folder, preloaded with your class name
   and `--validate-label exact` so a typo cannot create a second class. The step
   header counts annotated frames.
4. **Build dataset** — validates every polygon and shows the report inline.
   Only needed if you are training on this machine; the Colab notebook builds
   it for you otherwise.

Close the app whenever you like. Each step records what it produced beside the
data, and status is re-derived from disk on reload, so a deleted folder shows as
incomplete rather than as a step that lies about being done.

If this machine has no CUDA card, use **Get Colab bundle...** on the Annotate
step once the polygons are drawn. It writes the zip and lists what else to put
on Drive.

## From the command line

The same code, for scripting or a headless machine. These are installed as
commands with the package, so they work without cloning the repository:

```bash
# 1. tune the enhancement for this recording setup and save it
idtrackerai_d2_enhance --video clips/clip_01.mp4 --frame 500 \
    --output check.png --save-profile setups/tank_a.json

# 2. pull frames for annotation, spread across the whole of every clip
idtrackerai_d2_sample --videos clips/*.mp4 --n-frames 600 \
    --output annotate/ --preprocess-profile setups/tank_a.json

#    ...or give it a folder: 600 frames in total from every video beneath it,
#    shared out in proportion to length (--recursive includes subfolders;
#    without it, only the videos directly inside the folder are used)
idtrackerai_d2_sample --videos /data/experiment --recursive --n-frames 600 \
    --output annotate/ --preprocess-profile setups/tank_a.json

# 3. annotate in LabelMe: one polygon per animal, one consistent label
python -m labelme annotate/ --labels fish --validate-label exact

# 4. convert and split
idtrackerai_d2_dataset --input annotate/ --output dataset/ \
    --single-class fish --expected-instances 5 --val-fraction 0.15

# 5. check the annotations landed where you think, before spending GPU time
idtrackerai_d2_train --dataset dataset/ --output model/ --check-dataset

# 6. train
idtrackerai_d2_train --dataset dataset/ --output model/ --epochs 40

# 7. one contour file per clip
idtrackerai_d2_export --videos clips/*.mp4 \
    --weights model/model_final.pth --output-dir contours/ --max-instances 5
```

Then either add `external_contours = "contours/clip_01.h5"` to the clip's
`.toml`, or pick the file in the Segmentation App under **Segmentation →
External contours**.

For many sessions, put a `[segmentation_model]` table in each session's `.toml`
instead and run `idtrackerai_d2_export --session sessions/*.toml`: it exports
only the tracked frames, at the right detection cap per session, into the folder
tracking will look in. See [session-toml.md](session-toml.md).

## Things worth knowing

**The train/validation split is by recording, not by frame.** Frames sampled a
second apart show the same animals in the same poses. Splitting those at random
puts near-duplicates on both sides and the validation AP then measures
memorisation. `--split-by random` exists but warns.

Splitting by *file* is not enough either when a recording was saved in pieces:
`B3_N1_segment_1`, `_2` and `_3` are the same animals in the same arena minutes
apart. The pieces are folded into one group, guessed from the file name by
stripping trailing segment markers, numbers and re-encode suffixes -- so
`B3_N1_segment_2_cleaned` also lands under `B3_N1`.

That guess cannot always be right, so it is shown, not applied quietly. The
panel prints the grouping it inferred before you build, and the control offers
**By recording**, **By file** and **Random**; the CLI takes `--group-by`. Two
guards catch the obvious mistakes: a name that would reduce to nothing but a
piece-marker is left whole, because `clip_00` and `clip_01` are two recordings
rather than two pieces of one called "clip"; and if the grouping collapses every
clip into a single group it falls back to splitting by file and says so.

Because whole groups go to one side, the validation set can only land on a group
boundary, so the fraction you ask for is approximate. The report says what was
actually reached when it differs by more than five points.

**Annotations are validated, not trusted.** Two-point polygons, stray clicks,
bounding boxes drawn where a polygon was meant, shapes off the image edge and
label typos that silently become a second class are all caught and reported in
`dataset/report.json`.

**`--on-overlap` decides who resolves occlusions.** `merge` (default) fuses
overlapping animals into one blob so idtracker.ai's crossing detector and
interpolation reconstruct identities — the legacy behaviour. `split` lets the
detector separate them instead: fewer crossings, but mask assignment can flicker
frame to frame and that instability propagates into fragmentation.

**Missing detections are left missing.** If the model finds three animals in a
frame instead of five, that frame gets three contours. Do not fill the gap by
reusing the previous frame's masks: a mask *union* becomes one blob with a stale
centroid, which idtracker.ai reads as a crossing that never happened. Its own
interpolation works from the whole video and handles this properly.

**Polygon area is not mask pixel count.** Contours are approximated
(`CHAIN_APPROX_TC89_KCOS`) and interior holes are dropped by `RETR_EXTERNAL`, so
absolute areas differ from thresholded ones by a few percent. This does not
matter — `ModelArea` recalibrates from the data — but do not compare the numbers
to old runs.

**Watch the individual/crossing split on the first run.** Detectron2 areas are
more consistent than thresholded ones, so their standard deviation is smaller,
and `MODEL_AREA_SD_TOLERANCE` (default 4, in `idtrackerai/utils/confparams.py`)
sets the cutoff at `median + 4·std`. A tighter cutoff can flag large individuals
as crossings. If the log reports implausibly many, raise it.

**Better segmentation is not a substitute for validation.** When animals
genuinely occlude each other, identity assignment remains probabilistic. Check
the hard crossings before trusting a batch of trajectories.

## Running the GPU stages in Colab

Enhancement, sampling, annotation and dataset building stay local. Training
and contour export move to Colab via
the Colab notebook, which `idtrackerai_d2_bundle` packages for you.

```bash
idtrackerai_d2_bundle          # ~27 KB zip of just what Colab needs
```

Upload to `MyDrive/idtrackerai_detectron2/`:

```
colab_bundle.zip
dataset/        from the GUI's step 4, or idtrackerai_d2_dataset
```

**The videos must be on Drive too.** Inference decodes every frame, so the clips
themselves have to be reachable from the runtime, not just the dataset. They can
live anywhere on Drive — set `VIDEOS` in the notebook's config cell — which
matters when they run to tens of GB. A folder someone shared with you has no
path until you add a shortcut to it in My Drive (*Shared with me* → right-click
→ *Organise* → *Add shortcut to Drive*).

Open the notebook in Colab, set a GPU runtime, run the cells. Section 2 checks
the mount, opens every clip, decodes a frame from each, and reports what is
already exported before anything long begins:

```bash
idtrackerai_d2_videos --videos clips/*.mp4 --contours contours/
```

Run it locally too — it exits non-zero on an unopenable or truncated clip.

**Track the same files you exported contours from.** Both the exporter and
idtracker.ai read frame counts with the same `cv2.CAP_PROP_FRAME_COUNT` call, so
they agree by construction — but only for the same file. A re-encoded copy will
be rejected by the sidecar check.

**The export will not finish in one session.** At a plausible 10 fps, 48 clips ×
30 000 frames is about 40 hours, against a session ceiling of 12–24 hours. So it
is built to resume: each clip writes its own `.h5`, reruns skip finished files,
and the file is written to `.partial` and moved into place so a session killed
mid-write cannot leave a truncated file that looks done. Reconnect, rerun the
export cell, repeat.

Two further Colab specifics the notebook handles:

- **Detectron2 is built from source, ~10 minutes.** The wheel is cached to Drive
  and keyed by Python and torch version, so later sessions reuse it and a
  runtime upgrade rebuilds once.
- **Videos are copied to local disk before decoding.** OpenCV seeks constantly,
  and the Drive FUSE mount is slow enough at that to dominate inference time.

What comes back is tens of MB per clip rather than the gigabytes an enhanced
video would be.

## Citing

The tracking is idtracker.ai's; the segmentation here is Detectron2's. Cite
both — see the [repository README](../README.md#citing) for the full list, which
covers the idtracker.ai papers, Detectron2 and Mask R-CNN. Annotation uses
[LabelMe](https://github.com/wkentaro/labelme) (Wada, GPL-3.0) and the image
processing throughout is [OpenCV](https://opencv.org).

## Installing Detectron2

Only training and contour export need it. Everything the GUI does — enhancement,
sampling, dataset building — runs on OpenCV and NumPy alone, so the Segmentation
App works without Detectron2 installed.

```bash
pip install 'torch>=2.1' torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

There is no universal Detectron2 wheel — it builds against the installed
torch/CUDA pair, so install torch first and let it compile against that.
