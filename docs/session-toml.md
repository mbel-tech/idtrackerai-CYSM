# The session `.toml` with a segmentation model

Upstream idtracker.ai's session keys are documented on its
[usage page](https://idtracker.ai/latest/user_guide/usage.html). This page covers
what this fork adds, and what the upstream keys mean once the animals' outlines
come from a model instead of from thresholding.
[toml-parameters.md](toml-parameters.md) is the full key reference, upstream
keys included.

## One file for both stages

A session segmented by a model runs in two stages, usually on two machines:

1. **Export**, where the GPU is: the model runs over the videos and writes one
   contour file per video.
2. **Track**: idtracker.ai reads those files instead of thresholding.

The `[segmentation_model]` table describes the model once, in the session's own
`.toml`, and both stages read it:

```bash
idtrackerai_d2_export --session sessions/*.toml           # 1. export
idtrackerai --load sessions/R2_01_10_P1_SB.toml --track    # 2. track
```

The exporter runs the model only over the frames in `tracking_intervals`, keeps
at most `number_of_animals` detections per frame, and writes the files to
`contours_dir`. Tracking finds them there, and before segmenting it checks that
they were made with these weights and settings and that they cover the tracking
intervals. A mismatch stops the run with a message naming the file and what
differs; re-running the export redoes exactly the out-of-date files.

## Example

A broiler pen, two GoPro chapters, nine birds, 2 min skipped and 18 min tracked
at 30 fps:

```toml
name = 'R2_01_10_P1_SB'
video_paths = ['/content/drive/MyDrive/broilercamp/Trial 1.1 (2025)/room 2/oct 1/p01/GX010561.MP4',
               '/content/drive/MyDrive/broilercamp/Trial 1.1 (2025)/room 2/oct 1/p01/GX020561.MP4']
number_of_animals = 9
track_wo_identities = true
tracking_intervals = [[3600, 36000]]
roi_list = ['+ Polygon [[102, 88], [1820, 75], [1835, 1010], [95, 1022]]']
area_ths = [300.0, inf]
output_dir = '/content/drive/MyDrive/broilercamp/idtracker_out'
trajectories_formats = ['h5', 'csv_tidy', 'csv']
device = 'cuda'

# Tables go after every plain key: in TOML, anything below a [table] line
# belongs to that table.
[segmentation_model]
backend = 'detectron2'
weights = '/content/drive/MyDrive/broilercamp/model/model_final.pth'
score_threshold = 0.7
contours_dir = '/content/drive/MyDrive/broilercamp/contours/R2_01_10_P1_SB'
```

The same session with SAM 3 instead, no training needed:

```toml
[segmentation_model]
backend = 'sam3'
weights = '/content/drive/MyDrive/broilercamp/model/sam3.pt'
prompt = 'chicken'
score_threshold = 0.5
contours_dir = '/content/drive/MyDrive/broilercamp/contours/R2_01_10_P1_SB'
```

Give each session its own `contours_dir`. Contour files are named after their
video, and GoPro chapters from different cameras can share a name
(`GX010561.MP4`), so a folder shared by several pens could mix them up. The
exporter refuses two different videos writing to one file, but a folder per
session avoids the question.

## `[segmentation_model]` keys

| Key | Default | Meaning |
|---|---|---|
| `backend` | `'detectron2'` | `'detectron2'` (a Mask R-CNN) or `'sam3'` (prompted with a word). |
| `weights` | required | Path of `model_final.pth` (or `sam3.pt`). `'model_zoo'` loads the COCO checkpoint of `config` with no fine-tuning, and then needs `classes`. |
| `contours_dir` | none | Folder of this session's contour files, `<video stem>.h5` each. Needed unless `external_contours` lists the files. |
| `score_threshold` | `0.7` | Detections scoring below this are dropped. |
| `max_instances` | `number_of_animals` | Most detections kept per frame. Leave it unset so it follows the bird count; set it only to cap below or above that. With SAM 3, `0` keeps every detection it reports. |
| `config` | from `training_metadata.json`, else `mask_rcnn_R_50_FPN_3x` | Detectron2 model-zoo config the model was fine-tuned from. |
| `num_classes` | from `training_metadata.json`, else 1 (80 for `model_zoo`) | Detectron2 only. |
| `classes` | all | Class indices to keep, e.g. `[14]` for COCO's bird with `weights = 'model_zoo'`. Detectron2 only. |
| `min_size` | from `training_metadata.json`, else 640 | Short side the frame is resized to for inference. Must match training. Detectron2 only. |
| `prompt` | none | SAM 3 only, and required there: the word for the animal. |
| `on_overlap` | `'merge'` | `'merge'`: overlapping animals become one blob and idtracker.ai's crossing detection takes over. `'split'`: the detector separates them. `'keep'`: masks untouched. |
| `merge_overlap` | `0.15` | With `merge`: fuse two masks whose intersection covers this fraction of the smaller one. |
| `min_component`, `dilate`, `dedup_iou`, `min_area` | `80`, `1`, `0.7`, `1.0` | Mask clean-up, as the exporter's flags of the same names. |
| `device` | `'cuda'` | Torch device for the export. |

Unknown keys are rejected with the list of valid ones, so a typo such as
`score_treshold` fails at once instead of silently exporting at the default.

The enhancement the model sees is the one it was trained with, read from
`training_metadata.json` beside the weights, exactly as for a plain
`idtrackerai_d2_export` run.

### Checked at tracking time

From each contour file's own record, before any frame is segmented:

- **Coverage**: the file holds model output for every frame of
  `tracking_intervals`. Checked whether or not the table is present: frames
  outside an export are empty, and would read as an empty pen.
- **Model**: `backend`, `weights` (by content hash when the weights file is on
  the tracking machine, otherwise by file name), `score_threshold`,
  `on_overlap`, `classes`, `config` and `prompt` match the table.
- **Detection cap**: the file's `max_instances` equals what this session needs,
  so a bird count corrected after export forces a re-export.

## In the Segmentation App

Weights you already have go into the session from the app, with no training:

- **Segmentation → Detectron2 pipeline**, step **6. Export contours**: choose
  the `model_final.pth` under **Weights**, the **Contours folder** and the
  **Score threshold**. Steps 1 to 5 are only for training a new model; leave
  **Weights** empty and the export uses the one step 5 trains.
- **Segmentation → SAM 3**: choose `sam3.pt`, the prompt, the confidence and
  the animals per frame. Its export runs only the tracked frames and redoes a
  clip whose contour file was made with another prompt or threshold.

**Save parameters** writes them as the `[segmentation_model]` table, and opening
that `.toml` again brings them back and opens on the same panel while the
contours are still to be exported. With the contour files loaded (**External
contours**), the table is kept for the model the files record making them.
Choosing **Thresholding** drops it.

The export command the Detectron2 step builds carries the same weights, score
threshold and number of animals, for running in a terminal or in Colab.

## Upstream keys under a segmentation model

| Key | With a model |
|---|---|
| `number_of_animals` | As upstream, and also the per-frame detection cap. |
| `tracking_intervals` | As upstream, and also the frames the exporter runs. |
| `area_ths` | Still applied, as a filter on the model's contours. `[1, inf]` if unset. |
| `roi_list` | Still applied, per contour by its centroid: a bird whose centre is inside the perimeter is kept whole, never clipped at the edge. |
| `intensity_ths` | Ignored. May stay as a placeholder; `[0, 255]` if unset. |
| `use_bkg`, `background_subtraction_stat`, `number_of_frames_for_background` | Ignored: the model has already separated the animals from the background. `use_bkg = true` is harmless and logged. |
| `enhancement` | Applied to the frames identification images are cut from, not to the contours. Irrelevant with `track_wo_identities = true`. |
| everything else | As upstream. |

## `external_contours`

The contour files, one per entry of `video_paths` in the same order. Usually left
out when `[segmentation_model]` has a `contours_dir`, which implies it. Set it to
use files from somewhere else, or to track from a contour file without a model
table:

```toml
external_contours = ['contours/GX010561.h5', 'contours/GX020561.h5']
```

## Running the export in Colab

The Colab bundle (`idtrackerai_d2_bundle`) ships the exporter as
`tools/detectron2_export_contours.py`, which takes the same flag:

```bash
python tools/detectron2_export_contours.py \
    --session '/content/drive/MyDrive/broilercamp/sessions/*.toml' \
    --local-cache /content/cache
```

`--local-cache` copies each video off the Drive mount before decoding, which is
usually several times faster overall. The run ends with a table of how often each
video had no detection, or fewer birds than the session holds: a model trained on
few pens shows its blind spots there first.
