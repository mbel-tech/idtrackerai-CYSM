# Session `.toml` parameters

Every key a session `.toml` accepts in this fork: upstream idtracker.ai's (as of
v6.0.15) and the ones this fork adds for segmenting with Detectron2 or SAM 3.
[session-toml.md](session-toml.md) explains the model route in prose; this page
is the reference.

Syntax, as upstream:
- Keys are case-insensitive.
- `''` means "not set".
- `inf` is valid.
- Use single quotes for Windows paths: inside double quotes, backslashes start
  escape sequences.
- Several files can be loaded at once (`idtrackerai --load a.toml b.toml`); later
  files override earlier ones, and command-line flags override all files.

One rule the fork's tables add: a `[table]` such as `[segmentation_model]` must
come **after every plain key**. Everything below a `[table]` line belongs to
that table.

Sources for the upstream keys: the
[usage page](https://idtracker.ai/latest/user_guide/usage.html),
`start/arg_parser.py`, `session.py` and `utils/confparams.py`.

## 1. The two-stage workflow

When a model segments the video, a session runs in two stages, usually in Colab:

```bash
# 1. export: run the model over the tracked frames, one contour file per video
idtrackerai_d2_export --session '/content/drive/MyDrive/broilercamp/sessions/*.toml' --local-cache /content/cache

# 2. track from those contour files
idtrackerai --load /content/drive/MyDrive/broilercamp/sessions/R2_01_10_P1_SB.toml --track
```

Both stages read the same `.toml`:
- **The export** runs only the frames in `tracking_intervals` and keeps at most `number_of_animals` detections per frame. It writes to `contours_dir` and skips files that are already up to date. It ends with a table showing, per video, the share of frames with no bird or with fewer birds than the pen holds.
- **Tracking** finds the files in `contours_dir`. Before segmenting, it refuses any file that doesn't cover the tracking intervals or that was made with different weights, threshold, prompt or detection cap.

## 2. Keys added by the fork

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[segmentation_model]` | table | none | The model the contours come from. See section 3. |
| `external_contours` | path or list of paths | derived from `contours_dir` | Contour files (`.h5`), one per entry of `video_paths`, in the same order. Only needed if the files live outside `contours_dir`, or if you track from files without a model table. |
| `enhancement` | table | none (off) | Frame enhancement (CLAHE, lighting correction, light edits). See section 5. |
| `presence_intervals` | table | `{}` | Set by the Validator: frames in which each identity is present. Not set by hand. |

## 3. `[segmentation_model]`

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `backend` | `'detectron2'` | both | `'detectron2'` (your fine-tuned Mask R-CNN) or `'sam3'` (prompted with a word). |
| `weights` | **required** | both | Path to `model_final.pth`, or to `sam3.pt`. `'model_zoo'` loads the stock COCO checkpoint of `config` and needs `classes`. `'huggingface'` (sam3 only) fetches `sam3.pt` from Hugging Face, which needs a token with access. |
| `contours_dir` | none | both | This session's contour folder, holding one `<video stem>.h5` per video. **Use one folder per session.** GoPro chapter names such as `GX010561` repeat across cameras. |
| `score_threshold` | `0.7` | both | Detections scoring below this are dropped. |
| `max_instances` | `number_of_animals` | both | Most detections kept per frame. Leave it unset so it follows the bird count for that pen and day. With SAM 3, `0` keeps every detection. |
| `prompt` | none | sam3 (required) | The word for the animal, e.g. `'chicken'`. |
| `config` | from `training_metadata.json`, else `mask_rcnn_R_50_FPN_3x` | detectron2 | The model-zoo config the model was fine-tuned from. |
| `num_classes` | from `training_metadata.json`, else 1 (80 for `model_zoo`) | detectron2 | Number of classes in the model. |
| `classes` | all | detectron2 | Class indices to keep, e.g. `[14]` for COCO's "bird" with `weights = 'model_zoo'`. |
| `min_size` | from `training_metadata.json`, else 640 | detectron2 | The short side frames are resized to for inference. Must match training. |
| `on_overlap` | `'merge'` | both | `'merge'`: touching birds become one blob, and idtracker.ai's crossing logic resolves them. `'split'`: the model separates them. `'keep'`: masks are left untouched. |
| `merge_overlap` | `0.15` | both | With `merge`: fuse two masks when their overlap covers this fraction of the smaller one. |
| `min_component` | `80` | both | Drops mask fragments smaller than this many pixels. |
| `dilate` | `1` | both | Mask dilation, in pixels. |
| `dedup_iou` | `0.7` | both | Two detections overlapping more than this are treated as one. |
| `min_area` | `1.0` | both | Drops polygons with a smaller area. |
| `device` | `'cuda'` | both | Torch device for the export. |

Unknown keys are rejected and the error lists the valid ones, so a typo such as `score_treshold` fails immediately. The enhancement the model sees is the one it was trained with. The exporter reads it from `training_metadata.json` next to the weights.

### What tracking checks, before segmenting

| Check | Fails when |
|---|---|
| Files exist | A file is missing. The message tells you to run `idtrackerai_d2_export --session`. |
| Coverage | `tracking_intervals` includes frames the export didn't run. This is checked even without a model table. |
| Weights | The file was made with other weights. Compared by content hash if the weights file is on the tracking machine, otherwise by file name. |
| Settings | `backend`, `score_threshold`, `on_overlap`, `classes`, `config` or `prompt` differ from the table. |
| Detection cap | The file's cap doesn't match this session's `max_instances`, or the bird count when `max_instances` is unset. For example, a pen count corrected after the export. |

Re-running the export redoes exactly the files that fail these checks.

## 4. Upstream keys when a model segments

| Key | Behaviour with a model |
|---|---|
| `number_of_animals` | Works as upstream, and also sets the per-frame detection cap. |
| `tracking_intervals` | Works as upstream, and also sets which frames are exported. Frames are counted globally across all `video_paths`, with the end frame excluded. |
| `video_paths` | Works as upstream. Each chapter gets its own contour file. |
| `area_ths` | Still applied, as a filter on the model's contours. Defaults to `[1, inf]` if unset. |
| `roi_list` | Still applied, per contour, by its centroid. A bird whose centre is inside the perimeter is kept whole and never clipped at the edge. |
| `exclusive_rois`, `track_wo_identities`, `resolution_reduction`, `output_dir`, `trajectories_formats`, `device`, `data_policy`, ... | Same as upstream. |
| `intensity_ths` | **Ignored.** Can stay as a placeholder. Defaults to `[0, 255]` if unset. |
| `use_bkg`, `background_subtraction_stat`, `number_of_frames_for_background` | **Ignored**, because the model already separates birds from background. `use_bkg = true` is harmless and gets logged. |
| `enhancement` | Applied only to the frames that identification images are cut from, not to the contours. Irrelevant with `track_wo_identities = true`. |

## 5. `enhancement` table (optional)

```toml
[enhancement]
enhance = true
clahe_clip = 1.5           # CLAHE contrast limit
clahe_tile = 8             # CLAHE grid
correct_lighting = true    # flatten uneven illumination
illumination_sigma = 25.0
illumination_downsample = 4
# light edits, each default means "no change":
exposure = 0.0     # -3 to 3
brightness = 0     # -100 to 100
contrast = 0       # -100 to 100
gamma = 1.0        # 0.3 to 3
shadows = 0
highlights = 0
blacks = 0
whites = 0
sharpness = 0      # 0 to 100
denoise = 0
```

The Segmentation App writes this table. A partial table is filled in from the defaults above.

## 6. Setting it from the Segmentation App

- **Detectron2 with weights you already have:** go to *Segmentation → Detectron2 pipeline*, open step **6. Export contours**, and pick the `model_final.pth` under **Weights**. Then set the **Contours folder** and the **Score threshold**. Steps 1 to 5 are only for training a new model.
- **SAM 3:** go to *Segmentation → SAM 3*, pick `sam3.pt` or click **Use Hugging Face**, type the prompt (it starts empty), and set the confidence, animals per frame and **Contours folder**. **Copy command** gives the same export for a machine with a GPU.
- **Save parameters** writes these choices as `[segmentation_model]`. Opening the `.toml` again restores them, and opens on that panel while the contours still need exporting. Choosing **Thresholding** drops the table.

## 7. Complete example: a broiler pen (Detectron2)

```toml
name = 'R2_01_10_P1_SB'
video_paths = ['/content/drive/MyDrive/broilercamp/Trial 1.1 (2025)/room 2/oct 1/p01/GX010561.MP4',
               '/content/drive/MyDrive/broilercamp/Trial 1.1 (2025)/room 2/oct 1/p01/GX020561.MP4']
number_of_animals = 9                  # this pen's count on this day
track_wo_identities = true
tracking_intervals = [[3600, 36000]]   # skip 2 min, track 18 min at 30 fps
roi_list = ['+ Polygon [[102, 88], [1820, 75], [1835, 1010], [95, 1022]]']
area_ths = [300.0, inf]
use_bkg = true                         # ignored with a model; harmless
intensity_ths = [0, 155]               # ignored with a model; placeholder
resolution_reduction = 1
output_dir = '/content/drive/MyDrive/broilercamp/idtracker_out'
trajectories_formats = ['h5', 'csv_tidy', 'csv']
device = 'cuda'

[segmentation_model]
backend = 'detectron2'
weights = '/content/drive/MyDrive/broilercamp/model/model_final.pth'
score_threshold = 0.7
contours_dir = '/content/drive/MyDrive/broilercamp/contours/R2_01_10_P1_SB'
```

The same session with SAM 3 replaces the table with:

```toml
[segmentation_model]
backend = 'sam3'
weights = '/content/drive/MyDrive/broilercamp/model/sam3.pt'
prompt = 'chicken'
score_threshold = 0.5
contours_dir = '/content/drive/MyDrive/broilercamp/contours/R2_01_10_P1_SB'
```

The paths, ROI and thresholds above are illustrative.

## 8. Exporter flags alongside `--session`

`idtrackerai_d2_export` options that aren't about the model can still be passed
with `--session`: `--local-cache`, `--keep-cache`, `--overwrite`,
`--progress-every` and the enhancement flags. Model flags such as `--weights`
are refused with `--session`, because the table sets them.

The run ends with a table, per video, of the share of frames with no detection
or with fewer than the number of animals. A model fine-tuned on few setups shows
its blind spots there first.

## 9. Upstream keys

Unchanged from idtracker.ai. Section 4 says how some of them behave when a model
segments.

### Basic parameters (written by "Save parameters" in the Segmentation app)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | video file names joined with `&` | Session name. The output folder is `session_<name>`. |
| `video_paths` | list of paths | (required) | Video files, tracked as one continuous video in the order listed. |
| `intensity_ths` | [min, max] | none | Blob intensity thresholds. With background subtraction on, the second value is the background-difference threshold. |
| `area_ths` | [min, max] | none | Blob area thresholds in pixels. |
| `tracking_intervals` | list of [start, end] or `''` | `''` (whole video) | Frames to track, counted globally across all `video_paths`. Example: `[[3600, 36000]]`. |
| `number_of_animals` | int | 0 | Number of animals in the video. |
| `use_bkg` | bool | false | Compute a background and subtract it. |
| `resolution_reduction` | float 0 to 1 | automatic | Resolution reduction factor for identification images. 1 = no reduction. |
| `roi_list` | list of str | none | ROI polygons, for example `['+ Polygon [[x,y], ...]']`. `+` adds an area, `-` removes one. |
| `exclusive_rois` | bool | false | Treat each separate ROI as a closed group of identities, for example one pen per ROI. |
| `track_wo_identities` | bool | false | Track positions only, without the AI identification step. |
| `check_segmentation` | bool | false | Abort if any frame has more blobs than animals. |

### Output

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `''` (folder of the first video) | Where the session folder is created. |
| `trajectories_formats` | `["h5", "npy", "csv"]` | Any of `h5`, `npy`, `csv`, `csv_tidy`, `pickle`, `parquet`. |
| `bounding_box_images_in_ram` | false | Keep intermediate crops in RAM instead of on disk. This only helps on slow disks. |
| `data_policy` | `"idmatcher.ai"` | What to keep after tracking: `trajectories`, `validation`, `knowledge_transfer`, `idmatcher.ai` or `all`. |

### Background subtraction

| Key | Default | Meaning |
|---|---|---|
| `background_subtraction_stat` | `"median"` | `median`, `mean`, `max` or `min`. |
| `number_of_frames_for_background` | 50 | Frames sampled, spread evenly across the tracking intervals. |

### Parallel processing

| Key | Default | Meaning |
|---|---|---|
| `number_of_parallel_workers` | 0 | 0 = half the CPUs (max 4). A negative value = CPUs minus that value. 1 = no multiprocessing. |
| `frames_per_episode` | 500 | Maximum frames per chunk ("episode") processed in parallel. |

### Knowledge and identity transfer

| Key | Default | Meaning |
|---|---|---|
| `knowledge_transfer_folder` | none | Previous session or accumulation folder to reuse a trained model from. |
| `id_image_size` | automatic | Size of identification images in pixels. |

### Contrastive learning

| Key | Default | Meaning |
|---|---|---|
| `disable_contrastive` | false | Skip contrastive learning and go straight to accumulation. |
| `contrastive_max_mbytes` | automatic | RAM limit for preloading identification images. |
| `contrastive_min_accumulation` | 0.5 | Fraction of images that must be accumulated to accept the contrastive step. |
| `contrastive_batchsize` | 400 | Image pairs per batch. Larger values need more GPU memory. |
| `contrastive_silhouette_target` | 0.91 | Silhouette score at which contrastive training stops. |
| `contrastive_patience` | 30 | Steps without improvement before early stopping. |

### Advanced hyperparameters

| Key | Default | Meaning |
|---|---|---|
| `threshold_early_stop_accumulation` | 0.999 | Fraction of accumulated images at which accumulation stops early. |
| `maximal_images_per_animal` | 3000 | Maximum training images per animal in each accumulation step. |
| `device` | `''` (automatic) | Torch device, for example `'cuda'` or `'cpu'`. |
| `torch_compile` | false | Compile models with `torch.compile`. |

### Not real settings

- Removed keys: `add_time_column_to_csv`, `convert_trajectories_to_csv_and_json`, `protocol3_action`, `threshold_acceptable_accumulation`, `maximum_number_of_parachute_accumulations`, `max_ratio_of_pretrained_images`, `identity_transfer` and a few others. The CLI lists them as removed, so take them out of old files.
- `confparams.py` contains more internal constants (crossing-detector and accumulation learning rates, batch sizes and so on). The loader accepts any name in that class, but the docs do not cover them.
