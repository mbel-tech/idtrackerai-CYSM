# SAM 3 on the public ONNX weights, by default

Date: 2026-11-09
Status: approved in conversation; this document is for review before planning.

## Why

The SAM 3 route runs Meta's PyTorch package (`sam3`, from
`facebookresearch/sam3`) on `sam3.pt`, which lives in the gated
`facebook/sam3` Hugging Face repository. Using it means requesting access,
waiting for approval, and giving every machine a token or a 3.4 GB file copied
by hand.

LabelMe already runs SAM 3 without any of that. It goes through `osam`, which
loads an ONNX conversion of the same model from the public repository
`wkentaro/sam3-onnx-models-v0.6.0` (four files, pinned by SHA-256 in `osam`).
Same architecture, same 1008×1008 input, same text prompt, so the masks are
expected to match the PyTorch ones closely; that is to be confirmed on real
footage (see Testing).

## What changes, in one line

The SAM 3 backend gains an ONNX engine, and it becomes the default; the
PyTorch engine on `sam3.pt` stays available as an option.

## Weights choices

| Choice | `.toml` `weights` / CLI `--weights` | Engine | Needs |
|---|---|---|---|
| ONNX, fetched (**default**) | `"sam3-onnx"`, or left out | ONNX | `osam`; downloads ~3.4 GB once into `~/.cache/osam` |
| ONNX, a folder | path to a **directory** holding the four `sam3_*.onnx` files | ONNX | `osam`; no network |
| `sam3.pt`, fetched | `"huggingface"` (unchanged) | PyTorch | `sam3`, `huggingface_hub`, a token with granted access |
| `sam3.pt`, a file | path to a **file** (unchanged) | PyTorch | `sam3` |

Rules:

- A directory means ONNX; anything else that is not a keyword is taken as a
  `sam3.pt` path, as today. No new setting chooses the engine.
- `sam3-onnx` is case-insensitive, like `huggingface` and `model_zoo`, and
  only applies to `backend = "sam3"`; anywhere else it is an error, worded like
  the existing ones.
- `backend = "sam3"` with no `weights` normalises to `sam3-onnx`. Detectron2
  still requires `weights`; its error message is unchanged.
- Existing `.toml` files and commands that name `sam3.pt` or `huggingface`
  behave exactly as before.

The folder is what `hf download wkentaro/sam3-onnx-models-v0.6.0 --local-dir
<folder>` produces. It may hold other files (LICENSE, README); only the four
`.onnx` names osam pins are read.

## Components

### `sam3_onnx.py` (new, beside `sam3_predictor.py`)

Imports `osam` only inside functions, like every other optional backend, so
idtracker.ai and the Segmentation App start without it. It is added to the
Colab bundle next to `sam3_predictor.py`.

- `SAM3_ONNX = "sam3-onnx"` lives in `segmentation_model.py` with the other
  keywords; `sam3_onnx.py` uses it from there.
- `onnx_blobs()` — the `{key: Blob}` table of osam's `Sam3` model. The single
  source of truth for file names, URLs and hashes: nothing is copied into
  this repository, so an osam release that moves to new weights is followed
  automatically, and recorded as different weights (see Provenance).
- `prepare_folder(folder)` — for each pinned blob, find `<folder>/<filename>`;
  refuse with an error naming every missing file, or every file whose SHA-256
  differs from the pin (giving both hashes). When all match, place each in
  osam's cache path (`blob.path`) unless an identical file is already there:
  a hard link when on the same filesystem, else a copy. After this, osam's own
  `pull()` finds everything present and downloads nothing.
- `weights_identity()` — `(label, sha256)` for provenance: the label is the
  Hugging Face repository name taken from the blob URLs
  (`sam3-onnx-models-v0.6.0`); the hash is SHA-256 over the four pinned
  hashes joined in sorted key order. Computed from the pins, never by
  re-reading 3.4 GB.
- `Sam3OnnxPredictor(weights, prompt, device, score_threshold, max_instances)`
  — same interface and attributes as `Sam3Predictor`
  (`predict(frame) -> (masks, scores)`, `description`). On construction:
  `prepare_folder` when given a folder, then `osam`'s `Sam3(device=...)`, which
  pulls whatever is missing (with a message saying so first, and the size).
  `predict`:
  1. BGR to RGB (osam wants RGB; getting it wrong silently segments worse).
  2. `model.generate(GenerateRequest(model="sam3", image=..., prompt=Prompt(
     texts=[prompt], score_threshold=score_threshold,
     max_annotations=max_instances or None)))`.
  3. Each annotation's mask covers only its bounding box; paste it into a
     zero frame at the box position to get the full-frame 0/1 `uint8` mask the
     pipeline expects.
  4. Sort by score, drop below threshold, cap at `max_instances` (the
     same contract `Sam3Predictor` enforces, kept here rather than trusted to
     osam).
  `description` is `sam3-onnx ('<prompt>')`.

### `make_sam3_predictor(weights, **kwargs)` (in `sam3_predictor.py`)

Chooses the engine: `sam3-onnx` or a directory → `Sam3OnnxPredictor`;
`huggingface` or a file → `Sam3Predictor` (unchanged). Every place that builds
a SAM 3 predictor today calls this instead: `inference.py` (exporter),
`prelabel.py` (drafting), and both threads in
`segmentation_app/widgets/pipeline_threads.py`.

### Device

- ONNX: `--device cuda` asks osam for CUDA; when onnxruntime has no CUDA
  provider, fall back to CPU with the same message `resolve_device` prints
  today. `--device cpu` forces CPU.
- The app's capability check (`GpuReport.sam3_usable`, used by
  `Sam3Panel.has_gpu`) takes the chosen weights into account: for ONNX it means
  `osam` importable and `"CUDAExecutionProvider"` in
  `onnxruntime.get_available_providers()`; for `sam3.pt` it stays torch CUDA +
  `sam3`. The panel's behaviour on a CPU is unchanged: drafting offered behind
  its confirmation, whole-video export withheld with the Colab hint.

### Provenance and the `.toml` mismatch check

- The exporter records, for ONNX: `weights = "sam3-onnx-models-v0.6.0"` and
  `weights_sha256 = weights_identity()[1]`, whether the files came from the
  download or a folder (they are the same bytes, verified).
- `local_weights()` and `weights_sha256()` in `inference.py` learn the ONNX
  case and stop treating it as a file to hash.
- `segmentation_model.weights_label()` returns the ONNX label for
  `sam3-onnx` and for a directory, so `mismatches()` treats the two as the same
  weights and either as different from `sam3.pt`.

### CLI

- `idtrackerai_d2_export --backend sam3`: `--weights` becomes optional and
  defaults to `sam3-onnx`; still required for Detectron2.
- `idtrackerai_sam3_prelabel`: `--weights` becomes optional, default
  `sam3-onnx`; help text lists all four forms.

### Segmentation App (SAM 3 panel)

- Opens with the weights set to `sam3-onnx`, labelled
  "SAM 3 ONNX (public weights, no token), fetched at the first run".
- New **Choose ONNX folder** button beside **Choose sam3.pt** and
  **Use Hugging Face**, and a way back to the default (a **Use public ONNX**
  button). Tooltips in `tooltips.toml` for each.
- The generated export command and the saved session `.toml` carry the chosen
  form (`sam3-onnx`, the folder, `huggingface`, or the `.pt` path).
- The "not installed" message names `osam` for ONNX, `sam3` for `.pt`.

### Packaging

- New optional extra `sam3 = ['osam >= 0.10, < 0.11']`. 0.10.0 is the first
  release on `sam3-onnx-models-v0.6.0` (0.9.x used v0.5.0); 0.10.0–0.10.2
  pin identical hashes. The upper bound keeps a new weights version from
  arriving unannounced; raising it is a deliberate change that will also
  change the recorded identity.
- `osam` depends on CPU `onnxruntime`. GPU needs `onnxruntime-gpu` instead,
  which the docs and the notebook install explicitly.

### Colab notebook, section 8

Built on top of the token change in PR #15 (not yet merged), on the same
branch.

- Default path: install `osam` and swap `onnxruntime` for `onnxruntime-gpu`;
  `SAM3_WEIGHTS = MODEL / 'sam3-onnx'` on Drive; if that folder lacks the
  files, `hf download wkentaro/sam3-onnx-models-v0.6.0 --local-dir` it once
  (public, no token). Later sessions read it from Drive.
- Optional path, kept: `sam3.pt` uploaded to `MODEL`, or fetched with the
  `HF_TOKEN` secret, as #15 does. One variable chooses between them.
- Commands in 8a/8b pass `{SAM3_WEIGHTS}` as today, so they work for either.

### Docs

README, `docs/sam3-pipeline.md`, `docs/sam3-session-template.toml` and
`docs/sam3-toml-handoff.md` lead with the no-token route and describe
`sam3.pt` as the alternative; the licence note stays (the ONNX weights are a
conversion of Meta's and carry the SAM License too). `docs/source/user_guide/changelog.rst`
gets an entry.

## Errors

| Situation | Message says |
|---|---|
| `osam` missing, ONNX chosen | `pip install osam` (or the `[sam3]` extra) |
| `sam3` missing, `.pt` chosen | unchanged: `pip install sam3` |
| folder lacks files | the folder, and each missing file name; how to fill it with `hf download` |
| file hash differs | the file, expected and actual SHA-256; likely a different weights version |
| download fails | osam's error, plus: no token needed, check network/proxy, or use a folder |
| `sam3-onnx` with Detectron2 | `weights = "sam3-onnx" only applies to backend = "sam3"` |

## Testing

All with a stand-in `osam` module (the pattern the existing tests use for
`huggingface_hub`), so no download and no GPU:

- `make_sam3_predictor` picks the engine for each of the four forms.
- `normalize_model` defaults `weights` to `sam3-onnx` for SAM 3, keeps it
  required for Detectron2, and rejects `sam3-onnx` there.
- `prepare_folder`: all present → placed in cache, no download; one missing →
  error naming it; one wrong → error with both hashes; identical file already
  cached → left alone.
- `predict`: BGR→RGB reaches osam; box-cropped masks land at the right place
  in a full frame; threshold and cap; empty result.
- Provenance: label and hash for download and for folder are equal; differ
  from `sam3.pt`; `mismatches()` agrees.
- CLI: `--backend sam3` without `--weights` works; Detectron2 without it still
  errors.
- Panel: opens on ONNX; folder and back; command and `.toml` carry the form;
  `has_gpu` follows the engine.
- Bundle test: `sam3_onnx.py` is in the bundle.

Not possible in this sandbox (huggingface.co is blocked): real inference.
A command, `idtrackerai_sam3_compare` (module
`extra_tools/detectron2_pipeline/sam3_compare.py`, entry point in
`pyproject.toml`), runs both engines on one frame
and reports per-animal IoU and score, for the user to run where the weights
are reachable. The engines are expected to agree closely; if they do not, the
default is revisited before merging.

## Out of scope

- Running `facebookresearch/sam3`'s code on ONNX, or converting `sam3.pt` to
  ONNX ourselves.
- SAM 3's video (tracking) mode; per-frame stays, for the reasons in
  `sam3_predictor.py`.
- Moving osam's cache location; on Colab the folder form covers persistence.
