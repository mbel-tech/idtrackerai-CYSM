# Handoff: a SAM 3 session `.toml`

For whoever runs a session segmented by SAM 3: what goes in the file, where
each stage runs, and what to check. The full key reference is in
[session-toml.md](session-toml.md); why SAM 3 and when not to use it is in
[sam3-pipeline.md](sam3-pipeline.md).

Start from [sam3-session-template.toml](sam3-session-template.toml).

## The route in one picture

```
session.toml ──► 1. export  (GPU machine or Colab)  ──► contours_dir/<clip>.h5
     │                                                          │
     └──────────► 2. track   (any machine)  ◄───────────────────┘
```

One file drives both stages. The exporter runs SAM 3 with the prompt and writes
one contour file per clip. idtracker.ai reads them instead of thresholding,
after checking they were made with this prompt, these weights and these
settings.

## Filling in the file

| Key | What to put | Why it matters |
|---|---|---|
| `video_paths` | Every clip of the session, in recording order. | Read as one continuous recording. |
| `number_of_animals` | The count in the arena. | Also the per-frame detection cap unless `max_instances` says otherwise. Correcting it later forces a re-export. |
| `tracking_intervals` | The frames to track, or leave it out for all. | The exporter runs SAM 3 only on these frames. |
| `[segmentation_model] backend` | `'sam3'` | |
| `prompt` | **The animal in your videos**, one plain word or two: `'chicken'`, `'fish'`, `'mouse'`. | The template leaves it empty, and nothing runs until it is filled in. Keep it to the animal: `'fish in a tank'` also matches the tank. |
| `weights` | `'huggingface'`, or the path of a `sam3.pt`. | `'huggingface'` keeps the file portable: each machine fetches the checkpoint itself. |
| `contours_dir` | A folder for this session only. | Tracking finds the files there. Clips from different cameras can share a file name. |
| `score_threshold` | `0.5` to start. | Raise it when background is detected; lower it when animals are missed. |

Leave `max_instances`, `on_overlap` and the clean-up keys unset unless you have a
reason; their defaults are the exporter's.

Or skip the typing: in the Segmentation App choose **Segmentation → SAM 3**, fill
in the panel, and **Save parameters** writes the same table.

## Before the export: access to the weights

With `weights = 'huggingface'`, on the machine that exports:

```bash
pip install sam3 huggingface_hub
hf auth login                      # or: export HF_TOKEN=...
```

The account behind the token must have been granted access at
<https://huggingface.co/facebook/sam3> (request it once; it is gated). The first
run downloads about 3.4 GB into the Hugging Face cache; later runs reuse it. A
refused download stops with this same advice rather than an HTTP error.

With a path instead, copy `sam3.pt` to that path on the exporting machine.
Tracking does not need the weights at all: without them it checks contour files
by the recorded file name `sam3.pt` instead of by hash.

## 1. Export, where the GPU is

```bash
idtrackerai_d2_export --session session.toml
```

- Runs only the frames in `tracking_intervals`, writes `<contours_dir>/<clip>.h5`.
- Resumable: stop it and run it again, and it skips clips already done. A clip
  made with another prompt, threshold or frame range is redone.
- Several sessions at once: `--session 'sessions/*.toml'`.
- Colab: section 8 of `colab_detectron2_pipeline.ipynb`, with the bundle from
  `idtrackerai_d2_bundle` (`sam3.pt` alongside it only if `weights` is a path).

> **Enhancement.** `--session` exports SAM 3 with the built-in enhancement, not
> the session's `enhancement` key. If the session sets one, pass the same
> settings as flags (the SAM 3 panel's **Copy command** includes them), or
> contours exported from the app will look out of date and be redone.

From the app on a machine with a CUDA GPU, **Export contours** does the same
and switches the session onto the files when it finishes.

## 2. Track

```bash
idtrackerai --load session.toml --track
```

Before segmenting, tracking refuses a contour file that:

- does not cover the tracking intervals,
- was made with another backend, prompt, weights, score threshold or overlap
  policy,
- was capped at a different number of animals than this session needs.

The error names the file and what differs. `idtrackerai_d2_export --session` on
the same file redoes exactly those.

## Checklist

- [ ] `prompt` names the animal in these videos.
- [ ] `contours_dir` is this session's own folder.
- [ ] `number_of_animals` is right; it is also the detection cap.
- [ ] The exporting machine has `sam3`, a CUDA GPU, and either `sam3.pt` at
      `weights` or a Hugging Face token with access.
- [ ] Spot-check a few frames in the Segmentation App (**External contours**)
      before a long tracking run. SAM 3 has never seen this setup.
