# The SAM 3 route

SAM 3 segments what a short text prompt describes. Give it the word `fish` and
it finds the fish, with no annotation and no training run. This document covers
what that is good for, what it is not, and how to run it.

For the Detectron2 route — training a model on frames you outlined yourself —
see [detectron2-pipeline.md](detectron2-pipeline.md). The two produce the same
contour files and are read by the same code downstream.

## When to use it

SAM 3 has never seen your tank. It is a general model, and it is weakest in
exactly the situation that made thresholding fail in the first place: low
contrast, uneven lighting, small animals that look like one another. Do not
expect it to beat a Mask R-CNN fine-tuned on your own footage, because it will
not.

It earns its place in two ways.

**Getting a clip tracked today.** No annotation, no training, no waiting. Good
for a pilot, for checking whether a recording is usable at all, or for a
recording you will never build a model for.

**Drafting the annotations.** The slow part of the Detectron2 route is outlining
a few hundred animals by hand. SAM 3 finds most of them, so the job becomes
correcting outlines rather than drawing them. This is the use that survives
contact with difficult footage, because you are checking every frame anyway.

The two combine: draft with SAM 3, correct, train Detectron2 on the result.

## Getting the weights

`sam3.pt` is gated. Request access once at
[`facebook/sam3`](https://huggingface.co/facebook/sam3) on Hugging Face and
download it. It is about 3.4 GB.

It never goes in the repository — `.gitignore` refuses `*.pt` for that reason.
Keep it beside your data, or in the Drive project folder at `model/sam3.pt`.

Or let each machine fetch it. **Use Hugging Face** in the app, `weights =
'huggingface'` in the session `.toml`, or `--weights huggingface` on the
command line all download `sam3.pt` once into the Hugging Face cache and use it
from there. Access is still gated, so the machine needs a token from the
account that was granted it, and the `huggingface_hub` package:

```bash
pip install huggingface_hub
hf auth login          # or set HF_TOKEN
```

A refused download says so, and says this, rather than failing with an HTTP
error. The contour files record `sam3.pt` and its hash either way, so a fetched
checkpoint and a downloaded-by-hand one are interchangeable.

> **Licence.** The weights and the `sam3` package carry Meta's own
> **SAM License**, which is not an OSI-approved licence and has its own terms,
> including an acceptable use policy. It is independent of this project's
> GPL-3.0. Read it before publishing results that depend on it.
>
> The `ultralytics` package can load the same weights. It is not used here
> because it is AGPL-3.0.

## Installing the runtime

```bash
pip install sam3
```

It is not declared as a dependency of this fork, for the same reason Detectron2
is not: it is heavy, it wants a GPU, and idtracker.ai has to keep working
without it. Every import of it happens inside the function that needs it.

You also need PyTorch built for your CUDA version, which PyPI's default wheel
is not on Windows. `idtrackerai_d2_install_gpu` installs the right one — it was
written for the Detectron2 stages but PyTorch is the same requirement here, so
run it and then `pip install sam3`.

## Where it runs

SAM 3 wants a CUDA GPU.

| Task | On a CUDA GPU | On a CPU |
| --- | --- | --- |
| Drafting a few hundred frames | minutes | slow but survivable |
| Exporting a whole video | hours | not practical |

PyTorch has no ROCm build for Windows, so an AMD card is not an alternative.
Without CUDA, the Segmentation App offers drafting (behind a confirmation that
says what it will cost) and withholds exporting, pointing at Colab instead.

Section 8 of `colab_detectron2_pipeline.ipynb` runs both stages there. It is the
same bundle the Detectron2 stages use: build it with `idtrackerai_d2_bundle`,
upload it alongside `sam3.pt`, and run the cells.

## Drafting annotations

In the app: **Segmentation → SAM 3 → Draft annotations**, having chosen the
folder of sampled frames.

From a terminal:

```bash
idtrackerai_sam3_prelabel \
    --input-dir frames/ \
    --weights sam3.pt \
    --prompt fish \
    --max-instances 8
```

It writes one LabelMe `.json` beside each frame. Then annotate as usual —
`idtrackerai_d2_dataset` reads them without knowing a model drew them.

Three behaviours worth knowing:

- **It never overwrites an existing annotation.** Once you have corrected a
  frame, that file is the valuable one, and a tool meant to save annotation
  effort must not be able to destroy it. `--overwrite` is there for when
  redrawing is what you meant.
- **A frame it found nothing in gets no file**, rather than an empty one, so it
  still looks unannotated — which it is.
- **The outlines are simplified** so they can be dragged into place by hand. A
  traced mask has hundreds of vertices, which is accurate and unusable. Control
  it with `--simplify`; larger means fewer points.

The frames are already enhanced when sampling writes them, so nothing is
enhanced twice.

Every shape carries a `sam3_prelabel` flag, so machine-drawn outlines stay
distinguishable from hand-drawn ones when you come to describe how the training
set was built.

**They are drafts.** Correct them, and add what was missed, before training.

## Exporting contours

In the app: **Segmentation → SAM 3**, type the prompt (the box starts empty,
and Export stays disabled until it has a word in it), choose the **Contours
folder**, then **Export contours**. When it finishes the session switches onto
the files it wrote, if they fit the video it has loaded.

Without a GPU here, **Copy command** under the export button gives the same
export as a command to run on a machine that has one: the prompt, weights,
confidence, animals per frame and the session's enhancement are all written
out, so it produces the files the button would have.

Like the command line, it skips clips that already have a contour file, so a
run cancelled after three hours is picked up where it stopped rather than
started again. Tick *Re-export clips that already have contours* when the
existing files were made with a prompt or a confidence you no longer want.

From a terminal, it is the Detectron2 exporter with a different backend:

```bash
idtrackerai_d2_export \
    --backend sam3 \
    --prompt fish \
    --videos 'clips/*.mp4' \
    --weights sam3.pt \
    --output-dir contours/ \
    --max-instances 8
```

Everything the Detectron2 route gets, this gets: sequential decoding, the
`--local-cache` for slow Drive mounts, resumable batches that skip finished
clips, atomic writes, and the same `--on-overlap` policy. The resulting `.h5`
records which model made it and what it was prompted with.

Then, per clip:

```toml
external_contours = "contours/clip_01.h5"
```

## Choosing a prompt

Keep it to the animal. `fish` works; `fish in a tank` also matches the tank, the
shadows and the reflections. The prompt is also the class name written into
drafted annotations, so it should be the label you would have typed yourself.

`--score-threshold` trades misses against false positives. Raise it when the
background is being detected; lower it when animals are missed — it is passed
to SAM 3 itself, so lowering it genuinely admits weaker detections rather than
filtering an already-filtered list. On genuinely
hard footage neither setting rescues it, which is the honest limit of a model
that has never seen your setup — and the point at which drafting-then-training
is the better use of it.

`--max-instances` keeps the most confident detections per frame. Setting it to
the number of animals present trims spurious extras, but it cannot invent one
that was missed: short frames are left short on purpose, for idtracker.ai's
crossing detection and interpolation to resolve. They work from the whole video,
which is more than any per-frame fix-up can do.

## Why one frame at a time

SAM 3 can track through a video, carrying identity in a memory bank. That is
deliberately not used.

The contour sidecar stores outlines and nothing else — no identity field — so
SAM 3's track IDs would be computed and then discarded. Prompting each frame
independently keeps the single sequential pass over the video that makes the
exporter fast, and leaves identity where this fork wants it: in idtracker.ai's
crossing detection and fragmentation, which reason over the whole recording
rather than the previous frame.

If masks turn out to break down through occlusions, SAM 3's video mode is the
escalation, and it would mean buffering frames rather than streaming them.
