# idtrackerai-detectron2

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/fork%20of-idtracker.ai-lightgrey)](https://gitlab.com/polavieja_lab/idtrackerai)
[![eLife](https://img.shields.io/badge/cite-10.7554%2FeLife.107602-blue)](https://doi.org/10.7554/eLife.107602)

A fork of [idtracker.ai](https://idtracker.ai) that takes animal outlines from
an external instance-segmentation model — a fine-tuned Detectron2 Mask R-CNN, or
Meta's SAM 3 prompted with a word — instead of finding them by intensity
thresholding.

**Identity tracking is untouched.** Crossing detection, fragmentation, the
identification network and gap closing are upstream idtracker.ai's, unmodified.
Only the step that turns pixels into blobs is replaced.

> **New to idtracker.ai?** Start with the **[User Guide](USER_GUIDE.md)** —
> what this does, how to install it, and a first tracking run.

> [!IMPORTANT]
> This is an **unofficial derivative**. It is not affiliated with, endorsed by,
> or supported by the de Polavieja Lab or the Champalimaud Foundation. Please do
> not send problems with this fork to the upstream project — open an issue
> [here](https://github.com/mbel-tech/idtrackerai-detectron2/issues) instead.
>
> For the real thing, go to
> [gitlab.com/polavieja_lab/idtrackerai](https://gitlab.com/polavieja_lab/idtrackerai).

## Why

idtracker.ai segments by thresholding each frame. On low-contrast footage —
unevenly lit tanks, turbid water, fish the same brightness as the background —
that is the limiting step, and no amount of tracking-parameter tuning fixes it.

The usual workaround is to run a segmentation model first, paint its masks back
into the video (darkening the background, brightening the animals), re-encode,
and let idtracker.ai threshold *that*. It works, but it is a round trip through
pixels: precise instance masks are thrown away and approximately recovered, and
every clip needs its own intensity and area thresholds to undo the damage.

This fork removes the round trip. Contours go from the detector into
idtracker.ai's `Blob` objects directly. There is nothing to re-threshold, so
there is nothing to tune per clip.

## How it works

`process_frame()` is the only place in idtracker.ai where pixels become
contours, so it is the only place that changes. Given a contour file it reads
that frame's outlines instead of thresholding, and returns the frame unchanged
so identification images still come from the video's own pixels.

Everything downstream is unaffected, because `Blob` derives its centroid, area,
bounding box, extension and orientation *from the contour*. Supply a contour and
every blob feature exists, computed by the same code as before.

```
Detectron2 Mask R-CNN                          >  contours.h5  ->  idtracker.ai
SAM 3 ("fish")         /
                                             |
                                             +- crossing detection   unmodified
                                             +- fragmentation        unmodified
                                             +- identification CNN   unmodified
                                             +- gap closing          unmodified
```

When two animals genuinely overlap, the default (`--on-overlap merge`) hands
idtracker.ai a single blob and lets its crossing machinery reconstruct who was
who — the behaviour you already trust. The alternative, letting the detector
split them, is available but shifts that judgement to Mask R-CNN.

## Which model

Two can produce the contours, and they ask for opposite things.

**Detectron2** is fine-tuned on frames you outlined yourself. It costs a day of
annotation and a training run, and in exchange it knows your species, your tank
and your lighting. This is the one to use for results you will publish.

**SAM 3** is prompted with a word such as `fish` and segments immediately, with
no annotation and no training. It is a general model that has never seen your
setup, so it is weakest exactly where thresholding is weak — low contrast, small
animals that look alike. It earns its place in two ways: getting a clip tracked
today, or for a recording you will never train a model for; and **drafting the
annotations** for the Detectron2 route, which turns the slow part of preparing a
dataset from drawing into correcting.

They are not exclusive. Drafting with SAM 3 and then training Detectron2 on the
corrected result is the fastest way to a model that is actually good on your
footage.

## Using it

Install as you would upstream idtracker.ai, from this repository:

```bash
pip install git+https://github.com/mbel-tech/idtrackerai-detectron2
```

The import name is unchanged (`import idtrackerai`), so it cannot be installed
alongside upstream idtracker.ai in the same environment.

### Preparing a model

Open a video in the Segmentation App and set **Segmentation → Detectron2
pipeline**. A guided panel walks through the four local stages:

1. **Enhancement** — CLAHE and illumination correction, previewed live on the
   frame you are looking at, with a raw↔enhanced slider to compare. Enhancement
   belongs to a recording setup, not to the software, so this is where you judge
   it against your own footage and save it as that setup's profile.
2. **Sample frames** — stratified across the whole of every clip, so the model
   sees each recording's full range rather than a clump of it. The panel keeps
   its own list of videos to draw from, separate from the one being tracked, so
   a model can be trained across every recording from a setup while each is
   still tracked on its own. It shows the per-video breakdown before it runs.
3. **Annotate** — opens LabelMe on the folder, preloaded with your class name.
4. **Build dataset** — validates every polygon and converts to COCO, splitting
   train and validation **by recording** so the score is not measuring
   memorisation of near-duplicate frames. Clips saved as pieces of one recording
   are folded together; the grouping it inferred is shown before you build, and
   can be overridden.

Steps 5 and 6, training and contour export, need a CUDA GPU. The panel checks
whether this machine has one and runs training here if it does; otherwise it
says what is missing and points at the Colab notebook. The export is handed
over as a command either way, because it takes days over a collection.

PyTorch and Detectron2 are **not** installed with the package, deliberately.
PyPI's PyTorch wheel for Windows is CPU-only, so requiring it would quietly
stop idtracker.ai's own identity tracking using your GPU; and Detectron2 is
not on PyPI at all, so requiring it would make installing this software fail
on any machine without a C++ compiler. Instead:

```bash
idtrackerai_d2_install_gpu
```

works out what your machine needs, shows you the commands, and runs them only
if you agree. Step 5 offers the same thing as a button.

You can close the app between steps; each one records what it produced, and
status is re-derived from disk when you come back.

Training and inference need a GPU, so they run in Colab. The notebook ships
with the package; `idtrackerai_d2_bundle` packages it together with the GPU
scripts, ready to upload.

Every stage is also available as an installed command — `idtrackerai_d2_sample`,
`idtrackerai_d2_dataset`, `idtrackerai_d2_train` and so on — for scripting or a
headless machine. See **[docs/detectron2-pipeline.md](docs/detectron2-pipeline.md)**.

### Segmenting with SAM 3

Set **Segmentation → SAM 3**. The panel needs the checkpoint and a prompt, and
then does either of two things.

**Draft annotations** runs over the sampled frames and writes LabelMe files
beside them, so annotating becomes correcting. Frames you have already annotated
are never overwritten, and frames where SAM 3 found nothing get no file at all,
so they stay visibly unfinished. Everything it draws is a draft: correct it
before training on it.

**Export contours** writes one sidecar per clip, the same format the Detectron2
route produces, and switches the session onto them when it finishes.

Both are also installed commands:

```bash
idtrackerai_sam3_prelabel --input-dir frames/ --weights sam3.pt --prompt fish
idtrackerai_d2_export --backend sam3 --prompt fish     --videos 'clips/*.mp4' --weights sam3.pt --output-dir contours/
```

`sam3.pt` is gated: request access once at
[`facebook/sam3`](https://huggingface.co/facebook/sam3) on Hugging Face. It is
about 3.4 GB and never belongs in the repository.

SAM 3 needs a CUDA GPU. Exporting a whole video on a CPU is not slow but
impractical, so the app offers it only when there is one, and points at the
Colab notebook otherwise — section 8 of the same notebook the Detectron2 stages
use. Drafting a few hundred frames on a CPU is merely slow, and is offered.

See **[docs/sam3-pipeline.md](docs/sam3-pipeline.md)**.

### Tracking with the result

Point a session at the contour file, in the `.toml`:

```toml
external_contours = "contours/clip_01.h5"
```

…or pick it in the Segmentation App under **Segmentation → External contours**.

Or describe the model in the session's `.toml` and let one file drive both
stages. The exporter then runs only the tracked frames, caps detections at the
number of animals, and tracking refuses contour files made any other way:

```toml
number_of_animals = 9
tracking_intervals = [[3600, 36000]]

[segmentation_model]
backend = "detectron2"
weights = "model/model_final.pth"
contours_dir = "contours/pen_01"
```

```bash
idtrackerai_d2_export --session sessions/*.toml
idtrackerai --load sessions/pen_01.toml --track
```

Every key is in **[docs/session-toml.md](docs/session-toml.md)**.
The intensity thresholds and background subtraction grey out, because they take
no part once contours come from a model.

## The Validator

This fork also changes idtracker.ai's Validator, the tool for correcting
trajectories by hand after tracking. It gains undo, an autosave that survives
a lost drive, three new editing operations, filters that keep the error list
workable on long recordings, and a handful of shortcuts. Nothing about tracking
is affected.

What changed and why is in **[VALIDATOR.md](VALIDATOR.md)**.

## Citing

The tracking method is the de Polavieja Lab's work. **Cite their paper**,
whether or not you use this fork:

> Torrents, J., Costa, T., & de Polavieja, G. G. (2026). New idtracker.ai:
> rethinking multi-animal tracking as a representation learning problem to
> increase accuracy and reduce tracking times. *eLife*, 14, RP107602.
> https://doi.org/10.7554/eLife.107602

Depending on what you are reporting, the earlier papers may also be relevant:

> Romero-Ferrero, F., Bergomi, M. G., Hinz, R. C., Heras, F. J. H., &
> de Polavieja, G. G. (2019). idtracker.ai: tracking all individuals in small or
> large collectives of unmarked animals. *Nature Methods*, 16(2), 179–182.
> https://doi.org/10.1038/s41592-018-0295-5

> Pérez-Escudero, A., Vicente-Page, J., Hinz, R. C., Arganda, S., &
> de Polavieja, G. G. (2014). idTracker: tracking individuals in a group by
> automatic identification of unmarked animals. *Nature Methods*, 11(7),
> 743–748. https://doi.org/10.1038/nmeth.2994

If the segmentation stage matters to your results, also cite Detectron2 and
Mask R-CNN:

> Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019).
> Detectron2. https://github.com/facebookresearch/detectron2

> He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. In
> *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
> https://doi.org/10.1109/ICCV.2017.322

If you used the SAM 3 route, for segmentation or to draft annotations, cite it
too:

> Meta AI Research (2025). SAM 3: Segment Anything with Concepts.
> https://github.com/facebookresearch/sam3

[`CITATION.cff`](CITATION.cff) carries all of this in machine-readable form;
GitHub's *Cite this repository* button reads it.

## Licence

GPL-3.0-or-later, the same licence as upstream idtracker.ai. The original
[`LICENSE`](LICENSE) is included verbatim and unchanged.

[`NOTICE`](NOTICE) records the fork point and lists every modification, as
GPL-3.0 section 5(a) requires. The upstream README is preserved as
[`README.idtrackerai.md`](README.idtrackerai.md).

Detectron2 and SAM 3 are optional, installed separately, and not distributed
here. **SAM 3 carries Meta's own SAM License**, which is not an OSI-approved
licence and has its own terms including an acceptable use policy. It governs
both the `sam3` package and the `sam3.pt` weights, it is unaffected by this
project's GPL-3.0, and you should read it before publishing results that depend
on it.

## Acknowledgements

idtracker.ai is built by the de Polavieja Lab at the Champalimaud Foundation.
This fork is a thin change to one stage of their pipeline; the science, and
nearly all of the code, is theirs.

The SAM 3 route began as an independent implementation by
[Talha](https://github.com/laiquet) in
[idtrackerai-enhanced](https://github.com/laiquet/idtrackerai-enhanced), which
integrates SAM 3 and Detectron2 by a different route — running the models during
tracking rather than exporting contours beforehand. No code was copied; the
approach, and the demonstration that it was worth doing, came from there.
