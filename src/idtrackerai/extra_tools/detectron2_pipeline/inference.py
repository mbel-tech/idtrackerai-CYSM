"""Runs an instance-segmentation model over a video and writes the animals'
outlines to a sidecar file for idtracker.ai.

This replaces the older "enhance the video, then let idtracker.ai re-threshold
it" route. Nothing is painted back into pixels, so no per-clip intensity or area
threshold has to be tuned: the contours the model produces are the contours
idtracker.ai tracks.

Runs where the GPU is (Colab, a workstation) and produces one small .h5 per
video. Tracking then happens wherever idtracker.ai lives; the two no longer have
to be the same machine, and an enhanced video never has to be encoded or moved.

Two models can do the segmenting, and they ask for opposite things:

``--backend detectron2`` (default)
    A Mask R-CNN fine-tuned on frames you annotated yourself. Costs a day of
    annotation and a training run, and in exchange knows your species, your
    tank and your lighting.

``--backend sam3``
    Meta's SAM 3, prompted with a word such as ``fish``. No annotation and no
    training, but it is a general model that has never seen your setup, so it
    is weaker exactly where thresholding is weak: low contrast and small,
    similar animals. Also useful as a first pass to pre-label frames you then
    correct and train on — see ``prelabel.py``.

Whichever runs, the output is the same file in the same format, and everything
after the masks — cleanup, overlap policy, polygons, the sidecar — is shared.

Usage
-----
    python detectron2_export_contours.py \\
        --video clip_01.mp4 \\
        --weights model_final.pth \\
        --output clip_01_contours.h5 \\
        --max-instances 5

or, with no trained model of your own::

    python detectron2_export_contours.py \\
        --backend sam3 --prompt fish \\
        --video clip_01.mp4 \\
        --weights sam3.pt \\
        --output clip_01_contours.h5 \\
        --max-instances 5

Then in the idtracker.ai .toml for that clip::

    external_contours = "clip_01_contours.h5"

Overlap policy
--------------
``--on-overlap`` decides what happens when the model reports two animals whose
masks intersect:

``merge`` (default)
    Strongly overlapping masks become one contour. idtracker.ai sees a single
    blob, its crossing detector flags it, and the usual fragmentation and
    interpolation reconstruct who was who. Use this to keep identity
    reconstruction in idtracker.ai's hands.

``split``
    Contested pixels go to the higher-scoring instance and each animal is
    emitted separately. The detector resolves the occlusion instead. Fewer
    crossings, but mask assignment can flicker between frames, and that
    instability propagates into fragmentation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from . import preprocessing as fp
    from . import segmentation_model as sm
except ImportError:  # loaded by path, e.g. from a Colab bundle
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import preprocessing as fp  # type: ignore[no-redef]
    import segmentation_model as sm  # type: ignore[no-redef]


class ExportError(RuntimeError):
    """A video could not be exported; the batch goes on with the next one."""


# More unreadable frames than this fraction of a video means the file is
# damaged or truncated, and the contours would silently have holes in them.
MAX_READ_FAILURE_FRACTION = 0.01


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_sha256(weights) -> str:
    """Hash of the weights file; for model_zoo, a stable label instead."""
    if str(weights) == "model_zoo":
        return "model_zoo"
    return file_sha256(Path(weights))


def local_weights(backend: str, weights):
    """The weights as a file on this machine, for hashing and loading.

    Only SAM 3 has anything to resolve: ``huggingface`` becomes the cached
    sam3.pt, downloaded first if need be. Done before the hash, so a contour
    file records the checkpoint it was really made with.
    """
    if backend != "sam3":
        return weights
    try:
        from .sam3_predictor import resolve_checkpoint
    except ImportError:  # loaded by path, e.g. from a Colab bundle
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sam3_predictor import resolve_checkpoint  # type: ignore[no-redef]
    return resolve_checkpoint(weights)


def video_frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def stale_reason(output: Path, video: Path, args, enhancement: dict) -> str | None:
    """Why an existing contour file cannot be reused, or None if it is current.

    Skipping on mere existence would keep contours made with other weights, a
    different enhancement or threshold, or from a truncated earlier run.
    """
    import h5py

    try:
        with h5py.File(output, "r") as file:
            attrs = dict(file.attrs)
    except OSError as exc:
        return f"unreadable ({exc})"

    def differs(key, expected) -> bool:
        if key not in attrs:
            return True
        recorded = attrs[key]
        if isinstance(expected, float):
            return abs(float(recorded) - expected) > 1e-9
        return recorded != expected

    postprocessing = {
        "min_component": args.min_component,
        "dilate": args.dilate,
        "dedup_iou": args.dedup_iou,
        "merge_overlap": args.merge_overlap,
    }
    checks = {
        "weights_sha256": args.weights_sha256,
        "score_threshold": float(args.score_threshold),
        "max_instances": args.max_instances,
        "on_overlap": args.on_overlap,
        "n_frames": (
            min(video_frame_count(video), args.limit)
            if args.limit
            else video_frame_count(video)
        ),
    }
    for key, expected in checks.items():
        if differs(key, expected):
            return f"{key} differs (file: {attrs.get(key)!r}, now: {expected!r})"
    backend = getattr(args, "backend", "detectron2")
    if attrs.get("backend", "detectron2") != backend:
        return f"backend differs (file: {attrs.get('backend')!r}, now: {backend!r})"
    if backend == "sam3" and attrs.get("prompt") != args.prompt:
        return f"prompt differs (file: {attrs.get('prompt')!r}, now: {args.prompt!r})"
    try:
        recorded_classes = json.loads(attrs.get("classes", "null"))
    except (TypeError, ValueError):
        return "classes record is unreadable"
    if recorded_classes != getattr(args, "classes", None):
        return f"classes differ (file: {recorded_classes}, now: {getattr(args, 'classes', None)})"
    # A file exported over part of the video can be reused only for frames it
    # holds; one exported over all of it, for any part.
    needed = getattr(args, "intervals", None)
    try:
        exported = sm.exported_intervals_of(attrs)
    except (TypeError, ValueError):
        return "exported_intervals record is unreadable"
    if needed is None:
        if exported is not None:
            return "it holds only part of the video, the whole video is needed now"
    elif not sm.covers(exported, needed):
        return f"it holds frames {exported}, frames {needed} are needed now"
    try:
        if json.loads(attrs.get("enhancement", "null")) != enhancement:
            return "enhancement differs"
        if json.loads(attrs.get("postprocessing", "null")) != postprocessing:
            return "postprocessing differs"
    except (TypeError, ValueError):
        return "enhancement or postprocessing record is unreadable"
    return None


def load_training_metadata(weights: Path) -> dict | None:
    """Reads training_metadata.json written beside the weights, if present."""
    path = Path(weights).parent / "training_metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_writer():
    """Imports write_contours from idtrackerai, or straight from the repo.

    On a Colab runtime idtracker.ai is usually not installed, but the file only
    needs numpy and h5py, so loading it by path works fine.
    """
    try:
        from idtrackerai.base.animals_detection.external_contours import (  # type: ignore
            write_contours,
        )

        return write_contours
    except ImportError:
        pass

    # Colab bundle layout: this file is unpacked at <root>/tools/ and the
    # writer keeps its repository path under <root>/src/.
    candidate = (
        Path(__file__).resolve().parent.parent
        / "src/idtrackerai/base/animals_detection/external_contours.py"
    )
    if not candidate.is_file():
        raise ImportError(
            "Could not find external_contours.py. Either install this "
            "idtracker.ai fork, or keep this script inside the repo's tools/ "
            "folder."
        )
    spec = importlib.util.spec_from_file_location("external_contours", candidate)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["external_contours"] = module
    spec.loader.exec_module(module)  # type: ignore
    return module.write_contours


# --------------------------------------------------------------- mask cleanup
def clean_mask(
    mask: np.ndarray, min_component: int, kernel: np.ndarray, dilate: int
) -> np.ndarray:
    """Drops specks, closes pinholes, smooths the boundary, then grows slightly.

    Mirrors the post-processing that was already in use: small connected
    components removed, morphological close then open with a 3x3 ellipse, and a
    1-pixel dilation so the whole animal is inside the outline.
    """
    mask = mask.astype(np.uint8)

    if min_component > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels > 1:
            keep = np.zeros(n_labels, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component
            mask = keep[labels].astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if dilate > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    return mask


def deduplicate(masks: list[np.ndarray], scores: list[float], iou_th: float):
    """Removes near-duplicate detections, keeping the more confident one."""
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for i in order:
        duplicate = False
        for j in kept:
            intersection = np.logical_and(masks[i], masks[j]).sum()
            if not intersection:
                continue
            union = np.logical_or(masks[i], masks[j]).sum()
            if intersection / union >= iou_th:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [masks[i] for i in kept], [scores[i] for i in kept]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The biggest 8-connected piece of a binary mask."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def resolve_overlaps(masks: list[np.ndarray], scores: list[float], policy: str,
                     merge_th: float):
    """Applies the chosen overlap policy, returning the masks to be emitted."""
    if len(masks) < 2 or policy == "keep":
        return masks, scores

    if policy == "split":
        # Highest score claims contested pixels; every animal stays separate.
        order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
        claimed = np.zeros_like(masks[0], bool)
        out_masks, out_scores = [], []
        for i in order:
            exclusive = np.logical_and(masks[i], ~claimed)
            if exclusive.any():
                claimed |= exclusive.astype(bool)
                # removing the claimed pixels can cut one animal into pieces,
                # and each piece would become its own blob
                exclusive = largest_component(exclusive.astype(np.uint8))
                out_masks.append(exclusive)
                out_scores.append(scores[i])
        return out_masks, out_scores

    # policy == "merge": fuse masks that genuinely overlap into one blob and let
    # idtracker.ai's crossing machinery take it from there.
    groups = list(range(len(masks)))

    def root(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            intersection = np.logical_and(masks[i], masks[j]).sum()
            if not intersection:
                continue
            smaller = min(masks[i].sum(), masks[j].sum())
            if smaller and intersection / smaller >= merge_th:
                groups[root(i)] = root(j)

    fused: dict[int, np.ndarray] = {}
    fused_scores: dict[int, float] = {}
    for i, mask in enumerate(masks):
        r = root(i)
        if r in fused:
            fused[r] = np.logical_or(fused[r], mask).astype(np.uint8)
            fused_scores[r] = min(fused_scores[r], scores[i])
        else:
            fused[r] = mask
            fused_scores[r] = scores[i]
    return list(fused.values()), list(fused_scores.values())


def mask_to_contours(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Outer outlines of a binary mask, as [n_points, 2] int32 arrays."""
    found, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    return [
        c[:, 0].astype(np.int32)
        for c in found
        if len(c) >= 3 and cv2.contourArea(c) >= min_area
    ]


# ------------------------------------------------------------------ backends
# A backend is anything with a ``predict`` method taking one BGR frame and
# returning its instance masks and their scores:
#
#     predict(frame) -> (list[np.ndarray uint8 H*W], list[float])
#
# Everything after that point in this file — mask cleanup, deduplication, the
# overlap policy, the polygons and the sidecar itself — is the same whichever
# model produced the masks, and the sidecar format has never mentioned
# Detectron2. Keeping the model behind this one small interface is what lets a
# second one be added without touching the export loop.


class Detectron2Predictor:
    """A fine-tuned Mask R-CNN, trained on frames you annotated yourself."""

    def __init__(self, args, metadata: dict | None):
        from detectron2 import model_zoo  # type: ignore
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultPredictor  # type: ignore

        # Class count and test scale must match training. Take them from the
        # training record unless the user overrode them explicitly, because a
        # mismatch here produces a model that loads cleanly and predicts nonsense.
        model_zoo_weights = str(args.weights) == "model_zoo"
        num_classes = args.num_classes
        if num_classes is None:
            # the COCO checkpoint has COCO's 80 classes, whatever was trained here
            num_classes = 80 if model_zoo_weights else (metadata or {}).get("num_classes", 1)
        min_size = args.min_size
        if min_size is None:
            min_size = (metadata or {}).get("min_size_test", 640)
        config = args.config or (metadata or {}).get(
            "config", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        print(f"Model: {config}, {num_classes} class(es), test scale {min_size}")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))
        if model_zoo_weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
            print(f"Loading the COCO checkpoint of {config} (not fine-tuned)")
        else:
            print(f"Loading weights {args.weights} (unpickled by torch: trusted files only)")
            cfg.MODEL.WEIGHTS = str(args.weights)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = args.max_instances
        cfg.INPUT.MIN_SIZE_TEST = min_size
        cfg.MODEL.DEVICE = args.device
        self.classes = getattr(args, "classes", None)
        if self.classes:
            # Filtering happens after detection, so other classes must not use
            # up the cap: let the model return more, and cap again afterwards.
            cfg.TEST.DETECTIONS_PER_IMAGE = max(100, args.max_instances)
        self.max_instances = args.max_instances
        self._predictor = DefaultPredictor(cfg)
        self.description = config

    def predict(self, frame: np.ndarray):
        instances = self._predictor(frame)["instances"].to("cpu")
        if self.classes:
            keep = np.isin(instances.pred_classes.numpy(), self.classes)
            # instances come sorted by score, so the first ones are the best
            import torch  # type: ignore

            index = np.flatnonzero(keep)[: self.max_instances]
            instances = instances[torch.as_tensor(index, dtype=torch.long)]
        masks = list(instances.pred_masks.numpy().astype(np.uint8))
        scores = [float(s) for s in instances.scores.numpy()]
        return masks, scores


def build_predictor(args, metadata: dict | None):
    """Returns the backend named by ``--backend`` and a string describing it.

    Security: ``--weights`` is loaded by torch, which unpickles it, and
    unpickling can execute arbitrary code. Only use weights you trained
    yourself or obtained from a source you trust.

    The description is recorded in the sidecar so a contour file says which
    model made it, which matters once two of them can.
    """
    if args.backend == "sam3":
        try:
            from .sam3_predictor import Sam3Predictor
        except ImportError:  # loaded by path, e.g. from a Colab bundle
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from sam3_predictor import Sam3Predictor  # type: ignore[no-redef]

        predictor = Sam3Predictor(
            checkpoint=args.weights,
            prompt=args.prompt,
            device=args.device,
            score_threshold=args.score_threshold,
            max_instances=args.max_instances,
        )
    else:
        predictor = Detectron2Predictor(args, metadata)
    return predictor, predictor.description


def contours_of_frame(predictor, frame: np.ndarray, args, kernel):
    """The model's animals in one frame, cleaned up, as contours and scores."""
    masks, scores = predictor.predict(frame)

    masks = [clean_mask(m, args.min_component, kernel, args.dilate) for m in masks]
    keep = [i for i, m in enumerate(masks) if m.any()]
    masks = [masks[i] for i in keep]
    scores = [scores[i] for i in keep]

    if args.dedup_iou < 1.0 and len(masks) > 1:
        masks, scores = deduplicate(masks, scores, args.dedup_iou)

    masks, scores = resolve_overlaps(masks, scores, args.on_overlap, args.merge_overlap)

    frame_contours: list[np.ndarray] = []
    frame_scores: list[float] = []
    for mask, score in zip(masks, scores):
        for contour in mask_to_contours(mask, args.min_area):
            frame_contours.append(contour)
            frame_scores.append(score)
    return frame_contours, frame_scores


def seek(cap, frame_index: int) -> None:
    """Moves the capture so the next read() returns ``frame_index``.

    The same cv2 seek idtracker.ai's own segmentation relies on, checked, and
    falling back to decoding from the start when the container will not seek
    exactly, because an off-by-some seek would shift every contour in time.
    """
    if cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) and int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    ) == frame_index:
        return
    print(f"  seek to frame {frame_index} was inexact; decoding up to it", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame_index):
        if not cap.grab():
            break


def export_video(
    video: Path,
    output: Path,
    args,
    predictor,
    model_description: str,
    enhance,
    enhancement: dict,
    write_contours,
    progress=None,
    abort=None,
) -> dict | None:
    """Runs the model over one video and writes its contour file.

    Returns a stats dict, which the batch runner records so a later session can
    report on work done in an earlier one.

    ``progress(done, total)`` is called per frame, for a caller with a progress
    bar to fill. ``abort()`` is polled per frame, following the convention in
    ``sampling.py``; when it returns true this gives up and returns ``None``
    **without writing anything**. A half-exported clip must not leave a file
    behind, because the resume logic counts an existing file as a finished one
    and would skip the rest of the video for good.
    """
    started = datetime.now(timezone.utc)
    local = video
    temporary_copy = None

    if args.local_cache:
        # Drive's FUSE mount is slow for the seeking cv2 does, badly enough to
        # dominate inference time. Copying the clip to local disk first is
        # usually several times faster overall despite the up-front copy.
        args.local_cache.mkdir(parents=True, exist_ok=True)
        # the folder hash keeps same-named clips from different folders apart
        tag = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:8]
        temporary_copy = args.local_cache / f"{tag}_{video.name}"
        if (
            not temporary_copy.exists()
            or temporary_copy.stat().st_size != video.stat().st_size
        ):
            print(f"  copying to {temporary_copy} ...", flush=True)
            # copy beside the final name and rename, so an interrupted copy is
            # never mistaken for a complete one on the next run
            partial_copy = temporary_copy.with_name(temporary_copy.name + ".partial")
            shutil.copy2(video, partial_copy)
            if partial_copy.stat().st_size != video.stat().st_size:
                partial_copy.unlink(missing_ok=True)
                raise ExportError(f"Copy of {video} to the local cache is incomplete")
            partial_copy.replace(temporary_copy)
        local = temporary_copy

    cap = cv2.VideoCapture(str(local))
    try:
        return _export_opened_video(
            cap, video, local, temporary_copy, output, args, predictor, model_description,
            enhance, enhancement, write_contours, started, progress, abort,
        )
    finally:
        cap.release()
        if temporary_copy is not None and not args.keep_cache:
            temporary_copy.unlink(missing_ok=True)


def _export_opened_video(
    cap, video, local, temporary_copy, output, args, predictor, model_description, enhance,
    enhancement, write_contours, started, progress=None, abort=None,
) -> dict | None:
    if not cap.isOpened():
        raise ExportError(f"Could not open {local}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.limit:
        n_frames = min(n_frames, args.limit)

    print(f"{video.name}: {n_frames} frames at {width}x{height}", flush=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # With --session, only the frames idtracker.ai will track are run through
    # the model; the rest keep an empty slot, so frame numbers still line up
    # with the video. The file records which frames it holds, and tracking
    # refuses to use it for any others.
    intervals = getattr(args, "intervals", None)
    if intervals is None:
        todo = [[0, n_frames]]
    else:
        todo = sm.normalize_intervals(intervals, n_frames)
    n_todo = sum(end - start for start, end in todo)
    if intervals is not None:
        print(
            f"  exporting {n_todo} of {n_frames} frames: "
            + ", ".join(f"[{start}, {end})" for start, end in todo),
            flush=True,
        )

    contours_per_frame: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
    scores_per_frame: list[list[float]] = [[] for _ in range(n_frames)]
    empty_frames = 0
    short_frames = 0
    read_failures = 0
    done = 0
    clock = time.monotonic()
    position = 0

    for start, end in todo:
        if start != position:
            seek(cap, start)
        position = end
        for frame_index in range(start, end):
            if abort is not None and abort():
                cap.release()
                if temporary_copy is not None and not args.keep_cache:
                    temporary_copy.unlink(missing_ok=True)
                return None

            done += 1
            ok, frame = cap.read()
            if not ok:
                # Keep the frame slot so frame numbers stay aligned with the
                # video. idtracker.ai treats a frame with no contours as one
                # with no visible animals, which is what an unreadable frame
                # amounts to. Counted apart from frames the model found empty.
                read_failures += 1
                if read_failures <= 5:
                    print(f"  WARNING: could not read frame {frame_index}", flush=True)
                if read_failures > max(1, MAX_READ_FAILURE_FRACTION * n_todo):
                    raise ExportError(
                        f"{video.name}: more than {MAX_READ_FAILURE_FRACTION:.0%} of"
                        f" the frames are unreadable ({read_failures} so far, at"
                        f" frame {frame_index}); the file is probably damaged or"
                        " truncated"
                    )
                # Report it here too: a run whose tail is unreadable would
                # otherwise leave the bar short of the end, which reads as a hang.
                if progress is not None:
                    progress(done, n_todo)
                continue

            frame_contours, frame_scores = contours_of_frame(
                predictor, enhance(frame), args, kernel
            )
            contours_per_frame[frame_index] = frame_contours
            scores_per_frame[frame_index] = frame_scores

            if not frame_contours:
                empty_frames += 1
            elif len(frame_contours) < args.max_instances:
                short_frames += 1

            if progress is not None:
                progress(done, n_todo)

            if done > 1 and (done - 1) % args.progress_every == 0:
                rate = done / max(time.monotonic() - clock, 1e-6)
                remaining = (n_todo - done) / max(rate, 1e-6)
                print(
                    f"  {done}/{n_todo} ({100 * done / n_todo:.1f}%)"
                    f"  {rate:.1f} fps  ~{remaining / 60:.0f} min left",
                    flush=True,
                )

    if read_failures:
        print(f"  WARNING: {read_failures} unreadable frame(s) left empty", flush=True)

    # Write to a .partial first, then move into place. A session that dies
    # mid-write would otherwise leave a truncated .h5 that the resume logic
    # would count as a finished video.
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    # A contour file should say what made it, now that more than one thing can.
    backend_provenance = {"backend": args.backend}
    if getattr(args, "classes", None):
        backend_provenance["classes"] = json.dumps(args.classes)
    if intervals is not None:
        backend_provenance["exported_intervals"] = json.dumps(todo)
    if args.backend == "sam3":
        backend_provenance["prompt"] = args.prompt

    write_contours(
        partial,
        contours_per_frame,
        width=width,
        height=height,
        scores_per_frame=scores_per_frame,
        # This file is shipped to Colab under a different name, so read it off
        # the file itself rather than hardcoding one of the two.
        source=Path(__file__).name,
        video=video.name,
        model=model_description,
        weights=Path(args.weights).name,
        weights_sha256=args.weights_sha256,
        read_failures=read_failures,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
        on_overlap=args.on_overlap,
        enhancement=json.dumps(enhancement),
        **backend_provenance,
        postprocessing=json.dumps(
            {
                "min_component": args.min_component,
                "dilate": args.dilate,
                "dedup_iou": args.dedup_iou,
                "merge_overlap": args.merge_overlap,
            }
        ),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    partial.replace(output)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    total = sum(len(c) for c in contours_per_frame)
    stats = {
        "video": video.name,
        "output": str(output),
        "frames": len(contours_per_frame),
        "processed_frames": n_todo,
        "contours": total,
        "contours_per_frame": round(total / max(len(contours_per_frame), 1), 3),
        "empty_frames": empty_frames,
        "short_frames": short_frames,
        "read_failures": read_failures,
        "seconds": round(elapsed, 1),
        "fps": round(n_todo / max(elapsed, 1e-6), 2),
        "size_mb": round(output.stat().st_size / 1e6, 2),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(
        f"  done: {output.name} ({stats['size_mb']} MB), {total} contours,"
        f" {stats['fps']} fps, {empty_frames} empty / {short_frames} short frames",
        flush=True,
    )
    return stats


def _is_sidecar(path) -> bool:
    """Filesystem metadata that merely looks like a media file.

    macOS writes a "._name.mp4" companion beside every file it copies onto
    a non-Apple filesystem. It carries the same extension, so any *.mp4
    glob picks it up and OpenCV then reports it as an unreadable video.
    """
    name = Path(path).name
    return name.startswith("._") or name in (".DS_Store", "Thumbs.db", "desktop.ini")


def resolve_videos(args) -> list[Path]:
    if args.video:
        return [args.video]
    videos: list[Path] = []
    for pattern in args.videos:
        if any(ch in str(pattern) for ch in "*?"):
            videos.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            videos.append(pattern)
    return [v for v in videos if v.is_file() and not _is_sidecar(v)]


def main():
    parser = argparse.ArgumentParser(
        description="Export Detectron2 instance contours for idtracker.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("detectron2", "sam3"),
        default="detectron2",
        help=(
            "detectron2: a Mask R-CNN fine-tuned on frames you annotated;"
            " sam3: Meta's SAM 3 prompted with a word for the animal, which"
            " needs no annotation or training but is not specific to your setup"
        ),
    )
    parser.add_argument(
        "--prompt",
        help="with --backend sam3, the word describing the animal, e.g. 'fish'",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="a single video")
    source.add_argument(
        "--videos", type=Path, nargs="+", help="several videos, or a glob"
    )
    source.add_argument(
        "--session",
        type=Path,
        nargs="+",
        help=(
            "idtracker.ai session .toml file(s), or a glob. Each file's"
            " [segmentation_model] table says which model to run and where the"
            " contours go; only the frames in its tracking_intervals are run."
            " The model options below are then taken from the table, not from"
            " the command line"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="model_final.pth (or sam3.pt); 'model_zoo' for the COCO checkpoint"
        " of --config, not fine-tuned; 'huggingface' to fetch sam3.pt (gated:"
        " needs a token from an account granted access)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output", type=Path, help="contour file, with --video")
    target.add_argument(
        "--output-dir",
        type=Path,
        help="folder for one <video stem>.h5 per input, with --videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redo videos whose contour file already exists. The default is to"
        " skip them, which is what makes a batch resumable across sessions",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        help="copy each video here before decoding, e.g. /content/cache when the"
        " videos live on a slow Drive mount",
    )
    parser.add_argument(
        "--keep-cache", action="store_true", help="do not delete cached copies"
    )
    parser.add_argument(
        "--progress-every", type=int, default=500, help="frames between progress lines"
    )
    parser.add_argument(
        "--config",
        help="model_zoo config the model was fine-tuned from"
        " (default: taken from training_metadata.json)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="default: taken from training_metadata.json, else 1",
    )
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=5,
        help="expected number of animals; caps detections per frame",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        help="inference scale; default: the value training recorded, else 640",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-component",
        type=int,
        default=80,
        help="drop connected components smaller than this, in pixels",
    )
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--dedup-iou", type=float, default=0.7)
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="discard contours below this polygon area",
    )
    parser.add_argument(
        "--on-overlap",
        choices=("merge", "split", "keep"),
        default="merge",
        help=(
            "merge: overlapping animals become one blob for idtracker.ai's "
            "crossing detector; split: assign contested pixels to the higher "
            "score and keep animals separate; keep: emit masks untouched"
        ),
    )
    parser.add_argument(
        "--merge-overlap",
        type=float,
        default=0.15,
        help=(
            "with --on-overlap merge, fuse two masks when their intersection "
            "covers at least this fraction of the smaller one"
        ),
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="keep only these class indices, e.g. 14 for COCO's bird with"
        " --weights model_zoo (default: every class)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N frames (0 = whole video)"
    )
    fp.add_arguments(parser)
    args = parser.parse_args()
    args.intervals = None

    if args.session:
        model_flags = {
            "--weights": args.weights,
            "--output": args.output,
            "--output-dir": args.output_dir,
            "--config": args.config,
            "--num-classes": args.num_classes,
            "--min-size": args.min_size,
            "--prompt": args.prompt,
            "--classes": args.classes,
        }
        used = [flag for flag, value in model_flags.items() if value is not None]
        if used:
            parser.error(
                f"{', '.join(used)} cannot be combined with --session: the"
                " session's [segmentation_model] table sets them"
            )
        if args.limit:
            parser.error("--limit cannot be combined with --session")
        return session_main(args)

    if args.weights is None:
        parser.error("--weights is required")
    if not (args.output or args.output_dir):
        parser.error("one of --output or --output-dir is required")
    if args.classes is not None:
        args.classes = sorted(set(args.classes))
    if str(args.weights) == "model_zoo" and args.backend != "detectron2":
        parser.error("--weights model_zoo only applies to --backend detectron2")
    if str(args.weights).lower() == sm.HUGGINGFACE and args.backend != "sam3":
        parser.error("--weights huggingface only applies to --backend sam3")

    if args.videos and not args.output_dir:
        parser.error("--videos needs --output-dir")
    if args.video and not args.output:
        parser.error("--video needs --output")

    # Say so rather than ignoring them. A flag that looks accepted but does
    # nothing is how someone ends up believing they set a class count on a
    # model that has no classes.
    detectron2_only = {
        "--num-classes": args.num_classes,
        "--min-size": args.min_size,
        "--config": args.config,
        "--classes": args.classes,
    }
    if args.backend == "sam3":
        if not args.prompt:
            parser.error("--backend sam3 needs --prompt, e.g. --prompt fish")
        used = [flag for flag, value in detectron2_only.items() if value is not None]
        if used:
            verb = "only applies" if len(used) == 1 else "only apply"
            parser.error(f"{', '.join(used)} {verb} to --backend detectron2")
    elif args.prompt:
        parser.error("--prompt only applies to --backend sam3")

    videos = resolve_videos(args)
    if not videos:
        raise SystemExit("No videos matched")

    write_contours = load_writer()
    # SAM 3 was never fine-tuned on this setup, so there is no training record
    # to read and nothing to check the enhancement against. Looking for one
    # would only produce a warning about a file that was never meant to exist.
    metadata = (
        load_training_metadata(args.weights)
        if args.backend == "detectron2" and str(args.weights) != "model_zoo"
        else None
    )
    # Inference adopts whatever the model was trained with, unless the user
    # deliberately overrides it. Retyping the flags identically at inference
    # time was the old requirement, and an easy thing to get silently wrong.
    trained_with = (metadata or {}).get("enhancement")
    enhancement, notes = fp.resolve_settings(
        args, fallback=trained_with, fallback_label="the model's training record"
    )
    for note in notes:
        print(note)
    print(f"enhancement: {fp.describe(enhancement)}")
    enhance = fp.make_enhancer_from(enhancement)

    if args.backend == "detectron2" and str(args.weights) != "model_zoo":
        if metadata is None:
            print(
                f"No training_metadata.json beside {args.weights}; cannot check"
                " that inference matches how the model was trained."
            )
        else:
            mismatch = fp.check_settings_match(trained_with, enhancement, "training")
            if mismatch:
                print(f"\nWARNING: {mismatch}\n")

    def output_for(video: Path) -> Path:
        path = args.output if args.output else args.output_dir / f"{video.stem}.h5"
        if args.limit:
            # a partial run must never be mistaken for the real export
            path = path.with_name(f"{path.stem}.limit{args.limit}{path.suffix}")
        return path

    outputs = [output_for(v) for v in videos]
    if len(set(outputs)) != len(outputs):
        raise SystemExit(
            "Two videos would write the same contour file (same file name in"
            " different folders). Export them separately."
        )

    # Videos are identified in the log by their path relative to the folder they
    # share, so same-named clips in different folders do not overwrite each other.
    root = Path(os.path.commonpath([str(v.resolve().parent) for v in videos]))

    def log_key(video: Path) -> str:
        key = video.resolve().relative_to(root).as_posix()
        return f"{key}#limit{args.limit}" if args.limit else key

    try:
        args.weights = local_weights(args.backend, args.weights)
    except RuntimeError as exc:  # Sam3WeightsError: a refused download, explained
        raise SystemExit(str(exc)) from exc
    args.weights_sha256 = weights_sha256(args.weights)

    def needs_export(video: Path) -> bool:
        output = output_for(video)
        if args.overwrite or not output.exists():
            return True
        reason = stale_reason(output, video, args, enhancement)
        if reason:
            print(f"{output.name} exists but is out of date ({reason}); redoing it")
        return bool(reason)

    pending = [v for v in videos if needs_export(v)]
    done_already = len(videos) - len(pending)
    print(
        f"{len(videos)} video(s); {done_already} already exported,"
        f" {len(pending)} to do"
    )
    if not pending:
        print("Nothing to do. Pass --overwrite to redo them.")
        return

    predictor, model_description = build_predictor(args, metadata)

    log_path = (args.output_dir or args.output.parent) / (
        "export_log.limit.json" if args.limit else "export_log.json"
    )
    log = []
    if log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []

    failed: list[str] = []
    for i, video in enumerate(pending, start=1):
        print(f"\n[{i}/{len(pending)}] {video.name}", flush=True)
        try:
            stats = export_video(
                video,
                output_for(video),
                args,
                predictor,
                model_description,
                enhance,
                enhancement,
                write_contours,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Finished videos are kept; rerun to continue.")
            break
        except ExportError as exc:
            print()
            print(f"ERROR: {exc}", flush=True)
            failed.append(video.name)
            continue
        if stats is None:  # only reachable with an abort callback, but cheap
            break
        stats["key"] = log_key(video)
        log = [
            entry
            for entry in log
            if entry.get("key", entry.get("video")) != stats["key"]
        ]
        log.append(stats)
        # rewritten after every video, so a session that dies still leaves a record
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    total_frames = sum(e["frames"] for e in log)
    empty = sum(e["empty_frames"] for e in log)
    short = sum(e["short_frames"] for e in log)
    print(f"\n{len(log)}/{len(videos)} video(s) exported, {total_frames:,} frames")
    print(
        f"  {empty:,} frames with no detection,"
        f" {short:,} with fewer than {args.max_instances}"
    )
    print(f"  log: {log_path}")
    if failed:
        print()
        print(f"{len(failed)} video(s) FAILED: {', '.join(failed)}")
    if empty or short:
        print(
            "\nFrames with missing animals are left as they are, on purpose.\n"
            "Do not fill them in here: idtracker.ai's crossing detection and\n"
            "interpolation are built for exactly this and work from the whole\n"
            "video, not from one previous frame."
        )
    if failed:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# --session: export what a session .toml says it will track
# --------------------------------------------------------------------------

# table keys that become exporter options of the same name
_MODEL_OPTIONS = (
    "backend", "prompt", "config", "num_classes", "min_size", "classes",
    "score_threshold", "on_overlap", "merge_overlap", "min_component",
    "dilate", "dedup_iou", "min_area", "device",
)


def load_session_toml(path: Path) -> dict:
    """Reads a session .toml the way idtracker.ai does: keys are case
    insensitive and an empty string means "not set"."""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except ImportError:  # Python < 3.11
        import toml  # type: ignore

        data = toml.loads(path.read_text(encoding="utf-8"))
    return {
        str(key).lower(): (None if value == "" else value)
        for key, value in data.items()
    }


def _expand(patterns) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pattern = Path(pattern).expanduser()
        if any(ch in str(pattern) for ch in "*?"):
            paths.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            paths.append(pattern)
    return [p for p in paths if not _is_sidecar(p)]


def session_jobs(toml_paths, frame_count=video_frame_count) -> list[dict]:
    """One job per video of every session: what to run, over which frames,
    and where to write. Raises SystemExit naming the file at the first problem,
    before any GPU time is spent."""
    jobs: dict[Path, dict] = {}
    for toml_path in _expand(toml_paths):
        def fail(message: str, toml_path: Path = toml_path):
            raise SystemExit(f"{toml_path}: {message}")

        if not toml_path.is_file():
            fail("not found")
        try:
            params = load_session_toml(toml_path)
        except Exception as exc:  # noqa: BLE001 - any parse error, named
            fail(f"could not read it ({exc})")
        try:
            spec = sm.normalize_model(params.get("segmentation_model"))
        except sm.SegmentationModelError as exc:
            fail(str(exc))
        if spec is None:
            fail(
                "has no [segmentation_model] table, so there is no model to run."
                " Add one, or export with --video/--videos"
            )

        videos = params.get("video_paths")
        if not videos:
            fail("has no video_paths")
        if isinstance(videos, str):
            videos = [videos]
        videos = [Path(v).expanduser() for v in videos]
        missing = [str(v) for v in videos if not v.is_file()]
        if missing:
            fail("video(s) not found: " + ", ".join(missing))

        given = params.get("external_contours")
        try:
            if given:
                outputs = [Path(given)] if isinstance(given, str) else [Path(p) for p in given]
                outputs = [p.expanduser() for p in outputs]
                if len(outputs) != len(videos):
                    fail(
                        f"external_contours lists {len(outputs)} file(s) for"
                        f" {len(videos)} video(s); there must be one per video"
                    )
            else:
                outputs = sm.default_contour_paths(spec, videos)
            max_instances = sm.effective_max_instances(
                spec, params.get("number_of_animals")
            )
        except sm.SegmentationModelError as exc:
            fail(str(exc))

        n_frames = [frame_count(v) for v in videos]
        for video, n in zip(videos, n_frames):
            if n <= 0:
                fail(f"OpenCV cannot count the frames of {video}")
        per_video = sm.local_intervals(params.get("tracking_intervals"), n_frames)

        model_key = json.dumps(
            {k: v for k, v in spec.items() if k != "contours_dir"}
            | {"max_instances": max_instances},
            sort_keys=True,
        )
        for video, output, intervals in zip(videos, outputs, per_video):
            key = output.resolve()
            if key in jobs:
                # the same clip in two sessions: export the union once
                job = jobs[key]
                if job["video"].resolve() != video.resolve():
                    fail(
                        f"{output} is also where {job['session']} puts the"
                        f" contours of a different video ({job['video'].name})"
                    )
                if job["model_key"] != model_key:
                    fail(
                        f"{video.name} is also in {job['session']} with a"
                        " different [segmentation_model] or number of animals;"
                        " one contour file cannot serve both"
                    )
                job["intervals"] = sm.normalize_intervals(
                    job["intervals"] + intervals, job["n_frames"]
                )
                job["session"] += f", {toml_path.name}"
                continue
            jobs[key] = {
                "session": toml_path.name,
                "video": video,
                "output": output,
                "n_frames": n_frames[videos.index(video)],
                "intervals": intervals,
                "spec": spec,
                "max_instances": max_instances,
                "model_key": model_key,
            }
    if not jobs:
        raise SystemExit("No session files matched")
    return list(jobs.values())


def _args_for(args, spec: dict, max_instances: int):
    run_args = argparse.Namespace(**vars(args))
    for option in _MODEL_OPTIONS:
        setattr(run_args, option, spec[option])
    run_args.weights = (
        spec["weights"] if spec["weights"] == sm.MODEL_ZOO else Path(spec["weights"]).expanduser()
    )
    run_args.max_instances = max_instances
    run_args.output = None
    run_args.output_dir = None
    run_args.limit = 0
    return run_args


def session_main(args) -> None:
    jobs = session_jobs(args.session)
    write_contours = load_writer()
    print(f"{len(jobs)} video(s) from {len(args.session)} session argument(s)")

    groups: dict[str, list[dict]] = {}
    for job in jobs:
        groups.setdefault(job["model_key"], []).append(job)

    failed: list[str] = []
    exported = 0
    for group in groups.values():
        spec, max_instances = group[0]["spec"], group[0]["max_instances"]
        run_args = _args_for(args, spec, max_instances)
        try:
            run_args.weights = local_weights(run_args.backend, run_args.weights)
        except RuntimeError as exc:  # Sam3WeightsError: a refused download
            for job in group:
                failed.append(job["video"].name)
            print(f"ERROR: {exc}")
            continue
        if run_args.weights != sm.MODEL_ZOO and not Path(run_args.weights).is_file():
            for job in group:
                failed.append(job["video"].name)
            print(f"ERROR: weights not found: {run_args.weights}")
            continue

        metadata = (
            load_training_metadata(run_args.weights)
            if run_args.backend == "detectron2" and run_args.weights != sm.MODEL_ZOO
            else None
        )
        trained_with = (metadata or {}).get("enhancement")
        enhancement, notes = fp.resolve_settings(
            run_args, fallback=trained_with, fallback_label="the model's training record"
        )
        for note in notes:
            print(note)
        enhance = fp.make_enhancer_from(enhancement)
        run_args.weights_sha256 = weights_sha256(run_args.weights)

        pending = []
        for job in group:
            run_args.intervals = job["intervals"]
            output = job["output"]
            if args.overwrite or not output.exists():
                pending.append(job)
                continue
            reason = stale_reason(output, job["video"], run_args, enhancement)
            if reason:
                print(f"{output.name} exists but is out of date ({reason}); redoing it")
                pending.append(job)
        print(
            f"\n{sm.weights_label(spec)}, {max_instances} per frame:"
            f" {len(group) - len(pending)} already exported, {len(pending)} to do"
        )
        if not pending:
            continue

        predictor, model_description = build_predictor(run_args, metadata)
        for i, job in enumerate(pending, start=1):
            run_args.intervals = job["intervals"]
            print(f"\n[{i}/{len(pending)}] {job['video'].name} ({job['session']})", flush=True)
            try:
                stats = export_video(
                    job["video"], job["output"], run_args, predictor,
                    model_description, enhance, enhancement, write_contours,
                )
            except KeyboardInterrupt:
                print("\nInterrupted. Finished videos are kept; rerun to continue.")
                raise SystemExit(130)
            except ExportError as exc:
                print(f"\nERROR: {exc}", flush=True)
                failed.append(job["video"].name)
                continue
            if stats is None:
                break
            exported += 1
            stats["key"] = job["video"].resolve().as_posix()
            stats["sessions"] = job["session"]
            log_path = job["output"].parent / "export_log.json"
            log = []
            if log_path.is_file():
                try:
                    log = json.loads(log_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    log = []
            log = [e for e in log if e.get("key", e.get("video")) != stats["key"]]
            log.append(stats)
            log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n{exported} video(s) exported")
    # Per video, how often the model saw fewer animals than the session holds.
    # A model trained on few pens shows its blind spots here first.
    print("\nframes with no detection / fewer than the number of animals:")
    for job in jobs:
        log_path = job["output"].parent / "export_log.json"
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = job["video"].resolve().as_posix()
        entry = next((e for e in log if e.get("key") == key), None)
        if entry is None:
            continue
        processed = max(entry.get("processed_frames", entry["frames"]), 1)
        print(
            f"  {job['session']:<32} {job['video'].name:<16}"
            f" {100 * entry['empty_frames'] / processed:5.1f}%"
            f" / {100 * entry['short_frames'] / processed:5.1f}%"
            f"  (of {processed} frames, {job['max_instances']} animals)"
        )
    if failed:
        print(f"{len(failed)} video(s) FAILED: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
