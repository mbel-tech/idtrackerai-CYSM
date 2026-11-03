"""The ``[segmentation_model]`` table of a session .toml.

A session segmented by an external model needs two stages, usually on two
machines: the exporter runs the model over the videos and writes contour files,
and idtracker.ai tracks from those files. Without a shared description of the
model, each stage is configured separately, and nothing stops a session being
tracked from contours made by other weights, at another score threshold, with
a detection cap below the number of animals, or over frames that do not cover
the tracking interval. Each of those runs cleanly and produces wrong
trajectories.

This table is that shared description. The exporter reads it to know what to
run (``idtrackerai_d2_export --session session.toml``) and idtracker.ai reads it
to check, before segmenting, that the contour files were made that way.

Example::

    number_of_animals = 9
    tracking_intervals = [[3600, 36000]]

    [segmentation_model]
    backend = "detectron2"
    weights = "/content/drive/MyDrive/broilercamp/model/model_final.pth"
    score_threshold = 0.7
    min_size = 640
    contours_dir = "/content/drive/MyDrive/broilercamp/contours/R2_01_10_P1_SB"

Kept free of idtracker.ai imports: the Colab bundle ships this file next to the
exporter, where idtracker.ai is not installed.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BACKENDS = ("detectron2", "sam3")
OVERLAP_POLICIES = ("merge", "split", "keep")

# Same defaults as the exporter's command line, so a table that sets only
# backend and weights exports exactly what the plain command would.
DEFAULTS: dict[str, Any] = {
    "backend": "detectron2",
    "weights": None,
    "config": None,
    "num_classes": None,
    "classes": None,
    "min_size": None,
    "prompt": None,
    "score_threshold": 0.7,
    "max_instances": None,
    "on_overlap": "merge",
    "merge_overlap": 0.15,
    "min_component": 80,
    "dilate": 1,
    "dedup_iou": 0.7,
    "min_area": 1.0,
    "device": "cuda",
    "contours_dir": None,
}

DETECTRON2_ONLY = ("config", "num_classes", "min_size", "classes")

# weights = "model_zoo" loads the COCO-trained checkpoint of the config itself,
# with no fine-tuning: 80 classes, of which COCO_BIRD is the one poultry
# falls under. Usable straight away, but a model fine-tuned on your own frames
# is what to publish with.
MODEL_ZOO = "model_zoo"
COCO_BIRD = 14


class SegmentationModelError(ValueError):
    """The [segmentation_model] table is invalid, or the contour files do not
    match it."""


def _number(spec, key, kind, low=None, high=None, low_open=False):
    value = spec[key]
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentationModelError(
            f"segmentation_model.{key} must be a number, got {value!r}"
        )
    if kind is int and value != int(value):
        raise SegmentationModelError(
            f"segmentation_model.{key} must be a whole number, got {value!r}"
        )
    if low is not None and (value <= low if low_open else value < low):
        bound = f"greater than {low}" if low_open else f"at least {low}"
        raise SegmentationModelError(
            f"segmentation_model.{key} must be {bound}, got {value!r}"
        )
    if high is not None and value > high:
        raise SegmentationModelError(
            f"segmentation_model.{key} must be at most {high}, got {value!r}"
        )
    spec[key] = kind(value)


def normalize_model(spec: dict | None) -> dict | None:
    """Completes the table from the defaults and rejects anything invalid.

    Returns None when there is no table, so sessions without one behave as
    before. Empty strings mean "not set", as everywhere else in the .toml.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise SegmentationModelError(
            "segmentation_model must be a table ([segmentation_model] with keys"
            f" below it), got {type(spec).__name__}"
        )

    given = {
        str(key).lower(): (None if value == "" else value)
        for key, value in spec.items()
    }
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise SegmentationModelError(
            f"Unknown key(s) in segmentation_model: {', '.join(unknown)}."
            f" Valid keys: {', '.join(DEFAULTS)}"
        )
    out = DEFAULTS | {key: value for key, value in given.items() if value is not None}

    out["backend"] = str(out["backend"]).lower()
    if out["backend"] not in BACKENDS:
        raise SegmentationModelError(
            f"segmentation_model.backend must be one of {', '.join(BACKENDS)},"
            f" got {out['backend']!r}"
        )
    if not out["weights"]:
        raise SegmentationModelError(
            "segmentation_model.weights is missing: the path of the model the"
            " contours are made with (model_final.pth, or sam3.pt)"
        )
    out["weights"] = str(out["weights"])
    if out["weights"].lower() == MODEL_ZOO:
        out["weights"] = MODEL_ZOO
        if out["backend"] != "detectron2":
            raise SegmentationModelError(
                'weights = "model_zoo" only applies to backend = "detectron2"'
            )
    if out["contours_dir"] is not None:
        out["contours_dir"] = str(out["contours_dir"])

    if out["backend"] == "sam3":
        if not out["prompt"]:
            raise SegmentationModelError(
                'segmentation_model.backend = "sam3" needs a prompt, e.g.'
                ' prompt = "chicken"'
            )
        used = [key for key in DETECTRON2_ONLY if given.get(key) is not None]
        if used:
            raise SegmentationModelError(
                f"segmentation_model.{', '.join(used)} only apply to"
                ' backend = "detectron2"'
            )
    elif out["prompt"]:
        raise SegmentationModelError(
            'segmentation_model.prompt only applies to backend = "sam3"'
        )

    _number(out, "score_threshold", float, 0, 1)
    _number(out, "max_instances", int, 1)
    _number(out, "num_classes", int, 1)
    _number(out, "min_size", int, 32)
    _number(out, "merge_overlap", float, 0, 1, low_open=True)
    _number(out, "min_component", int, 0)
    _number(out, "dilate", int, 0)
    _number(out, "dedup_iou", float, 0, 1, low_open=True)
    _number(out, "min_area", float, 0)

    if out["classes"] is not None:
        classes = out["classes"]
        if isinstance(classes, (int, float)) and not isinstance(classes, bool):
            classes = [classes]
        if (
            not isinstance(classes, list)
            or not classes
            or any(
                isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in classes
            )
        ):
            raise SegmentationModelError(
                "segmentation_model.classes must be a list of class indices,"
                f" e.g. [{COCO_BIRD}] for COCO's bird, got {out['classes']!r}"
            )
        out["classes"] = sorted(set(classes))
    elif out["weights"] == MODEL_ZOO:
        # every COCO class would otherwise become a blob: feeders, people, ...
        raise SegmentationModelError(
            'weights = "model_zoo" detects all 80 COCO classes; set classes ='
            f" [{COCO_BIRD}] (bird) to keep only the animals"
        )

    out["on_overlap"] = str(out["on_overlap"]).lower()
    if out["on_overlap"] not in OVERLAP_POLICIES:
        raise SegmentationModelError(
            "segmentation_model.on_overlap must be one of"
            f" {', '.join(OVERLAP_POLICIES)}, got {out['on_overlap']!r}"
        )
    return out


def effective_max_instances(spec: dict, number_of_animals: int | None) -> int:
    """The detection cap per frame. Defaults to the number of animals.

    The exporter's own default is 5, which silently drops animals in any arena
    holding more. With the count known from the session there is no reason to
    guess, and a pen whose count changes by day gets the right cap each day.
    """
    if spec.get("max_instances"):
        return int(spec["max_instances"])
    if number_of_animals and number_of_animals > 0:
        return int(number_of_animals)
    raise SegmentationModelError(
        "Cannot tell how many detections to keep per frame: set"
        " number_of_animals, or segmentation_model.max_instances"
    )


def default_contour_paths(spec: dict, video_paths: Sequence[Path | str]) -> list[Path]:
    """One contour file per video, ``<contours_dir>/<video stem>.h5``.

    GoPro chapters from different cameras can share a name (GX010561.MP4), so
    give each session its own contours_dir rather than one for the whole trial.
    """
    if not spec.get("contours_dir"):
        raise SegmentationModelError(
            "Neither external_contours nor segmentation_model.contours_dir is"
            " set, so there is no telling where this session's contour files are."
            " Set contours_dir (one folder per session) or list the files in"
            " external_contours"
        )
    folder = Path(spec["contours_dir"]).expanduser()
    paths = [folder / f"{Path(video).stem}.h5" for video in video_paths]
    if len(set(paths)) != len(paths):
        raise SegmentationModelError(
            "Two videos of this session have the same file name, so they would"
            " share a contour file in contours_dir. List the files explicitly in"
            " external_contours"
        )
    return paths


def normalize_intervals(tracking_intervals, total_frames: int) -> list[list[int]]:
    """Global tracking intervals as sorted, merged [start, end) pairs."""
    if tracking_intervals is None or tracking_intervals == "":
        return [[0, total_frames]]
    intervals = list(tracking_intervals)
    if intervals and isinstance(intervals[0], (int, float)):
        intervals = [intervals]
    cleaned = sorted(
        [max(0, int(start)), min(total_frames, int(end))]
        for start, end in intervals
        if min(total_frames, int(end)) > max(0, int(start))
    )
    merged: list[list[int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def local_intervals(
    tracking_intervals, video_n_frames: Sequence[int]
) -> list[list[list[int]]]:
    """Splits global tracking intervals into each video's own frame numbers.

    idtracker.ai reads several videos as one timeline, but each contour file
    is numbered from the first frame of its own video.
    """
    total = int(sum(video_n_frames))
    global_intervals = normalize_intervals(tracking_intervals, total)
    per_video: list[list[list[int]]] = []
    offset = 0
    for n_frames in video_n_frames:
        own = []
        for start, end in global_intervals:
            start, end = max(start, offset), min(end, offset + n_frames)
            if end > start:
                own.append([start - offset, end - offset])
        per_video.append(own)
        offset += n_frames
    return per_video


def covers(exported: list[list[int]] | None, needed: list[list[int]]) -> bool:
    """True if every needed frame was exported. None means the whole video."""
    if exported is None:
        return True
    merged = normalize_intervals(exported, max([e for _, e in exported], default=0))
    for start, end in needed:
        if not any(s <= start and end <= e for s, e in merged):
            return False
    return True


def exported_intervals_of(attrs: dict) -> list[list[int]] | None:
    """The frames a contour file holds model output for. None: all of them."""
    raw = attrs.get("exported_intervals")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return [[int(s), int(e)] for s, e in json.loads(raw)]


def weights_label(spec: dict) -> str:
    """What the exporter records as ``weights``: the file name, or model_zoo."""
    return MODEL_ZOO if spec["weights"] == MODEL_ZOO else Path(spec["weights"]).name


def _close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def mismatches(
    attrs: dict,
    spec: dict,
    max_instances: int,
    weights_sha256: str | None = None,
) -> list[str]:
    """How a contour file's provenance differs from the model table.

    ``weights_sha256`` is compared when the caller could hash the weights;
    otherwise only their file name is, which is all that can be checked on a
    machine the weights were never copied to.
    """
    problems = []

    backend = _text(attrs.get("backend", "detectron2"))
    if backend != spec["backend"]:
        problems.append(f"backend is {backend!r}, the table says {spec['backend']!r}")

    if weights_sha256 and "weights_sha256" in attrs:
        if _text(attrs["weights_sha256"]) != weights_sha256:
            problems.append(
                f"made with other weights ({_text(attrs.get('weights', '?'))},"
                f" a different file from {Path(spec['weights']).name})"
            )
    elif "weights" in attrs and _text(attrs["weights"]) != weights_label(spec):
        problems.append(
            f"made with {_text(attrs['weights'])}, the table says"
            f" {Path(spec['weights']).name}"
        )

    if "score_threshold" in attrs and not _close(
        attrs["score_threshold"], spec["score_threshold"]
    ):
        problems.append(
            f"score_threshold is {float(attrs['score_threshold']):g}, the table"
            f" says {spec['score_threshold']:g}"
        )
    if "max_instances" in attrs and int(attrs["max_instances"]) != max_instances:
        problems.append(
            f"max_instances is {int(attrs['max_instances'])}, this session needs"
            f" {max_instances}"
        )
    if "on_overlap" in attrs and _text(attrs["on_overlap"]) != spec["on_overlap"]:
        problems.append(
            f"on_overlap is {_text(attrs['on_overlap'])!r}, the table says"
            f" {spec['on_overlap']!r}"
        )
    if (
        spec.get("config")
        and "model" in attrs
        and _text(attrs["model"]) != spec["config"]
    ):
        problems.append(
            f"model is {_text(attrs['model'])}, the table says {spec['config']}"
        )
    if "classes" in attrs or spec.get("classes"):
        recorded = json.loads(_text(attrs.get("classes", "null")))
        if recorded != spec.get("classes"):
            problems.append(
                f"classes are {recorded}, the table says {spec.get('classes')}"
            )
    if (
        spec["backend"] == "sam3"
        and "prompt" in attrs
        and _text(attrs["prompt"]) != spec["prompt"]
    ):
        problems.append(
            f"prompt is {_text(attrs['prompt'])!r}, the table says {spec['prompt']!r}"
        )
    return problems
