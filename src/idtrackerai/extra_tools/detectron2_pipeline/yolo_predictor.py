"""A YOLO segmentation model as a segmentation backend.

For a model you already trained with ultralytics (YOLO11-seg, YOLO26-seg, ...):
its ``best.pt`` runs over the video and its instance masks become the contours
idtracker.ai tracks, exactly as a Mask R-CNN's or SAM 3's do. It is the fast
option -- a medium model runs at tens of frames per second on one GPU, where
SAM 3 manages one or two -- and, being trained on your own frames, the one
that knows your pens.

Outlines, not boxes or points
-----------------------------
idtracker.ai tells animals apart from the pixels inside each outline, on every
frame it tracks. So the weights have to be a *segmentation* model (``-seg``);
a detection model's boxes are refused here rather than tracked as rectangles.
For the same reason a table of centre points that an earlier YOLO run saved
cannot be tracked from: the points say where each animal was, not which
pixels were the animal. Run the model again over the video, which is what
this backend does.

Installation
------------
The runtime is the ``ultralytics`` package::

    pip install ultralytics

Like Detectron2 and ``sam3``, it is deliberately not declared in
``pyproject.toml`` and is imported only inside the code that needs it, so
idtracker.ai itself never depends on it. That matters more here than for the
others: ultralytics is AGPL-3.0, and so, by its authors' terms, are weights
trained with it. This file only calls it; it does not contain or
redistribute it. Read ``NOTICE`` before shipping anything built on this route.

Per frame, without YOLO's tracker
---------------------------------
ultralytics can also track (ByteTrack, BoT-SORT). That is not used: the
sidecar holds contours and no identity, and identity is what idtracker.ai's
fragmentation and identification work out over the whole video rather than
from the previous frame.
"""

from pathlib import Path

import cv2
import numpy as np


class YoloWeightsError(RuntimeError):
    """The YOLO weights cannot make contours: wrong kind of model, or no
    ultralytics to load them with."""


def resolve_device(requested: str) -> str:
    """The device asked for, or the CPU when CUDA was asked for and is absent.

    ``--device`` defaults to ``cuda``. Saying so up front beats a CUDA error
    from inside ultralytics on the first frame.
    """
    if requested != "cuda":
        return requested
    try:
        import torch  # noqa: PLC0415  (heavy, optional, GPU-only dependency)

        available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch: ultralytics will say so itself
        available = False
    if not available:
        print(
            "No CUDA device available; falling back to CPU. YOLO on a CPU runs"
            " at a few frames per second: enough to try a clip, not to export"
            " a trial."
        )
        return "cpu"
    return requested


class YoloPredictor:
    """Segments one frame at a time with a trained YOLO ``-seg`` model.

    Exposes the same ``predict(frame) -> (masks, scores)`` interface as the
    other backends, so the export loop does not know which it has.

    ``imgsz`` is the size the frame is resized to for the network. Left at
    None, ultralytics uses the size the model was trained at, which is nearly
    always what you want; set it to trade accuracy on small animals for speed.
    ``max_instances`` at 0 keeps everything the model reports.

    Security: the weights are unpickled by torch, which can run arbitrary
    code. Only load weights you trained or trust.
    """

    def __init__(
        self,
        weights,
        device: str = "cuda",
        score_threshold: float = 0.5,
        max_instances: int = 0,
        imgsz: int | None = None,
        classes: list[int] | None = None,
    ):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}. Pass the best.pt of a"
                " segmentation model trained with ultralytics."
            )
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise YoloWeightsError(
                "The YOLO backend needs the ultralytics package, which is not"
                f" installed here ({exc}). Install it with:"
                "  pip install ultralytics"
            ) from exc

        self.device = resolve_device(device)
        self.score_threshold = float(score_threshold)
        self.max_instances = int(max_instances or 0)
        self.imgsz = int(imgsz) if imgsz else None
        self.classes = list(classes) if classes else None

        print(f"Model: YOLO, {weights.name}, on {self.device}")
        self._model = YOLO(str(weights))
        task = getattr(self._model, "task", "segment")
        if task != "segment":
            raise YoloWeightsError(
                f"{weights.name} is a YOLO {task} model, not a segmentation"
                " model. idtracker.ai tracks from each animal's outline, which"
                " only a -seg model (task 'segment') produces. Train or pick"
                " the -seg variant, e.g. yolo26m-seg."
            )
        self.description = f"yolo ({weights.name})"

    def predict(self, frame: np.ndarray):
        """Returns the masks YOLO finds in one BGR frame, and their scores."""
        options: dict = {
            "conf": self.score_threshold,
            "device": self.device,
            # Masks at the frame's own resolution. Without it they come back
            # at the network's, and upscaling those afterwards gives contours
            # visibly coarser than the animals.
            "retina_masks": True,
            "verbose": False,
        }
        if self.max_instances > 0:
            options["max_det"] = self.max_instances
        if self.imgsz:
            options["imgsz"] = self.imgsz
        if self.classes:
            options["classes"] = self.classes

        # ultralytics takes an OpenCV frame as it is: for arrays it assumes
        # BGR. Converting to RGB first, as SAM 3 needs, would swap the
        # channels and quietly segment worse.
        result = self._model.predict(frame, **options)[0]

        if getattr(result, "masks", None) is None:
            return [], []
        masks = _to_numpy(result.masks.data)
        scores = _to_numpy(result.boxes.conf)
        if masks is None or len(masks) == 0:
            return [], []

        # ultralytics already applied conf and max_det. Both are enforced
        # again because they are ours to guarantee: the sidecar records them,
        # and tracking is checked against that record.
        keep = [i for i, s in enumerate(scores) if float(s) >= self.score_threshold]
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        if self.max_instances > 0:
            keep = keep[: self.max_instances]

        height, width = frame.shape[:2]
        out_masks = [_as_binary_mask(masks[i], width, height) for i in keep]
        out_scores = [float(scores[i]) for i in keep]
        return out_masks, out_scores


def _to_numpy(value):
    """Detaches a torch tensor to numpy, and passes anything else through."""
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)


def _as_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """One instance mask as the uint8 0/1 image, frame-sized, the pipeline
    expects. With retina_masks the size already matches; the resize is for an
    ultralytics that ignores it, since a mask at another size would put every
    contour in the wrong place without any error."""
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask
