"""SAM 3 as a segmentation backend, prompted with a word for the animal.

Meta's SAM 3 segments what a short text prompt describes, so it needs neither
annotation nor training. That is its whole appeal and also its limitation: it
has never seen your tank, so it is weakest exactly where thresholding is weak,
on low-contrast animals that look like each other. Treat it as the option for
getting started, for clips a trained model has not been built for, and for
pre-labelling frames you will correct and train on.

Installation
------------
The runtime is Meta's own ``sam3`` package, not ``ultralytics``::

    pip install sam3

Both can load this checkpoint. ``sam3`` is preferred here because
``ultralytics`` is AGPL-3.0, which is a licence this fork would rather not pull
into a scientific pipeline it distributes. Note that SAM 3's own weights carry
Meta's "SAM License", which has its own terms: read them before publishing
results, and see ``NOTICE``.

Like Detectron2, ``sam3`` and ``torch`` are deliberately not declared in
``pyproject.toml``. They are heavy, they are GPU-bound, and idtracker.ai itself
must keep working without them, so everything imports inside the functions that
need it.

Weights
-------
``sam3.pt`` is the official checkpoint from the gated ``facebook/sam3``
repository on Hugging Face; access has to be requested there once. It is a flat
state dict of ``detector.*`` and ``tracker.*`` tensors, about 3.4 GB in fp32,
and it is passed by path like any other weights file. It never belongs in the
repository.

Per frame, not per video
------------------------
SAM 3 can also track through a whole video, carrying object identity across
frames in its memory bank. That is deliberately not used here. The sidecar
format stores contours and nothing else — it has no identity field, and
``ExternalContours`` reads only vertices and offsets — so SAM 3's track IDs
would be computed and then thrown away. Prompting each frame independently
keeps the sequential single-pass read that makes the exporter fast, and leaves
identity where this fork wants it: in idtracker.ai's crossing detection and
fragmentation, which reason over the whole video rather than the previous
frame.

If masks turn out to break down through occlusions, SAM 3's video mode is the
escalation, and it would mean buffering frames rather than streaming them.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from . import segmentation_model as sm
except ImportError:  # loaded by path, e.g. from a Colab bundle
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import segmentation_model as sm  # type: ignore[no-redef]


class Sam3WeightsError(RuntimeError):
    """The SAM 3 checkpoint could not be fetched from Hugging Face."""


def resolve_checkpoint(weights) -> Path:
    """The local sam3.pt to load: the path given, or the official one fetched.

    ``"huggingface"`` (any case) downloads sam3.pt from the gated
    facebook/sam3 repository into the Hugging Face cache, once, and returns the
    cached file. Everything downstream -- hashing, provenance, loading -- then
    sees an ordinary file, so a fetched checkpoint and a chosen one behave the
    same. Anything else is taken as a path and returned unchanged.

    The download needs a token from an account that was granted access. It is
    read the usual Hugging Face ways: ``HF_TOKEN``, or ``hf auth login``.
    """
    if str(weights).strip().lower() != sm.HUGGINGFACE:
        return Path(weights)

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise Sam3WeightsError(
            'weights = "huggingface" needs the huggingface_hub package:'
            "  pip install huggingface_hub"
        ) from exc

    print(
        f"Fetching {sm.SAM3_CHECKPOINT} from Hugging Face ({sm.HUGGINGFACE_REPO});"
        " about 3.4 GB the first time, then read from the cache."
    )
    try:
        path = hf_hub_download(repo_id=sm.HUGGINGFACE_REPO, filename=sm.SAM3_CHECKPOINT)
    except Exception as exc:  # noqa: BLE001 - every failure gets the same advice
        raise Sam3WeightsError(
            f"Could not download {sm.SAM3_CHECKPOINT} from Hugging Face: {exc}\n\n"
            f"The checkpoint is gated. Request access once at"
            f" https://huggingface.co/{sm.HUGGINGFACE_REPO}, then give this machine"
            " a token from that account: run  hf auth login  or set HF_TOKEN."
            " Without a network connection, download sam3.pt elsewhere and"
            " choose the file instead."
        ) from exc
    return Path(path)


def resolve_device(requested: str) -> str:
    """Picks a device, preferring the one asked for but never assuming CUDA.

    ``--device`` defaults to ``cuda`` for the Detectron2 path, which is
    reasonable on Colab and wrong on a laptop. Silently producing a CUDA error
    several minutes into a run is worse than saying so up front.
    """
    import torch  # noqa: PLC0415  (heavy, optional, GPU-only dependency)

    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "No CUDA device available; falling back to CPU. SAM 3 on CPU is "
            "slow enough that it is only practical for pre-labelling a few "
            "hundred frames, not for exporting a whole video."
        )
        return "cpu"
    return requested


class Sam3Predictor:
    """Segments one frame at a time from a text prompt.

    Exposes the same ``predict(frame) -> (masks, scores)`` interface as
    ``Detectron2Predictor``, so the export loop does not know which it has.
    """

    def __init__(
        self,
        checkpoint,
        prompt: str,
        device: str = "cuda",
        score_threshold: float = 0.5,
        max_instances: int = 0,
    ):
        from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415

        checkpoint = resolve_checkpoint(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM 3 checkpoint not found: {checkpoint}. Download sam3.pt "
                "from the facebook/sam3 repository on Hugging Face (access "
                "must be requested) and pass it with --weights, or pass"
                " --weights huggingface to have it fetched."
            )

        self.device = resolve_device(device)
        self.prompt = prompt
        self.score_threshold = score_threshold
        self.max_instances = max_instances

        print(f"Model: SAM 3, prompt {prompt!r}, on {self.device}")
        # load_from_HF=False keeps this offline: the checkpoint we were given
        # is the one that gets loaded, rather than a second one downloaded
        # silently into a cache.
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint), load_from_HF=False, device=self.device
        )
        # The processor keeps its own device and its own confidence threshold,
        # and both of its defaults are wrong for us. device defaults to "cuda"
        # and it allocates tensors in its constructor, so leaving it out undoes
        # the CPU fallback above and crashes before the first frame.
        # confidence_threshold defaults to 0.5 and is applied *inside* the
        # model's forward pass, so leaving it out puts a floor under
        # --score-threshold: anything lower would be filtered out before we
        # ever saw it, and lowering the threshold to recover missed animals
        # would silently do nothing.
        self._processor = Sam3Processor(
            model, device=self.device, confidence_threshold=score_threshold
        )
        self.description = f"sam3 ({prompt!r})"

    def predict(self, frame: np.ndarray):
        """Returns the masks SAM 3 finds for the prompt, and their scores."""
        from PIL import Image  # noqa: PLC0415

        # SAM 3 wants RGB; OpenCV hands us BGR. Getting this backwards does not
        # raise, it just quietly segments worse, which is the hardest kind of
        # mistake to notice in a batch run.
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=self.prompt)

        masks = _to_numpy(output["masks"])
        scores = _to_numpy(output["scores"])
        if masks is None or len(masks) == 0:
            return [], []

        # The processor already filtered on this same quantity, so this keeps
        # nothing out that it let through. It stays because the threshold is
        # ours to enforce and should not depend on the processor continuing to
        # apply the value it was constructed with.
        keep = [i for i, s in enumerate(scores) if float(s) >= self.score_threshold]
        # Keep the most confident ones when more were found than there are
        # animals. The export loop's own --max-instances only counts what it
        # was given, so the trimming has to happen here.
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        if self.max_instances > 0:
            keep = keep[: self.max_instances]

        out_masks = [_as_binary_mask(masks[i]) for i in keep]
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


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Flattens one instance mask to the uint8 0/1 image the pipeline expects.

    SAM 3 returns masks with a leading channel dimension, and depending on the
    call they are probabilities rather than booleans. ``clean_mask`` downstream
    assumes a plain binary image, so normalise both here.
    """
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.dtype != np.uint8 or mask.max() > 1:
        mask = (mask > 0.5).astype(np.uint8)
    return mask.astype(np.uint8)
