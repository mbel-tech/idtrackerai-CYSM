"""The frame enhancement applied before Detectron2, shared by every stage.

Enhancement is a property of a *recording setup*, not of this software. A well
lit tank with good contrast may need none at all; a turbid one under uneven
lighting may need strong CLAHE. So nothing here is hardwired: the settings live
in a profile file you keep per setup, and every tool reads it.

This has to be one module, not a copy in each script, because the model learns
the appearance of the frames it was annotated on. The frames extracted for
annotation, the frames used for training and the frames passed to the predictor
at inference must all go through the same function with the same settings. A
CLAHE value that differs between training and inference is a silent accuracy
loss that looks like a model problem, so the settings travel with the model and
inference adopts them by default.

Where settings come from, highest priority first:

1. explicit command-line flags
2. --preprocess-profile FILE
3. the settings recorded with the trained weights (inference only)
4. the built-in defaults below

The built-in defaults reproduce the pipeline this fork replaced. They are a
starting point for one particular setup, not a recommendation. Tune them
against your own footage:

    idtrackerai_d2_enhance --video clip.mp4 --frame 500 \\
        --output check.png --save-profile setups/tank_a.json

Detectron2's R-50-FPN configs expect 3-channel BGR input, so the enhanced
single-channel image is replicated across three channels by :func:`for_detectron2`.
"""

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .errors import PreprocessingError
except ImportError:  # loaded by path, e.g. from a Colab bundle with no package

    class PreprocessingError(Exception):  # type: ignore[no-redef]
        """A profile could not be read or made sense of."""


# Starting point, not a recommendation. Override per setup with a profile.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = 8
ILLUMINATION_DOWNSAMPLE = 4
ILLUMINATION_SIGMA = 25.0
ILLUMINATION_PIVOT = 128

DEFAULT_SETTINGS: dict[str, Any] = {
    "enhance": True,
    "clahe_clip": CLAHE_CLIP_LIMIT,
    "clahe_tile": CLAHE_TILE_GRID,
    "illumination_sigma": ILLUMINATION_SIGMA,
    "illumination_downsample": ILLUMINATION_DOWNSAMPLE,
    "correct_lighting": True,
    # Light edits; every default means "no change".
    "exposure": 0.0,
    "brightness": 0,
    "contrast": 0,
    "gamma": 1.0,
    "shadows": 0,
    "highlights": 0,
    "blacks": 0,
    "whites": 0,
    "sharpness": 0,
    "denoise": 0,
}

# Inclusive (min, max) per light-edit key.
LIGHT_EDIT_RANGES: dict[str, tuple[float, float]] = {
    "exposure": (-3.0, 3.0),
    "brightness": (-100, 100),
    "contrast": (-100, 100),
    "gamma": (0.3, 3.0),
    "shadows": (-100, 100),
    "highlights": (-100, 100),
    "blacks": (-100, 100),
    "whites": (-100, 100),
    "sharpness": (0, 100),
    "denoise": (0, 10),
}

SETTING_KEYS = tuple(DEFAULT_SETTINGS)


def normalize_settings(settings: dict | None) -> dict | None:
    """Fills a possibly partial settings dict from the defaults, rejecting typos.

    ``None`` stays ``None`` (no enhancement configured). A partial dict such as
    ``{"clahe_clip": 2.0}`` is completed, so a .toml that sets one key does not
    raise a KeyError deep inside a pool worker. Unknown keys are an error rather
    than silently ignored, because a misspelt key would otherwise leave the
    setting at its default without a word.
    """
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise PreprocessingError(
            f"enhancement must be a table of settings, got {type(settings).__name__}"
        )
    unknown = sorted(set(settings) - set(SETTING_KEYS))
    if unknown:
        raise PreprocessingError(
            f"Unknown enhancement setting(s): {', '.join(map(repr, unknown))}. "
            f"Valid settings are: {', '.join(SETTING_KEYS)}"
        )
    merged = {**DEFAULT_SETTINGS, **settings}
    for key in ("clahe_clip", "illumination_sigma"):
        if isinstance(merged[key], bool) or not isinstance(merged[key], (int, float)):
            raise PreprocessingError(f"enhancement '{key}' must be a number")
    for key in ("clahe_tile", "illumination_downsample"):
        if isinstance(merged[key], bool) or not isinstance(merged[key], int):
            raise PreprocessingError(f"enhancement '{key}' must be an integer")
    for key, (lo, hi) in LIGHT_EDIT_RANGES.items():
        value = merged[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise PreprocessingError(
                f"enhancement '{key}' must be a number between {lo} and {hi}"
            )
    return merged


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim > 2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def correct_illumination(
    gray: np.ndarray,
    downsample: int = ILLUMINATION_DOWNSAMPLE,
    sigma: float = ILLUMINATION_SIGMA,
    pivot: int = ILLUMINATION_PIVOT,
) -> np.ndarray:
    """Flattens uneven lighting by removing a blurred estimate of the background.

    The estimate is built on a downsampled copy, which is both faster and
    smoother than blurring at full resolution.

    The result is re-centred on ``pivot`` rather than saturated at zero. A plain
    subtraction clips everything darker than the local background to black,
    which erases exactly the contrast a dark animal on a dark background needs.
    """
    if downsample > 1:
        small = cv2.resize(
            gray,
            None,
            fx=1 / downsample,
            fy=1 / downsample,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    background = cv2.GaussianBlur(small, (0, 0), sigma / max(downsample, 1))

    if downsample > 1:
        background = cv2.resize(
            background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    return cv2.addWeighted(gray, 1.0, background, -1.0, pivot)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: int = CLAHE_TILE_GRID,
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


@functools.lru_cache(maxsize=64)
def tone_lut(
    exposure: float,
    brightness: float,
    contrast: float,
    gamma: float,
    shadows: float,
    highlights: float,
    blacks: float,
    whites: float,
) -> np.ndarray:
    """256-entry lookup table for the tone controls, identity when all are neutral.

    Cached, so the returned array is read-only.
    """
    x = np.arange(256) / 255.0

    # Levels: positive whites pull the white point in (brighter whites) and
    # positive blacks push the black point out (lifted blacks).
    black_in = -0.5 * blacks / 100
    white_in = max(1 - 0.5 * whites / 100, black_in + 0.05)
    x = np.clip((x - black_in) / (white_in - black_in), 0, 1)
    x = np.clip(x * 2.0**exposure, 0, 1)
    x = np.clip(x ** (1 / gamma), 0, 1)
    # Shadow/highlight bumps. 0.25 is the largest amplitude that keeps the
    # curve monotonic.
    x = np.clip(x + 0.25 * shadows / 100 * np.where(x < 0.5, (1 - 2 * x) ** 2, 0), 0, 1)
    x = np.clip(
        x + 0.25 * highlights / 100 * np.where(x > 0.5, (2 * x - 1) ** 2, 0), 0, 1
    )
    x = np.clip(0.5 + (x - 0.5) * (1 + contrast / 100), 0, 1)
    x = np.clip(x + brightness / 100, 0, 1)

    lut = np.round(x * 255).astype(np.uint8)
    lut.flags.writeable = False
    return lut


def denoise(gray: np.ndarray, strength: float) -> np.ndarray:
    """Edge-preserving smoothing; ``strength`` 0 returns the input unchanged."""
    if strength <= 0:
        return gray
    return cv2.bilateralFilter(gray, 5, 8 * strength, strength)


_denoise = denoise  # enhance() has a parameter of the same name


def sharpen(gray: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask; ``amount`` 0 returns the input unchanged."""
    if amount <= 0:
        return gray
    a = amount / 50
    return cv2.addWeighted(gray, 1 + a, cv2.GaussianBlur(gray, (0, 0), 2.0), -a, 0)


def enhance(
    frame: np.ndarray,
    clahe_clip: float = CLAHE_CLIP_LIMIT,
    clahe_tile: int = CLAHE_TILE_GRID,
    downsample: int = ILLUMINATION_DOWNSAMPLE,
    sigma: float = ILLUMINATION_SIGMA,
    correct_lighting: bool = True,
    *,
    exposure: float = 0.0,
    brightness: float = 0,
    contrast: float = 0,
    gamma: float = 1.0,
    shadows: float = 0,
    highlights: float = 0,
    blacks: float = 0,
    whites: float = 0,
    sharpness: float = 0,
    denoise: float = 0,
) -> np.ndarray:
    """Full enhancement, returning a single-channel uint8 image.

    Order: denoise, lighting evenness, tone, CLAHE, sharpen. A step left at its
    default is skipped without touching the pixels.
    """
    gray = to_gray(frame)
    gray = _denoise(gray, denoise)
    if correct_lighting:
        gray = correct_illumination(gray, downsample, sigma)
    tone = (exposure, brightness, contrast, gamma, shadows, highlights, blacks, whites)
    if tone != (0, 0, 0, 1.0, 0, 0, 0, 0):
        gray = cv2.LUT(gray, tone_lut(*tone))
    gray = apply_clahe(gray, clahe_clip, clahe_tile)
    return sharpen(gray, sharpness)


def enhance_with_settings(frame: np.ndarray, settings: dict) -> np.ndarray:
    """``enhance`` driven by a settings dict; missing keys take their defaults.

    The ``enhance`` on/off flag is the caller's business and is ignored here.
    """
    s = {**DEFAULT_SETTINGS, **settings}
    return enhance(
        frame,
        clahe_clip=s["clahe_clip"],
        clahe_tile=s["clahe_tile"],
        downsample=s["illumination_downsample"],
        sigma=s["illumination_sigma"],
        correct_lighting=s["correct_lighting"],
        **{k: s[k] for k in LIGHT_EDIT_RANGES},
    )


def for_detectron2(frame: np.ndarray, **kwargs) -> np.ndarray:
    """Enhanced frame as 3-channel BGR, which is what the R-50-FPN configs want."""
    return cv2.cvtColor(enhance(frame, **kwargs), cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ profiles
def load_profile(path: Path | str) -> dict:
    """Reads a profile file, rejecting unknown or malformed keys loudly.

    A typo'd key silently falling back to a default is exactly the kind of
    mismatch this module exists to prevent, so it is an error instead.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PreprocessingError(f"Preprocessing profile not found: {path}")
    except json.JSONDecodeError as exc:
        raise PreprocessingError(
            f"Preprocessing profile {path} is not valid JSON: {exc}"
        )

    if not isinstance(data, dict):
        raise PreprocessingError(
            f"Preprocessing profile {path} must contain a JSON object"
        )

    settings = data.get("enhancement", data)
    unknown = set(settings) - set(SETTING_KEYS) - {"name", "notes"}
    if unknown:
        raise PreprocessingError(
            f"Preprocessing profile {path} has unknown key(s): {sorted(unknown)}.\n"
            f"Valid keys: {list(SETTING_KEYS)}"
        )
    return {k: v for k, v in settings.items() if k in SETTING_KEYS}


def save_profile(
    path: Path | str, settings: dict, name: str = "", notes: str = ""
) -> Path:
    """Writes a profile, so a setup's settings can be reused and version controlled."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if name:
        payload["name"] = name
    if notes:
        payload["notes"] = notes
    payload.update({k: settings[k] for k in SETTING_KEYS})
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the enhancement options, so every tool exposes the same names.

    Every flag defaults to None so that "not given" is distinguishable from
    "given the default value". That is what lets inference adopt the settings
    recorded at training time unless you deliberately override them.
    """
    group = parser.add_argument_group(
        "frame enhancement",
        "Per-setup image preparation. Defaults come from --preprocess-profile,"
        " or from the settings recorded with the model, or from the built-ins.",
    )
    group.add_argument(
        "--preprocess-profile",
        type=Path,
        help="JSON file of enhancement settings for this recording setup",
    )
    group.add_argument(
        "--no-enhance",
        action="store_true",
        default=None,
        help="feed raw frames to the model, for footage that needs no help",
    )
    group.add_argument(
        "--enhance",
        dest="force_enhance",
        action="store_true",
        default=None,
        help="enhance even if the profile or the model's record disables it",
    )
    group.add_argument("--clahe-clip", type=float, default=None)
    group.add_argument("--clahe-tile", type=int, default=None)
    group.add_argument("--illumination-sigma", type=float, default=None)
    group.add_argument("--illumination-downsample", type=int, default=None)
    group.add_argument(
        "--no-illumination-correction",
        action="store_true",
        default=None,
        help="apply CLAHE only, leaving uneven lighting in place",
    )
    for key, (low, high) in LIGHT_EDIT_RANGES.items():
        group.add_argument(
            f"--{key}",
            type=float,
            default=None,
            help=f"light edit, from {low:g} to {high:g} (default: {DEFAULT_SETTINGS[key]:g})",
        )


def settings_to_flags(settings: dict | None) -> list[str]:
    """The command-line flags that reproduce ``settings`` from the built-ins.

    The inverse of :func:`resolve_settings` with no profile and no fallback:
    only what differs from the defaults is written, so a command for default
    settings stays short. With enhancement off, nothing else matters.
    """
    settings = normalize_settings(settings or {})
    assert settings is not None
    if not settings["enhance"]:
        return ["--no-enhance"]
    flags: list[str] = []
    if not settings["correct_lighting"]:
        flags.append("--no-illumination-correction")
    for key in (
        "clahe_clip",
        "clahe_tile",
        "illumination_sigma",
        "illumination_downsample",
        *LIGHT_EDIT_RANGES,
    ):
        if settings[key] != DEFAULT_SETTINGS[key]:
            # str, not :g, which rounds to 6 digits and would not parse back
            flags += [f"--{key.replace('_', '-')}", str(settings[key])]
    return flags


def _explicit_from_args(args) -> dict:
    """Only the settings the user actually typed."""
    explicit: dict[str, Any] = {}
    for flag, key in (
        ("clahe_clip", "clahe_clip"),
        ("clahe_tile", "clahe_tile"),
        ("illumination_sigma", "illumination_sigma"),
        ("illumination_downsample", "illumination_downsample"),
        *((key, key) for key in LIGHT_EDIT_RANGES),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            explicit[key] = value

    if getattr(args, "no_enhance", None):
        explicit["enhance"] = False
    elif getattr(args, "force_enhance", None):
        explicit["enhance"] = True

    if getattr(args, "no_illumination_correction", None):
        explicit["correct_lighting"] = False
    return explicit


def resolve_settings(
    args, fallback: dict | None = None, fallback_label: str = ""
) -> tuple[dict, list[str]]:
    """Layers flags over a profile over a fallback over the built-in defaults.

    Returns the settings and a list of human-readable notes describing where
    each layer came from, which the tools print so the choice is never silent.
    """
    notes: list[str] = []
    settings = dict(DEFAULT_SETTINGS)
    source = "built-in defaults"

    if fallback:
        usable = {k: v for k, v in fallback.items() if k in SETTING_KEYS}
        if usable:
            settings.update(usable)
            source = fallback_label or "recorded settings"

    profile_path = getattr(args, "preprocess_profile", None)
    if profile_path:
        settings.update(load_profile(profile_path))
        notes.append(f"enhancement profile: {profile_path}")
        source = str(profile_path)
    else:
        notes.append(f"enhancement settings from {source}")

    explicit = _explicit_from_args(args)
    if explicit:
        settings.update(explicit)
        notes.append(
            "overridden on the command line: "
            + ", ".join(f"{k}={v}" for k, v in sorted(explicit.items()))
        )

    if not settings["enhance"]:
        notes.append("enhancement is OFF; raw frames are used")
    return normalize_settings(settings), notes


def settings_from_args(args) -> dict:
    """The settings as a dict, for recording alongside outputs."""
    settings, _ = resolve_settings(args)
    return settings


def make_enhancer_from(settings: dict):
    """Builds the callable each tool applies to every frame."""
    if not settings.get("enhance", True):
        return lambda frame: (
            frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        )

    def enhancer(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(enhance_with_settings(frame, settings), cv2.COLOR_GRAY2BGR)

    return enhancer


def make_enhancer(args):
    """Convenience wrapper for tools that do not need a fallback layer."""
    return make_enhancer_from(settings_from_args(args))


def describe(settings: dict) -> str:
    if not settings.get("enhance", True):
        return "none (raw frames)"
    parts = [f"CLAHE clip={settings['clahe_clip']} tile={settings['clahe_tile']}"]
    if settings["correct_lighting"]:
        parts.append(
            f"illumination sigma={settings['illumination_sigma']}"
            f" downsample={settings['illumination_downsample']}"
        )
    else:
        parts.append("no illumination correction")
    edits = [
        f"{key}={settings[key]:+g}"
        for key in LIGHT_EDIT_RANGES
        if key in settings and settings[key] != DEFAULT_SETTINGS[key]
    ]
    if edits:
        parts.append(", ".join(edits))
    return "; ".join(parts)


def check_settings_match(
    recorded: dict | None, current: dict, label: str
) -> str | None:
    """Compares enhancement settings, returning a warning when they differ.

    Used to catch the case where a model trained on enhanced frames is later run
    on raw ones, which degrades detection without any error being raised.
    """
    if not recorded:
        return None
    differences = [
        f"{key}: trained with {was!r}, now {current[key]!r}"
        for key in current
        if key in DEFAULT_SETTINGS
        for was in [recorded.get(key, DEFAULT_SETTINGS[key])]
        if was != current[key]
    ]
    if not differences:
        return None
    return (
        f"Frame enhancement differs from {label}:\n  "
        + "\n  ".join(differences)
        + "\nThe model sees different-looking images than it was trained on."
    )


def _demo():
    """Writes a before/after strip, to tune the settings on real footage.

    Use this to pick settings for a new recording setup, then --save-profile to
    keep them.
    """
    parser = argparse.ArgumentParser(
        description="Preview and save enhancement settings for a recording setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("enhancement_demo.png"))
    parser.add_argument(
        "--save-profile",
        type=Path,
        help="write these settings to a profile file for reuse",
    )
    parser.add_argument("--name", default="", help="a label for the saved profile")
    parser.add_argument("--notes", default="", help="a note for the saved profile")
    add_arguments(parser)
    args = parser.parse_args()

    settings, notes = resolve_settings(args)
    for note in notes:
        print(note)
    print(f"enhancement: {describe(settings)}")

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise PreprocessingError(f"Could not read frame {args.frame} of {args.video}")

    gray = to_gray(frame)
    if settings["enhance"]:
        strip = np.hstack([gray, enhance_with_settings(frame, settings)])
        caption = "grayscale | enhanced"
    else:
        strip = gray
        caption = "grayscale only (enhancement off)"

    cv2.imencode(".png", strip)[1].tofile(str(args.output))
    print(f"Wrote {args.output} ({caption})")

    if args.save_profile:
        path = save_profile(args.save_profile, settings, args.name, args.notes)
        print(f"Saved profile to {path}")
        print(f"Use it with: --preprocess-profile {path}")


if __name__ == "__main__":
    _demo()
