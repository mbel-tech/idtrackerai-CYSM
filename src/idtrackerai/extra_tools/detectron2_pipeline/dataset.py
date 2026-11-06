"""Turning LabelMe polygons into a COCO instance-segmentation dataset.

Annotations are validated rather than trusted. LabelMe will happily save a
two-point "polygon", a shape drawn outside the image, or a label typo that
silently becomes a second class, and each of those either crashes training or
quietly degrades it. Everything found is reported rather than raised, so the
caller — a terminal or a GUI panel — can show the whole picture at once.

The split defaults to grouping by source video. Frames sampled from one clip a
few seconds apart show the same animals in nearly the same poses, so a random
split puts near-duplicates in both train and validation, and the validation AP
then measures memorisation rather than generalisation.
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .errors import DatasetError
except ImportError:  # loaded by path, without the package around it
    from errors import DatasetError  # type: ignore[no-redef]

# Sidecars written by the sampling step, which are not annotations.
RESERVED_JSON = {
    "sampling_manifest.json",
    "preprocess_profile.json",
    "report.json",
}


@dataclass
class DatasetRequest:
    input_dir: Path
    output_dir: Path
    val_fraction: float = 0.15
    split_by: str = "video"
    seed: int = 0
    group_by: str = "recording"
    """How clips are grouped when ``split_by="video"``.

    ``"recording"`` folds the pieces of one recording together (the segments of
    a trial), ``"file"`` treats every clip separately, which is what this did
    before grouping existed.
    """
    group_overrides: dict[str, str] | None = None
    """Corrections to the inferred grouping, as ``{clip stem: group}``."""
    labels: Sequence[str] | None = None
    single_class: str | None = None
    expected_instances: int | None = None
    min_area: float = 10.0
    copy_images: bool = True


@dataclass
class DatasetReport:
    categories: list[dict] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    train_images: int = 0
    train_annotations: int = 0
    train_videos: list[str] = field(default_factory=list)
    val_images: int = 0
    val_annotations: int = 0
    val_videos: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped_json: list[str] = field(default_factory=list)
    enhancement: dict | None = None
    output_dir: Path | None = None

    def as_dict(self) -> dict:
        return {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "enhancement": self.enhancement,
            "categories": self.categories,
            "label_counts": self.label_counts,
            "train": {
                "images": self.train_images,
                "annotations": self.train_annotations,
                "videos": self.train_videos,
            },
            "val": {
                "images": self.val_images,
                "annotations": self.val_annotations,
                "videos": self.val_videos,
            },
            "problems": self.problems,
            "notes": self.notes,
        }


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area. Not cv2.contourArea, so this works without OpenCV."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def is_labelme(path: Path) -> bool:
    """Recognises an annotation by shape, not by filename.

    Blacklisting names would break the moment another sidecar is added, so a
    file counts as an annotation when it parses as an object with "shapes". A
    file that will not parse counts as one too, so the reader reports it as a
    broken annotation rather than silently ignoring it.
    """
    if path.name in RESERVED_JSON:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    return isinstance(data, dict) and "shapes" in data


def count_annotated(folder: Path) -> int:
    """How many annotations sit in a folder. Used by the GUI's progress display."""
    return sum(1 for p in folder.glob("*.json") if is_labelme(p))


# Suffixes stripped from a recorded source so that a manifest written before
# the spelling was settled still names the same clip as the filename fallback.
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")

# Trailing name parts that mark a piece of a recording rather than a recording.
SEGMENT_WORDS = ("segment", "seg", "part", "clip", "chunk", "piece", "vid", "video")

# Trailing name parts left behind by re-encoding or tidying a file, which do
# not make it a different recording.
REENCODE_WORDS = frozenset(
    ("cleaned", "clean", "fixed", "trimmed", "trim", "raw",
     "edited", "edit", "copy", "final", "new", "old")
)


def normalise_source(source: str) -> str:
    """One spelling for a clip, whatever spelling was recorded."""
    name = source.strip()
    lowered = name.lower()
    for suffix in VIDEO_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def source_video_of(file_name: str, manifest: dict | None) -> str:
    """Which clip a frame came from, for the group-aware split."""
    if manifest:
        entry = manifest.get(file_name)
        if entry and entry.get("source_video"):
            return normalise_source(entry["source_video"])
    # fall back to the naming the sampling step uses: <video stem>_f<number>.png
    stem = Path(file_name).stem
    if "_f" in stem:
        return normalise_source(stem.rsplit("_f", 1)[0])
    return normalise_source(stem)


def _recorded_frames(manifest_path: Path) -> dict | None:
    """The manifest's frames by file name, or None when it cannot be trusted.

    The preview must never be the thing that fails, so a manifest that is
    missing, unreadable or the wrong shape is simply not used. Entries that do
    not name their source are left out, and those frames fall back to their
    file names like frames with no manifest at all.
    """
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list):
        return None
    return {
        entry["file_name"]: entry
        for entry in frames
        if isinstance(entry, dict)
        and isinstance(entry.get("file_name"), str)
        and isinstance(entry.get("source_video"), str)
    }


def sources_in_folder(input_dir: Path) -> list[str]:
    """The clips a folder of frames came from, as the split will see them.

    Each image is named by the manifest when it records the frame, else by its
    file name, exactly as ``build_coco_dataset`` does. Feed the result to
    ``group_videos`` to preview the grouping before anything is built.
    """
    manifest = _recorded_frames(input_dir / "sampling_manifest.json")
    return sorted(
        {
            source_video_of(path.name, manifest)
            for path in input_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
        }
    )


def group_key(stem: str) -> str:
    """The recording a clip belongs to: B3_N1_segment_3 -> B3_N1.

    Recordings are routinely saved in pieces, and the pieces show the same
    animals in the same arena minutes apart. Splitting train from validation
    between two pieces of one recording is the same near-duplicate leak that
    splitting within a single clip would be, so the pieces are folded together.

    This is a guess made from a file name, and it can be wrong -- clips
    genuinely named trial_1 and trial_2 would be merged. The caller is expected
    to show what was inferred and let it be corrected, rather than apply it
    silently.
    """
    name = normalise_source(stem)
    kept = name.split("_")
    while len(kept) > 1:
        token = kept[-1].lower()
        bare = token.rstrip("0123456789")
        if (
            token.isdigit()
            or token in REENCODE_WORDS
            or token in SEGMENT_WORDS
            or (bare in SEGMENT_WORDS and bare != token)
        ):
            kept.pop()
        else:
            break

    # "clip_00" and "clip_01" are two recordings, not two pieces of one called
    # "clip". If nothing survives but the word marking a piece, the name never
    # carried a recording name, so there is nothing to group by: leave it.
    if all(t.lower().rstrip("0123456789") in SEGMENT_WORDS for t in kept):
        return name
    return "_".join(kept)


def group_of(source: str, overrides: dict[str, str] | None, group_by: str) -> str:
    """The split group for a clip, honouring any correction the user made."""
    source = normalise_source(source)
    if overrides and source in overrides:
        return overrides[source]
    if group_by == "recording":
        return group_key(source)
    return source


def group_videos(sources: Sequence[str], overrides: dict[str, str] | None = None,
                 group_by: str = "recording") -> dict[str, list[str]]:
    """The inferred grouping, for showing before anything is built."""
    groups: dict[str, list[str]] = defaultdict(list)
    for source in sorted({normalise_source(s) for s in sources}):
        groups[group_of(source, overrides, group_by)].append(source)
    return dict(groups)


def read_enhancement(input_dir: Path) -> dict | None:
    """The enhancement settings the frames were prepared with.

    Training records these with the weights so inference can reproduce exactly
    the images the model saw. If they do not make it this far the chain breaks
    silently: training records null, inference falls back to built-in defaults,
    and the mismatch check has nothing to compare against so it cannot fire.
    """
    profile = input_dir / "preprocess_profile.json"
    if profile.is_file():
        data = _read_json_object(profile)
        return {k: v for k, v in data.items() if k not in ("name", "notes")}
    manifest = input_dir / "sampling_manifest.json"
    if manifest.is_file():
        return _read_json_object(manifest).get("enhancement")
    return None


def _read_json_object(path: Path) -> dict:
    """A JSON file's top-level object, or a DatasetError saying what is wrong."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{path} could not be read as JSON: {exc}")
    if not isinstance(data, dict):
        raise DatasetError(f"{path} should contain a JSON object")
    return data


def _image_name_of(data: dict, json_path: Path) -> str:
    """The image a LabelMe file points at, as a bare file name.

    LabelMe on Windows stores ``imagePath`` with backslashes, which
    ``Path(...).name`` does not split on Linux (Colab), so they are normalised
    first. A missing or empty path falls back to the JSON's own name.
    """
    raw = data.get("imagePath")
    name = str(raw).replace("\\", "/").rsplit("/", 1)[-1] if raw else ""
    return name or (json_path.stem + ".png")


def build_coco_dataset(
    request: DatasetRequest,
    progress: Callable[[int], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> DatasetReport:
    """Validates, converts and splits. Everything found lands in the report."""
    report = DatasetReport(output_dir=request.output_dir)
    all_json = sorted(request.input_dir.glob("*.json"))
    json_files = [p for p in all_json if is_labelme(p)]
    report.skipped_json = [p.name for p in all_json if not is_labelme(p)]

    if not json_files:
        raise DatasetError(f"No LabelMe .json files in {request.input_dir}")

    manifest = None
    manifest_path = request.input_dir / "sampling_manifest.json"
    if manifest_path.is_file():
        data = _read_json_object(manifest_path)
        try:
            manifest = {f["file_name"]: f for f in data["frames"]}
        except (KeyError, TypeError) as exc:
            raise DatasetError(
                f"{manifest_path} is malformed (missing {exc}); delete it or"
                " sample again"
            )

    report.enhancement = read_enhancement(request.input_dir)

    label_counter: Counter[str] = Counter()
    per_image: list[dict] = []

    # ------------------------------------------------------ read and validate
    for done, json_path in enumerate(json_files, start=1):
        if abort is not None and abort():
            raise DatasetError("Cancelled")
        if progress is not None:
            progress(done)

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            report.problems.append(f"{json_path.name}: not valid JSON ({exc})")
            continue
        if not isinstance(data, dict):
            report.problems.append(f"{json_path.name}: not a LabelMe object")
            continue

        image_name = _image_name_of(data, json_path)
        image_path = request.input_dir / image_name
        if not image_path.is_file():
            report.problems.append(
                f"{json_path.name}: image {image_name} not found beside it"
            )
            continue

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            report.problems.append(f"{json_path.name}: missing imageWidth/imageHeight")
            continue

        shapes = []
        for i, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "")
            kind = shape.get("shape_type", "polygon")
            points = np.asarray(shape.get("points", []), dtype=float)

            if kind != "polygon":
                report.problems.append(
                    f"{json_path.name} shape {i}: shape_type is {kind!r}, expected"
                    " 'polygon'. Instance segmentation needs outlines, not boxes."
                )
                continue
            if len(points) < 3:
                report.problems.append(
                    f"{json_path.name} shape {i} ({label}): only {len(points)}"
                    " point(s), a polygon needs 3"
                )
                continue

            area = polygon_area(points)
            if area < request.min_area:
                report.problems.append(
                    f"{json_path.name} shape {i} ({label}): area {area:.1f} below"
                    f" the minimum {request.min_area}, probably a stray click"
                )
                continue

            outside = (
                (points[:, 0] < 0).any()
                or (points[:, 1] < 0).any()
                or (points[:, 0] > width).any()
                or (points[:, 1] > height).any()
            )
            if outside:
                report.notes.append(
                    f"{json_path.name} shape {i} ({label}): extends past the image"
                    " border, clipped"
                )
                points[:, 0] = points[:, 0].clip(0, width)
                points[:, 1] = points[:, 1].clip(0, height)

            label_counter[label] += 1
            shapes.append({"label": label, "points": points, "area": area})

        if not shapes:
            report.problems.append(
                f"{json_path.name}: no usable polygons, image excluded"
            )
            continue

        if request.expected_instances and len(shapes) != request.expected_instances:
            report.notes.append(
                f"{json_path.name}: {len(shapes)} instance(s), expected"
                f" {request.expected_instances}"
            )

        per_image.append(
            {
                "file_name": image_name,
                "path": image_path,
                "width": int(width),
                "height": int(height),
                "shapes": shapes,
                "source_video": source_video_of(image_name, manifest),
                "group": group_of(
                    source_video_of(image_name, manifest),
                    request.group_overrides,
                    request.group_by,
                ),
            }
        )

    report.label_counts = dict(label_counter)

    if not per_image:
        raise DatasetError(
            "No usable annotations found.\n  " + "\n  ".join(report.problems[:20])
        )

    # ------------------------------------------------------------- categories
    if request.single_class:
        categories = [{"id": 1, "name": request.single_class}]

        def category_of(_label):
            return 1

    else:
        names = list(request.labels or sorted(label_counter))
        unknown = set(label_counter) - set(names)
        if unknown:
            report.problems.append(
                "labels present but not requested, their shapes are dropped:"
                f" {sorted(unknown)}"
            )
        categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
        ids = {n: i + 1 for i, n in enumerate(names)}
        category_of = ids.get

    if len(label_counter) > 1 and not request.single_class and not request.labels:
        report.notes.append(
            f"{len(label_counter)} distinct labels found: {dict(label_counter)}."
            " If these are meant to be one class, set a single class name."
        )
    report.categories = categories

    # ------------------------------------------------------------------ split
    train_images, val_images = _split(per_image, request, report)

    # ------------------------------------------------------------------ write
    request.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = request.output_dir / "images"
    if request.copy_images:
        images_dir.mkdir(exist_ok=True)

    train = _build_coco(train_images, "train", categories, category_of,
                        images_dir if request.copy_images else None)
    val = _build_coco(val_images, "val", categories, category_of,
                      images_dir if request.copy_images else None)
    if request.copy_images:
        # a rebuilt dataset must not keep images the annotations dropped
        wanted = {i["file_name"] for i in train_images + val_images}
        for stale in images_dir.iterdir():
            if (
                stale.is_file()
                and stale.suffix.lower() in IMAGE_SUFFIXES
                and stale.name not in wanted
            ):
                stale.unlink()

    (request.output_dir / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (request.output_dir / "val.json").write_text(json.dumps(val), encoding="utf-8")

    report.train_images = len(train["images"])
    report.train_annotations = len(train["annotations"])
    report.train_videos = sorted({i["group"] for i in train_images})
    report.val_images = len(val["images"])
    report.val_annotations = len(val["annotations"])
    report.val_videos = sorted({i["group"] for i in val_images})

    (request.output_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )

    # Copy the profile itself too, so the settings travel when only the dataset
    # folder is uploaded to Colab.
    profile = request.input_dir / "preprocess_profile.json"
    if profile.is_file():
        shutil.copy2(profile, request.output_dir / "preprocess_profile.json")
    elif report.enhancement is None:
        report.problems.append(
            "No enhancement record found beside the annotations. Training will"
            " not know how these frames were prepared, and inference will fall"
            " back to the built-in defaults without warning."
        )

    if not report.val_images:
        report.problems.append(
            "The validation set is empty, so training cannot be evaluated."
        )
    return report


def _split(per_image, request: DatasetRequest, report: DatasetReport):
    rng = random.Random(request.seed)

    if request.split_by != "video":
        report.notes.append(
            "Random split requested. If these frames come from the same clips,"
            " train and validation will contain near-duplicate images and the"
            " validation AP will overstate real performance."
        )
        return _random_split(per_image, request.val_fraction, rng)

    by_video: dict[str, list[dict]] = defaultdict(list)
    for image in per_image:
        by_video[image["group"]].append(image)

    # Grouping is inferred from file names, and names do not always say whether
    # "trial_1" and "trial_2" are one recording in two pieces or two
    # recordings. When the guess collapses everything into a single group there
    # is nothing left to split on, and per-clip grouping -- weaker, but real --
    # beats falling back to a random split.
    sources = {image["source_video"] for image in per_image}
    if len(by_video) < 2 <= len(sources):
        report.notes.append(
            f"Grouping by recording put all {len(sources)} clips in one group"
            f" ({sorted(by_video)[0]}), which leaves nothing to split on, so"
            " the split is by clip instead. If these clips really are pieces of"
            " one recording, the validation score will be optimistic."
        )
        by_video = defaultdict(list)
        for image in per_image:
            # rewrite the group itself, so the report names what was actually
            # split on rather than the grouping that was abandoned
            image["group"] = image["source_video"]
            by_video[image["source_video"]].append(image)

    videos = sorted(by_video)
    rng.shuffle(videos)

    singletons = sum(1 for v in videos if len(by_video[v]) == 1)
    if singletons == len(videos) and len(videos) > 2:
        report.notes.append(
            f"Every one of the {len(videos)} frames was attributed to a"
            " different source, so grouping them achieves nothing and this is"
            " in effect a random split. That usually means the frames were not"
            " named by the sampling step, so their source could not be read."
        )

    if len(videos) < 2:
        report.notes.append(
            f"All frames come from one source ({videos[0]}), so a by-video split"
            " is impossible; falling back to a random split. The validation"
            " score will be optimistic because neighbouring frames are"
            " near-duplicates."
        )
        return _random_split(per_image, request.val_fraction, rng)

    # Whole groups go to one side or the other, so the validation set can only
    # land on group boundaries. Take a group only while doing so gets closer to
    # the target than stopping would: adding it regardless, which is what this
    # did before, overshot a requested 15% to 26% on sixteen uneven recordings.
    target = len(per_image) * request.val_fraction
    # Start from the group nearest the target, not whichever the shuffle put
    # first: that one is taken unconditionally, so an oversized group would
    # swallow the whole validation budget.
    candidates = [v for v in videos if len(by_video[v]) <= len(per_image) - 1]
    if candidates:
        first = min(candidates, key=lambda v: abs(len(by_video[v]) - target))
        videos.remove(first)
        videos.insert(0, first)
    val_images: list[dict] = []
    train_images: list[dict] = []
    taken = 0
    for video in videos:
        group = by_video[video]
        fits = len(val_images) + len(group) <= len(per_image) - 1
        closer = abs(taken + len(group) - target) < abs(taken - target)
        if fits and (taken == 0 or closer):
            val_images += group
            taken += len(group)
        else:
            train_images += group

    achieved = len(val_images) / len(per_image) if per_image else 0.0
    if abs(achieved - request.val_fraction) > 0.05:
        report.notes.append(
            f"Validation is {achieved:.0%} of the frames rather than the"
            f" {request.val_fraction:.0%} requested. Whole recordings go to one"
            " side, so the split can only land on a recording boundary."
        )
    return train_images, val_images


def _random_split(per_image, val_fraction: float, rng: random.Random):
    images = list(per_image)
    rng.shuffle(images)
    cut = max(1, round(len(images) * val_fraction))
    return images[cut:], images[:cut]


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _same_file(source: Path, target: Path) -> bool:
    """True when target is an up-to-date copy of source (copy2 keeps mtime)."""
    if not target.exists():
        return False
    a, b = source.stat(), target.stat()
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 2


def _build_coco(subset, name, categories, category_of, images_dir: Path | None) -> dict:
    coco: dict = {
        "info": {
            "description": f"{name} set from LabelMe annotations",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    annotation_id = 1
    for image_id, image in enumerate(subset, start=1):
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
        )
        for shape in image["shapes"]:
            category_id = category_of(shape["label"])
            if category_id is None:
                continue
            points = shape["points"]
            x0, y0 = points.min(0)
            x1, y1 = points.max(0)
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [points.flatten().round(2).tolist()],
                    "area": round(shape["area"], 2),
                    "bbox": [
                        round(float(x0), 2),
                        round(float(y0), 2),
                        round(float(x1 - x0), 2),
                        round(float(y1 - y0), 2),
                    ],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        if images_dir is not None:
            target = images_dir / image["file_name"]
            if not _same_file(image["path"], target):
                shutil.copy2(image["path"], target)
    return coco
