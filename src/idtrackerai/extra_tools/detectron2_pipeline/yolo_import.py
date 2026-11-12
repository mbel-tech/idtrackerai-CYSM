"""Imports YOLO segmentation label files as a contour sidecar for idtracker.ai.

``--backend yolo`` of the exporter runs the model itself. This is for when it
already ran: a ``yolo segment predict ... save_txt=True save_conf=True`` over a
video leaves one text file per frame in ``labels/``, each line one animal::

    <class> <x1> <y1> <x2> <y2> ... <xn> <yn> [<confidence>]

with coordinates as fractions of the frame. This turns a folder of those into
the same ``.h5`` the exporter writes, which idtracker.ai then tracks from
through ``external_contours``.

    idtrackerai_yolo_import --video pen5.mp4 --labels runs/segment/predict/labels \\
        --output contours/pen5.h5 --weights-name best.pt

What cannot be imported
-----------------------
*Boxes.* A detection model's label files hold ``class cx cy w h``. A box is
not an outline, and idtracker.ai identifies animals from the pixels inside
one, so such files are refused.

*Centre points, or detections on only some of the frames.* idtracker.ai links
blobs from each frame to the next and needs every frame it tracks. A table of
centroids every fourth frame has neither the outlines nor the frames. Run the
segmentation model over the video instead (``idtrackerai_d2_export --backend
yolo``).

*Label files written by ``yolo track``.* Those append a track id to each line,
which cannot be told from one more coordinate. Use ``predict``; identity is
idtracker.ai's job here.

The outlines are imported as YOLO wrote them: none of the exporter's mask
clean-up or overlap handling is applied, because there are no masks here to
apply it to.
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


class YoloImportError(RuntimeError):
    """The label files cannot be turned into contours for this video."""


def read_label_file(path: Path, width: int, height: int):
    """One frame's outlines in pixels, with their scores and classes.

    A line with an even number of values after the class is a polygon; with an
    odd number, the last is the confidence (``save_conf=True``). Without it
    the score is 1.
    """
    contours: list[np.ndarray] = []
    scores: list[float] = []
    classes: list[int] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise YoloImportError(
                f"{path}, line {number}: not a YOLO label line ({exc})"
            ) from exc
        score = 1.0
        if len(values) % 2:
            score = values.pop()
        if len(values) < 6:
            raise YoloImportError(
                f"{path}, line {number}: this line holds a box, and these files"
                " hold boxes, not outlines. idtracker.ai tracks from each"
                " animal's outline, so the labels must come from a segmentation"
                " (-seg) model."
            )
        points = np.asarray(values, np.float64).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
        points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
        contours.append(np.rint(points).astype(np.int32))
        scores.append(score)
        classes.append(cls)
    return contours, scores, classes


def label_files(labels_dir: Path, stem: str) -> dict[int, Path]:
    """``<stem>_<number>.txt`` in a folder, by number."""
    pattern = re.compile(re.escape(stem) + r"_(\d+)\.txt$")
    found: dict[int, Path] = {}
    for path in Path(labels_dir).iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def import_video(
    video: Path,
    labels_dir: Path,
    output: Path,
    first_index: int = 1,
    score_threshold: float | None = None,
    max_instances: int | None = None,
    classes: list[int] | None = None,
    weights_name: str | None = None,
    progress=None,
) -> dict:
    """Writes the contour file of one video from its label files.

    ``first_index`` is the number ultralytics gave the video's first frame: 1.
    A frame without a file had no detections, which is how ultralytics writes
    it, so it is imported empty.
    """
    from idtrackerai.base.animals_detection.external_contours import (  # noqa: PLC0415
        write_contours,
    )

    video, labels_dir, output = Path(video), Path(labels_dir), Path(output)
    if not labels_dir.is_dir():
        raise YoloImportError(f"Labels folder not found: {labels_dir}")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise YoloImportError(f"Could not open {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    files = label_files(labels_dir, video.stem)
    if not files:
        raise YoloImportError(
            f"No label files for {video.name} in {labels_dir}. ultralytics names"
            f" them {video.stem}_<frame>.txt; check the folder, and that the"
            " prediction was run on this video with save_txt=True."
        )
    beyond = sorted(n - first_index for n in files if not 0 <= n - first_index < n_frames)
    if beyond:
        raise YoloImportError(
            f"{labels_dir} holds a label file for frame {beyond[-1]}, but"
            f" {video.name} has {n_frames} frames (0 to {n_frames - 1}). These"
            " labels were made from another video or at another frame rate, or"
            " --first-index is wrong."
        )

    contours_per_frame: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
    scores_per_frame: list[list[float]] = [[] for _ in range(n_frames)]
    total = 0
    for done, (number, path) in enumerate(sorted(files.items()), start=1):
        contours, scores, found_classes = read_label_file(path, width, height)
        keep = [
            i
            for i in range(len(contours))
            if (classes is None or found_classes[i] in classes)
            and (score_threshold is None or scores[i] >= score_threshold)
        ]
        keep.sort(key=lambda i: scores[i], reverse=True)
        if max_instances:
            keep = keep[:max_instances]
        frame = number - first_index
        contours_per_frame[frame] = [contours[i] for i in keep]
        scores_per_frame[frame] = [scores[i] for i in keep]
        total += len(keep)
        if progress is not None:
            progress(done, len(files))

    # Only what was actually applied is recorded, so a [segmentation_model]
    # table is checked against facts and not against guesses.
    provenance: dict = {}
    if weights_name:
        provenance["weights"] = weights_name
    if score_threshold is not None:
        provenance["score_threshold"] = float(score_threshold)
    if max_instances:
        provenance["max_instances"] = int(max_instances)
    if classes:
        import json  # noqa: PLC0415

        provenance["classes"] = json.dumps(sorted(set(classes)))

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    write_contours(
        partial,
        contours_per_frame,
        width=width,
        height=height,
        scores_per_frame=scores_per_frame,
        source=Path(__file__).name,
        video=video.name,
        backend="yolo",
        model="yolo (imported label files)",
        labels_dir=str(labels_dir),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance,
    )
    partial.replace(output)
    return {
        "video": video.name,
        "output": str(output),
        "frames": n_frames,
        "label_files": len(files),
        "empty_frames": n_frames - len(files),
        "contours": total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import YOLO segmentation label files as idtracker.ai contours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="folder of <video stem>_<frame>.txt files written with save_txt=True",
    )
    parser.add_argument("--output", type=Path, required=True, help="contour file (.h5)")
    parser.add_argument(
        "--first-index",
        type=int,
        default=1,
        help="the number in the first frame's file name; ultralytics starts at 1",
    )
    parser.add_argument(
        "--score-threshold", type=float, help="drop outlines scored below this"
    )
    parser.add_argument(
        "--max-instances", type=int, help="keep at most this many per frame, best first"
    )
    parser.add_argument("--classes", type=int, nargs="+", help="keep only these classes")
    parser.add_argument(
        "--weights-name",
        help="file name of the weights the labels were made with, recorded in"
        " the contour file so a [segmentation_model] table can be checked",
    )
    args = parser.parse_args()

    try:
        stats = import_video(
            args.video,
            args.labels,
            args.output,
            first_index=args.first_index,
            score_threshold=args.score_threshold,
            max_instances=args.max_instances,
            classes=args.classes,
            weights_name=args.weights_name,
        )
    except YoloImportError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    files = stats["label_files"]
    print(
        f"{stats['output']}: {stats['contours']} contours from {files} label"
        f" file{'' if files == 1 else 's'}; {stats['empty_frames']} of"
        f" {stats['frames']} frames have none"
    )
    print(f'Track with:  external_contours = "{stats["output"]}"')


if __name__ == "__main__":
    main()
