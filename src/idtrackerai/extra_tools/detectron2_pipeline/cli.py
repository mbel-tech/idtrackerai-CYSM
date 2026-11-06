"""Command-line front ends for the local pipeline stages.

These do the same work as the Segmentation App's guided panel, for scripting or
for a headless machine. They are installed as commands with the package, so
they are available to anyone who pip-installed this fork without cloning it:

    idtrackerai_d2_enhance   tune the enhancement and save a setup profile
    idtrackerai_d2_sample    extract frames for annotation
    idtrackerai_d2_dataset   validate LabelMe polygons and build COCO
    idtrackerai_d2_train     fine-tune Mask R-CNN            (needs a GPU)
    idtrackerai_d2_export    run the model over videos       (needs a GPU)
    idtrackerai_d2_videos    preflight a set of videos
    idtrackerai_d2_bundle    build the Colab bundle

Errors are raised as PipelineError by the library and converted to SystemExit
here, so the library stays usable from the GUI, where SystemExit would abort
the process.
"""

import argparse
from pathlib import Path

from . import dataset as dataset_mod
from . import preprocessing as fp
from . import sampling as sampling_mod
from .errors import PipelineError, SamplingError


def sample_main():
    parser = argparse.ArgumentParser(
        prog="idtrackerai_d2_sample",
        description="Sample frames from videos for LabelMe annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "video files, globs, or folders; a folder contributes the videos "
            "directly inside it (see --recursive)"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="for each folder given to --videos, also use every video in its subfolders",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--n-frames",
        type=int,
        required=True,
        help="total frames across all videos, not per video",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--even",
        action="store_true",
        help="take the same number from each video instead of proportionally",
    )
    parser.add_argument(
        "--uniform-random",
        action="store_true",
        help="sample uniformly at random instead of one frame per time block",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="png is lossless; jpg artefacts can shift annotated boundaries",
    )
    parser.add_argument(
        "--force-new-enhancement",
        action="store_true",
        help=(
            "allow sampling into a folder whose recorded enhancement differs "
            "from the settings given; the folder's profile is then overwritten"
        ),
    )
    fp.add_arguments(parser)
    args = parser.parse_args()

    try:
        videos = sampling_mod.resolve_videos(args.videos, recursive=args.recursive)
        if not videos:
            raise SamplingError(f"No videos matched {args.videos}")

        # Sampling again into a folder reuses its recorded enhancement unless
        # told otherwise, so a growing corpus is not silently mixed.
        recorded = sampling_mod.recorded_enhancement(args.output)
        settings, notes = fp.resolve_settings(
            args, fallback=recorded, fallback_label=f"the profile in {args.output}"
        )
        for note in notes:
            print(note)
        print(f"enhancement: {fp.describe(settings)}")
        if args.format == "jpg":
            print(
                "WARNING: --format jpg writes lossy frames. Training would see"
                " JPEG artefacts that inference on video frames never has, so"
                " the model trains on a different image distribution than it is"
                " run on. Prefer png unless disk space forces otherwise."
            )

        plans = sampling_mod.plan_sampling(
            videos, args.n_frames, args.seed, args.even, args.uniform_random
        )
        print(f"\n{len(videos)} video(s), {sum(p.available for p in plans)} frames total")
        for plan in plans:
            print(f"  {plan.video.name}: taking {plan.requested} of {plan.available}")

        last = [0]

        def report(written: int) -> None:
            if written - last[0] >= 25 or written == args.n_frames:
                last[0] = written
                print(f"    {written}/{args.n_frames} written", flush=True)

        result = sampling_mod.sample_frames(
            videos,
            args.output,
            args.n_frames,
            settings,
            seed=args.seed,
            even=args.even,
            uniform_random=args.uniform_random,
            image_format=args.format,
            progress=report,
            force_new_enhancement=args.force_new_enhancement,
        )
    except PipelineError as exc:
        raise SystemExit(str(exc))

    if result.unreadable:
        print(f"\n{len(result.unreadable)} frame(s) could not be read:")
        for item in result.unreadable[:10]:
            print(f"  {item}")

    print(f"\n{result.written} frames written to {result.output}")
    if result.manifest_path:
        print(f"Manifest: {result.manifest_path}")
    if result.profile_path:
        print(f"Profile:  {result.profile_path}")
    print(
        "\nNext: annotate them in LabelMe, then run idtrackerai_d2_dataset.\n"
        f"  python -m labelme {result.output} --labels fish --validate-label exact"
    )


def dataset_main():
    parser = argparse.ArgumentParser(
        prog="idtrackerai_d2_dataset",
        description="LabelMe -> COCO for Detectron2 instance segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="folder of images + LabelMe .json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-by",
        choices=("video", "random"),
        default="video",
        help="'video' keeps frames from one clip on one side of the split",
    )
    parser.add_argument(
        "--group-by",
        choices=("recording", "file"),
        default="recording",
        help=(
            "with --split-by video: 'recording' folds the pieces of one "
            "recording together (clip_segment_1/2/3 -> clip), 'file' keeps "
            "every clip separate"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="labels to keep, in category order. Default: every label found",
    )
    parser.add_argument(
        "--single-class",
        metavar="NAME",
        help="merge every label into one class with this name",
    )
    parser.add_argument(
        "--expected-instances",
        type=int,
        help="flag frames whose instance count differs from this (e.g. 5 fish)",
    )
    parser.add_argument(
        "--min-area", type=float, default=10.0, help="reject polygons smaller than this"
    )
    parser.add_argument(
        "--no-copy-images", action="store_true", help="reference images in place"
    )
    args = parser.parse_args()

    request = dataset_mod.DatasetRequest(
        input_dir=args.input,
        output_dir=args.output,
        val_fraction=args.val_fraction,
        split_by=args.split_by,
        group_by=args.group_by,
        seed=args.seed,
        labels=args.labels,
        single_class=args.single_class,
        expected_instances=args.expected_instances,
        min_area=args.min_area,
        copy_images=not args.no_copy_images,
    )

    try:
        report = dataset_mod.build_coco_dataset(request)
    except PipelineError as exc:
        raise SystemExit(str(exc))

    if report.skipped_json:
        print(f"Ignoring non-annotation JSON: {', '.join(report.skipped_json)}")

    names = [c["name"] for c in report.categories]
    print(f"\nCategories: {names}  (NUM_CLASSES = {len(names)})")
    print(f"Train: {report.train_images:4d} images, {report.train_annotations:5d} instances")
    print(f"Val:   {report.val_images:4d} images, {report.val_annotations:5d} instances")
    if args.split_by == "video":
        print(f"  train videos: {report.train_videos}")
        print(f"  val videos:   {report.val_videos}")

    if report.enhancement:
        print(f"\nEnhancement carried forward: {fp.describe(report.enhancement)}")
    else:
        print(
            "\nWARNING: no enhancement record found beside the annotations."
            "\nTraining will record null, so inference falls back to the"
            " built-in defaults\nand the mismatch check cannot fire."
        )

    for label, items in (("problem", report.problems), ("note", report.notes)):
        if items:
            print(f"\n{len(items)} {label}(s):")
            for line in items[:15]:
                print(f"  {line}")
            if len(items) > 15:
                print(f"  ... and {len(items) - 15} more, see report.json")

    print(f"\nWritten to {report.output_dir}")
    print(f"Next: idtrackerai_d2_train --dataset {report.output_dir} --output model/")
