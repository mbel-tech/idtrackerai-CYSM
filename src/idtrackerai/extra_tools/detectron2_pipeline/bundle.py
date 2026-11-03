"""Builds the zip the Colab notebook unpacks.

Ships from the installed package rather than from a repository checkout, so
``idtrackerai_d2_bundle`` works for anyone who pip-installed this fork. That
matters because the GPU stages are the part most likely to be run by someone
who never cloned anything.

What travels is everything needed from the annotated frames onward: building
the dataset, training, exporting contours. Choosing the enhancement, sampling
frames and annotating them stay in the Segmentation App, because each wants
eyes on the footage -- and LabelMe could not run on a Colab runtime in any
case, having no display to draw on.

Several files are written at archive paths that differ from their module names,
because the notebook and the scripts locate each other by those paths once
unpacked:

- ``preprocessing`` lands at ``tools/preprocessing.py``, which is where
  ``inference`` looks for it when idtracker.ai is not installed, as it is not
  on a Colab runtime
- ``external_contours`` keeps its repository path under ``src/``, which is
  where ``inference.load_writer`` looks for it
"""

import argparse
import zipfile
from importlib import resources
from pathlib import Path

# (module inside this package, path inside the zip)
PIPELINE_FILES = [
    ("preprocessing.py", "tools/preprocessing.py"),
    ("errors.py", "tools/errors.py"),
    # the [segmentation_model] table of a session .toml, read by the exporter's
    # --session mode; beside it, because inference imports it by bare name
    ("segmentation_model.py", "tools/segmentation_model.py"),
    # imported by name from a flat folder on the runtime, which is why it
    # falls back to a bare "from errors import ..." with no package
    ("dataset.py", "tools/dataset.py"),
    ("training.py", "tools/train_detectron2.py"),
    ("inference.py", "tools/detectron2_export_contours.py"),
    ("videos.py", "tools/check_videos.py"),
    # SAM 3 needs a GPU as much as Detectron2 does, so both of its stages
    # travel too: the backend the exporter loads for --backend sam3, and the
    # pre-labelling step, which is only a local step on a machine that has a
    # CUDA card to run it on.
    ("sam3_predictor.py", "tools/sam3_predictor.py"),
    ("prelabel.py", "tools/sam3_prelabel.py"),
    ("notebooks/colab_detectron2_pipeline.ipynb", "colab_detectron2_pipeline.ipynb"),
]

# files that live elsewhere in idtrackerai
EXTERNAL_FILES = [
    (
        "idtrackerai.base.animals_detection",
        "external_contours.py",
        "src/idtrackerai/base/animals_detection/external_contours.py",
    )
]


def build(output: Path) -> Path:
    """Writes the bundle, reading everything through importlib.resources."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    here = resources.files("idtrackerai.extra_tools.detectron2_pipeline")

    written: list[tuple[str, int]] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arcname in PIPELINE_FILES:
            source = here
            for part in name.split("/"):
                source = source / part
            data = source.read_bytes()
            zf.writestr(arcname, data)
            written.append((arcname, len(data)))

        for package, name, arcname in EXTERNAL_FILES:
            data = (resources.files(package) / name).read_bytes()
            zf.writestr(arcname, data)
            written.append((arcname, len(data)))

    build.last_contents = written  # type: ignore[attr-defined]
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build the Colab bundle for the GPU stages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("colab_bundle.zip")
    )
    args = parser.parse_args()

    path = build(args.output)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    for arcname, size in getattr(build, "last_contents", []):
        print(f"  {arcname}  ({size / 1024:.0f} KB)")
    print(
        "\nUpload this to your Drive project folder alongside the dataset and\n"
        "your videos, then open colab_detectron2_pipeline.ipynb in Colab."
    )


if __name__ == "__main__":
    main()
