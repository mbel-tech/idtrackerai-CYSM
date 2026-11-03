"""Checks the Colab bundle unpacks into something whose imports resolve.

The bundle writes several files at archive paths that differ from their module
names, because the notebook and the scripts find each other by those paths.
That renaming is load-bearing and invisible: everything imports perfectly well
in the repository and then fails on a Colab runtime, hours into someone's day,
where idtracker.ai is not installed and the exporter is called by a name its
own module does not have.

So this builds a real bundle, unpacks it, and loads the scripts the way Colab
does -- by path, with idtracker.ai unavailable -- in a subprocess, so blocking
the import cannot leak into the rest of the suite.
"""

import subprocess
import sys
import textwrap
import zipfile

import pytest

from idtrackerai.extra_tools.detectron2_pipeline.bundle import build

# What the notebook expects to find once it has unzipped the bundle.
EXPECTED = {
    "tools/preprocessing.py",
    "tools/errors.py",
    "tools/segmentation_model.py",
    "tools/train_detectron2.py",
    "tools/detectron2_export_contours.py",
    "tools/check_videos.py",
    "tools/sam3_predictor.py",
    "tools/sam3_prelabel.py",
    "colab_detectron2_pipeline.ipynb",
    "src/idtrackerai/base/animals_detection/external_contours.py",
}

LOAD_IN_A_COLAB_LIKE_RUNTIME = textwrap.dedent(
    """
    import importlib.abc, importlib.util, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "idtrackerai" or name.startswith("idtrackerai."):
                raise ImportError("blocked for this test")
            return None

    sys.meta_path.insert(0, Block())
    try:
        import idtrackerai
    except ImportError:
        pass
    else:
        raise SystemExit("idtrackerai was importable; the simulation is wrong")

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    prelabel = load("tools/sam3_prelabel.py", "sam3_prelabel")
    exporter = load("tools/detectron2_export_contours.py", "detectron2_export_contours")
    predictor = load("tools/sam3_predictor.py", "sam3_predictor")

    assert prelabel.mask_to_contours is not None
    assert exporter.load_writer().__name__ == "write_contours"
    assert predictor.Sam3Predictor is not None
    print("OK")
    """
)


@pytest.fixture
def unpacked(tmp_path):
    archive = build(tmp_path / "colab_bundle.zip")
    root = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        handle.extractall(root)
    return root, names


def test_the_bundle_carries_what_the_notebook_looks_for(unpacked):
    _, names = unpacked
    assert EXPECTED <= names, EXPECTED - names


def test_the_gpu_stages_travel_but_the_local_ones_do_not(unpacked):
    """Sampling happens before anything is uploaded.

    dataset.py does travel: training imports it on the runtime.
    """
    _, names = unpacked
    assert not any("sampling" in n for n in names)


def test_the_scripts_find_each_other_once_unpacked(unpacked):
    """The real check: loaded by path, with idtracker.ai not installed.

    sam3_prelabel imports mask_to_contours from the exporter, which the bundle
    renames, so this fails if only the in-repository module name is tried.
    """
    root, _ = unpacked
    result = subprocess.run(
        [sys.executable, "-c", LOAD_IN_A_COLAB_LIKE_RUNTIME],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
