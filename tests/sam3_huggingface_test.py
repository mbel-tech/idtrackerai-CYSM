"""weights = "huggingface": SAM 3's checkpoint fetched instead of chosen.

sam3.pt is gated on Hugging Face, so the download needs a token from someone
who was granted access. These tests stand in for ``huggingface_hub`` and never
touch the network: what they pin is that the word "huggingface" becomes a real
local file before anything hashes or loads it, and that a refused download
says how to fix it rather than dumping an HTTP traceback.
"""

import sys
import types
from pathlib import Path

import pytest

from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A stand-in huggingface_hub whose download returns a local file."""
    calls = []
    cached = tmp_path / "hf-cache" / "sam3.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"not really a checkpoint")

    def hf_hub_download(repo_id, filename, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return str(cached)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    from idtrackerai.extra_tools.detectron2_pipeline import sam3_predictor

    # the stand-in file is tiny; size checks have their own tests below
    monkeypatch.setattr(sam3_predictor, "MIN_CHECKPOINT_BYTES", 0)
    return hub, calls, cached


# ------------------------------------------------------------------ the table


def test_the_table_accepts_huggingface_for_sam3():
    spec = sm.normalize_model(
        {"backend": "sam3", "weights": "HuggingFace", "prompt": "bird"}
    )
    assert spec["weights"] == sm.HUGGINGFACE == "huggingface"


def test_huggingface_is_only_for_sam3():
    with pytest.raises(sm.SegmentationModelError, match='backend = "sam3"'):
        sm.normalize_model({"weights": "huggingface"})


def test_files_fetched_from_hugging_face_are_recorded_as_sam3_pt():
    """The exporter records the file name, and the download is sam3.pt.

    Without this the tracking check would compare "sam3.pt" in the file against
    "huggingface" in the table and refuse contours that are exactly right.
    """
    spec = sm.normalize_model(
        {"backend": "sam3", "weights": "huggingface", "prompt": "rat"}
    )
    assert sm.weights_label(spec) == "sam3.pt"
    attrs = {
        "backend": "sam3",
        "weights": "sam3.pt",
        "prompt": "rat",
        "score_threshold": spec["score_threshold"],
        "max_instances": 3,
        "on_overlap": "merge",
    }
    assert sm.mismatches(attrs, spec, 3) == []


# --------------------------------------------------------------- the resolver


def test_a_path_is_passed_through_untouched(tmp_path):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
        resolve_checkpoint,
    )

    given = tmp_path / "my_sam3.pt"
    assert resolve_checkpoint(given) == given
    assert resolve_checkpoint(str(given)) == given


def test_huggingface_downloads_the_official_checkpoint(fake_hub):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
        resolve_checkpoint,
    )

    _, calls, cached = fake_hub
    resolved = resolve_checkpoint("huggingface")

    assert resolved == cached
    assert isinstance(resolved, Path)
    assert [(c["repo_id"], c["filename"]) for c in calls] == [
        ("facebook/sam3", "sam3.pt")
    ]


def test_the_word_is_recognised_however_it_is_written(fake_hub):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
        resolve_checkpoint,
    )

    _, calls, cached = fake_hub
    assert resolve_checkpoint(Path("HuggingFace")) == cached
    assert len(calls) == 1


def test_without_huggingface_hub_it_says_what_to_install(monkeypatch):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
        Sam3WeightsError,
        resolve_checkpoint,
    )

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(Sam3WeightsError, match="pip install huggingface_hub"):
        resolve_checkpoint("huggingface")


def test_a_refused_download_says_how_to_get_access(fake_hub):
    """Gated: no token, or a token whose owner was never granted access."""
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
        Sam3WeightsError,
        resolve_checkpoint,
    )

    hub, _, _ = fake_hub

    def refused(repo_id, filename, **kwargs):
        raise RuntimeError(
            "401 Client Error: access to model facebook/sam3 is restricted"
        )

    hub.hf_hub_download = refused
    with pytest.raises(Sam3WeightsError) as caught:
        resolve_checkpoint("huggingface")

    message = str(caught.value)
    assert "huggingface.co/facebook/sam3" in message
    assert "HF_TOKEN" in message
    assert "hf auth login" in message
    # and what actually went wrong, for the case none of that is it
    assert "401 Client Error" in message


# ------------------------------------------------------- the handoff template

TEMPLATE = Path(__file__).resolve().parents[1] / "docs/sam3-session-template.toml"


def _template_table() -> dict:
    import toml

    return toml.load(TEMPLATE)["segmentation_model"]


def test_the_template_refuses_to_run_until_the_animal_is_named():
    """Its prompt is left empty so nobody exports with someone else's word."""
    with pytest.raises(sm.SegmentationModelError, match="needs a prompt"):
        sm.normalize_model(_template_table())


def test_the_template_is_valid_once_the_prompt_is_filled_in():
    spec = sm.normalize_model(_template_table() | {"prompt": "mouse"})

    assert spec["backend"] == "sam3"
    assert spec["weights"] == sm.HUGGINGFACE
    assert spec["prompt"] == "mouse"


def test_a_stub_left_by_a_drive_cache_is_replaced_by_a_local_copy(
    fake_hub, monkeypatch, tmp_path
):
    from idtrackerai.extra_tools.detectron2_pipeline import sam3_predictor as sp

    hub, calls, stub = fake_hub
    stub.write_bytes(b"x" * 79)
    monkeypatch.setattr(sp, "MIN_CHECKPOINT_BYTES", 1000)
    monkeypatch.setattr(sp.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    real = tmp_path / "home" / "real" / "sam3.pt"

    def hf_hub_download(repo_id, filename, **kwargs):
        calls.append(kwargs)
        if "local_dir" not in kwargs:
            return str(stub)
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(b"y" * 2000)
        return str(real)

    hub.hf_hub_download = hf_hub_download
    assert sp.resolve_checkpoint("huggingface") == real
    expected = tmp_path / "home" / ".cache" / "idtrackerai" / "sam3"
    assert calls[-1]["local_dir"] == str(expected)


def test_a_stub_that_cannot_be_replaced_says_to_move_the_cache(fake_hub, monkeypatch):
    from idtrackerai.extra_tools.detectron2_pipeline import sam3_predictor as sp

    _, _, stub = fake_hub
    stub.write_bytes(b"x" * 79)
    monkeypatch.setattr(sp, "MIN_CHECKPOINT_BYTES", 1000)
    with pytest.raises(sp.Sam3WeightsError, match="HF_HOME"):
        sp.resolve_checkpoint("huggingface")
