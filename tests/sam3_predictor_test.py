"""What Sam3Predictor hands to SAM 3, checked against a stand-in.

Both of the defects this file exists to prevent were the same mistake: SAM 3's
processor keeps its own device and its own confidence threshold, and both of
its defaults are wrong for us. Leaving either out is invisible in review,
survives every test that does not actually construct the thing, and fails only
on a machine or a setting nobody tried.

So a fake sam3 stands in for the real package and records what it was given.
No torch, no GPU, no 3.4 GB checkpoint.
"""

import sys
import types

import numpy as np
import pytest


class FakeProcessor:
    """Records its keyword arguments, and answers like the real one."""

    last = None

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.output = {"masks": np.zeros((0, 1, 4, 4), np.uint8), "scores": np.zeros(0)}
        FakeProcessor.last = self

    def set_image(self, image):
        return {"image": image}

    def set_text_prompt(self, state, prompt):
        self.prompt = prompt
        return self.output


@pytest.fixture
def fake_sam3(monkeypatch, tmp_path):
    """Installs a stand-in sam3 (and torch) for the duration of one test."""
    built = {}

    def build_sam3_image_model(checkpoint_path=None, load_from_HF=True, device="cuda"):
        built.update(
            checkpoint_path=checkpoint_path, load_from_HF=load_from_HF, device=device
        )
        return object()

    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_image_model = build_sam3_image_model

    processor_mod = types.ModuleType("sam3.model.sam3_image_processor")
    processor_mod.Sam3Processor = FakeProcessor

    sam3 = types.ModuleType("sam3")
    sam3.model_builder = model_builder
    sam3_model = types.ModuleType("sam3.model")
    sam3_model.sam3_image_processor = processor_mod
    sam3.model = sam3_model

    cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: "")
    torch = types.ModuleType("torch")
    torch.cuda = cuda

    for name, module in {
        "sam3": sam3,
        "sam3.model": sam3_model,
        "sam3.model.sam3_image_processor": processor_mod,
        "sam3.model_builder": model_builder,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # only stand in for torch when it is genuinely absent, so this does not
    # quietly replace a real installation on a machine that has one
    if "torch" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch", torch)

    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"not really a checkpoint")
    FakeProcessor.last = None
    return checkpoint, built


def make(checkpoint, **kwargs):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import Sam3Predictor

    options = dict(prompt="fish", device="cuda", score_threshold=0.5, max_instances=0)
    options.update(kwargs)
    return Sam3Predictor(checkpoint=checkpoint, **options)


def test_the_processor_is_told_which_device_to_use(fake_sam3):
    """Its default is "cuda", and it allocates in its own constructor.

    Leaving this out undoes the CPU fallback entirely: the model loads on the
    CPU and the processor then dies reaching for a GPU that is not there.
    """
    checkpoint, _ = fake_sam3
    predictor = make(checkpoint)

    assert predictor.device == "cpu"  # no CUDA in this fake
    assert FakeProcessor.last.device == "cpu"


def test_the_processor_is_told_our_confidence_threshold(fake_sam3):
    """Its default of 0.5 would otherwise be a floor under --score-threshold.

    The processor filters inside the forward pass, before we see anything, so
    asking for 0.3 and being silently given 0.5 is indistinguishable from SAM 3
    simply not finding the animals.
    """
    checkpoint, _ = fake_sam3
    make(checkpoint, score_threshold=0.3)

    assert FakeProcessor.last.confidence_threshold == 0.3


def test_a_lower_threshold_actually_reaches_the_model(fake_sam3):
    checkpoint, _ = fake_sam3
    for asked in (0.1, 0.25, 0.9):
        make(checkpoint, score_threshold=asked)
        assert FakeProcessor.last.confidence_threshold == asked


def test_the_checkpoint_we_were_given_is_the_one_loaded(fake_sam3):
    """load_from_HF would otherwise fetch a second copy into a cache."""
    checkpoint, built = fake_sam3
    make(checkpoint)

    assert built["checkpoint_path"] == str(checkpoint)
    assert built["load_from_HF"] is False
    assert built["device"] == "cpu"


def test_a_missing_checkpoint_is_reported_before_anything_loads(fake_sam3, tmp_path):
    from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import Sam3Predictor

    with pytest.raises(FileNotFoundError, match="sam3.pt"):
        Sam3Predictor(checkpoint=tmp_path / "absent.pt", prompt="fish")


def test_masks_come_back_as_plain_binary_images(fake_sam3):
    """SAM 3 returns (N, 1, H, W) probabilities; the pipeline wants (H, W) 0/1."""
    checkpoint, _ = fake_sam3
    predictor = make(checkpoint, score_threshold=0.4)

    masks = np.zeros((2, 1, 8, 8), np.float32)
    masks[0, 0, 1:4, 1:4] = 0.9
    masks[1, 0, 5:7, 5:7] = 0.8
    FakeProcessor.last.output = {"masks": masks, "scores": np.array([0.95, 0.85])}

    out_masks, out_scores = predictor.predict(np.zeros((8, 8, 3), np.uint8))

    assert len(out_masks) == 2
    for mask in out_masks:
        assert mask.shape == (8, 8)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}
    assert out_scores == [pytest.approx(0.95), pytest.approx(0.85)]


def test_detections_below_the_threshold_are_dropped(fake_sam3):
    checkpoint, _ = fake_sam3
    predictor = make(checkpoint, score_threshold=0.6)

    masks = np.zeros((2, 1, 8, 8), np.float32)
    masks[:, 0, 1:4, 1:4] = 0.9
    FakeProcessor.last.output = {"masks": masks, "scores": np.array([0.95, 0.20])}

    out_masks, out_scores = predictor.predict(np.zeros((8, 8, 3), np.uint8))

    assert len(out_masks) == 1
    assert out_scores == [pytest.approx(0.95)]


def test_only_the_most_confident_animals_are_kept(fake_sam3):
    checkpoint, _ = fake_sam3
    predictor = make(checkpoint, max_instances=2)

    masks = np.zeros((4, 1, 8, 8), np.float32)
    masks[:, 0, 1:4, 1:4] = 0.9
    FakeProcessor.last.output = {
        "masks": masks,
        "scores": np.array([0.60, 0.99, 0.70, 0.80]),
    }

    _, out_scores = predictor.predict(np.zeros((8, 8, 3), np.uint8))

    assert out_scores == [pytest.approx(0.99), pytest.approx(0.80)]


def test_nothing_found_is_not_an_error(fake_sam3):
    checkpoint, _ = fake_sam3
    predictor = make(checkpoint)

    out_masks, out_scores = predictor.predict(np.zeros((8, 8, 3), np.uint8))

    assert out_masks == []
    assert out_scores == []


def test_huggingface_loads_the_downloaded_checkpoint(fake_sam3, monkeypatch, tmp_path):
    """weights = "huggingface" reaches SAM 3 as the cached file, not the word.

    This is what makes drafting annotations work with the fetched checkpoint:
    the prelabel command and the app's drafting thread both build their model
    through Sam3Predictor.
    """
    import types as _types

    cached = tmp_path / "cache" / "sam3.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"fetched")
    hub = _types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, **kw: str(cached)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    _, built = fake_sam3
    make("huggingface")

    assert built["checkpoint_path"] == str(cached)
