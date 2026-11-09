"""Enhancement settings must be validated early and kept consistent with the
background model they were used to build."""

import cv2
import numpy as np
import pytest

from idtrackerai import IdtrackeraiError, Session
from idtrackerai.base.animals_detection.segmentation import (
    apply_enhancement,
    load_custom_background,
)
from idtrackerai.extra_tools.detectron2_pipeline import inference
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp


def test_partial_enhancement_is_completed():
    merged = Session.normalized_enhancement({"clahe_clip": 2.0})
    assert merged == {**fp.DEFAULT_SETTINGS, "clahe_clip": 2.0}
    assert Session.normalized_enhancement(None) is None


def test_unknown_enhancement_key_is_rejected_clearly():
    with pytest.raises(IdtrackeraiError, match="clahe_clp"):
        Session.normalized_enhancement({"clahe_clp": 2.0})


def test_apply_enhancement_accepts_partial_dict():
    frame = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    out = apply_enhancement(frame, {"clahe_clip": 2.0})
    assert out.shape == frame.shape and out.dtype == np.uint8
    with pytest.raises(IdtrackeraiError):
        apply_enhancement(frame, {"bogus": 1})


def test_custom_background_is_enhanced_only_when_enabled(tmp_path):
    image = np.tile(np.linspace(40, 90, 64, dtype=np.uint8), (64, 1))
    path = tmp_path / "bkg.png"
    cv2.imencode(".png", image)[1].tofile(path)

    raw = load_custom_background(str(path))
    assert np.array_equal(raw, image)
    assert np.array_equal(load_custom_background(str(path), None, None), image)
    off = load_custom_background(str(path), None, {"enhance": False})
    assert np.array_equal(off, image)

    on = load_custom_background(str(path), None, {"clahe_clip": 4.0})
    assert not np.array_equal(on, image)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Session, "background_path", property(lambda self: tmp_path / "background.png")
    )
    return object.__new__(Session)


def test_background_records_and_checks_its_enhancement(session):
    bkg = np.full((8, 8), 100, np.uint8)
    session.enhancement = {"clahe_clip": 2.0}
    assert session.background_matches_enhancement() is None  # nothing saved yet

    session.bkg_model = bkg
    assert session.background_matches_enhancement() is True

    session.enhancement = {"clahe_clip": 3.0}
    assert session.background_matches_enhancement() is False

    session.enhancement = {"enhance": False}
    assert session.background_matches_enhancement() is False  # built enhanced

    del session.bkg_model
    assert not session.background_settings_path.exists()


def test_background_without_record_is_unknown(session):
    session.enhancement = None
    session.bkg_model = np.zeros((8, 8), np.uint8)
    session.background_settings_path.unlink()
    assert session.background_matches_enhancement() is None


def test_split_keeps_one_piece_per_animal():
    # the high-score mask cuts the low-score one into two pieces
    low = np.zeros((20, 40), np.uint8)
    low[5:15, 2:38] = 1
    high = np.zeros((20, 40), np.uint8)
    high[:, 18:22] = 1
    masks, scores = inference.resolve_overlaps(
        [low, high], [0.6, 0.9], "split", merge_th=0.15
    )
    assert len(masks) == 2
    for mask in masks:
        n_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        assert n_labels == 2  # background + one component


# ------------------------------------------- settings back into command flags
# The SAM 3 panel hands a command to a terminal. SAM 3 has no training record
# for the exporter to adopt the enhancement from, so the command has to carry
# the session's settings itself, and parse back to exactly them.


def _parsed(flags):
    import argparse

    from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

    parser = argparse.ArgumentParser()
    fp.add_arguments(parser)
    settings, _ = fp.resolve_settings(parser.parse_args(flags))
    return settings


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"enhance": False},
        {"enhance": True, "clahe_clip": 3.5, "gamma": 1.4, "denoise": 2},
        {"correct_lighting": False, "shadows": -20, "illumination_sigma": 40.0},
    ],
)
def test_settings_survive_the_trip_through_command_flags(settings):
    from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

    flags = fp.settings_to_flags(settings)

    assert _parsed(flags) == fp.normalize_settings(settings)


def test_default_settings_need_no_flags():
    from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

    assert fp.settings_to_flags({}) == []
    assert fp.settings_to_flags({"enhance": False}) == ["--no-enhance"]
