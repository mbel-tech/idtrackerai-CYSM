"""The SAM 3 panel's wiring, checked without a display or a GPU.

These are not tests of how the panel looks. They pin the two things that are
easy to break silently and expensive to discover late: that the controls stay
disabled until a run would actually work, and that every control the panel asks
for a tooltip by name actually has one.
"""

import pytest
import toml

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

from pathlib import Path  # noqa: E402

TOOLTIPS = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/segmentation_app/tooltips.toml"
)


@pytest.fixture(scope="module")
def qt_app():
    """One offscreen QApplication for the module."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def report(cuda=False, sam3=None, detectron2=None, torch="2.7.0"):
    """A GpuReport in a known state, so tests do not depend on this machine."""
    from idtrackerai.extra_tools.detectron2_pipeline.gpu import GpuReport

    return GpuReport(
        torch_version=torch,
        detectron2_version=detectron2,
        sam3_version=sam3,
        cuda_available=cuda,
        device_name="Test GPU" if cuda else None,
        checked=True,
    )


@pytest.fixture
def panel(qt_app):
    from idtrackerai.segmentation_app.widgets import EnhancementWidget, Sam3Panel

    widget = Sam3Panel(EnhancementWidget())
    # The panel probes the real machine on construction. Pin it instead, so
    # these assertions mean the same thing on a laptop and on a CUDA box.
    widget.gpu_thread.wait(30000)
    widget.gpu = report()
    widget._refresh_enabled()
    yield widget
    widget.close()


def test_the_selector_offers_sam3(qt_app):
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        DETECTRON2,
        EXTERNAL,
        SAM3,
        THRESHOLDING,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    modes = [widget.source.itemText(i) for i in range(widget.source.count())]

    assert modes == [THRESHOLDING, EXTERNAL, DETECTRON2, SAM3]


def test_choosing_sam3_does_not_open_a_file_dialog(qt_app):
    """SAM 3 mode produces contours; it must not demand one up front.

    If this regressed, selecting the mode would block on a modal dialog, which
    in an offscreen test means hanging rather than failing.
    """
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        SAM3,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    emitted = []
    widget.valueChanged.connect(emitted.append)

    widget.source.setCurrentText(SAM3)

    assert widget.mode() == SAM3
    assert emitted == [None]
    assert widget.paths == []


def test_nothing_runs_before_a_checkpoint_is_chosen(panel):
    assert not panel.prelabel_button.isEnabled()
    assert not panel.export_button.isEnabled()


def test_export_stays_disabled_without_a_gpu(panel, tmp_path):
    """A whole video on CPU is not slow, it is impractical. Do not offer it."""
    panel.weights = tmp_path / "sam3.pt"
    panel.set_video_context([tmp_path / "clip.mp4"])

    assert not panel.has_gpu
    assert not panel.export_button.isEnabled()


def test_export_is_offered_once_sam3_can_actually_run(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel.prompt.setText("fish")
    panel.set_video_context([tmp_path / "clip.mp4"])
    panel.gpu = report(cuda=True, sam3="0.1.4")
    panel._refresh_enabled()

    assert panel.has_gpu
    assert panel.export_button.isEnabled()


def test_sam3_does_not_wait_for_detectron2(panel, tmp_path):
    """GpuReport.usable wants Detectron2; SAM 3 has no use for it.

    Sharing one notion of "usable" would refuse SAM 3 on a machine that can
    run it perfectly well.
    """
    panel.gpu = report(cuda=True, sam3="0.1.4", detectron2=None)

    assert not panel.gpu.usable
    assert panel.gpu.sam3_usable
    assert panel.has_gpu


def test_drafting_needs_a_frames_folder(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel.prompt.setText("fish")
    panel._refresh_enabled()
    assert not panel.prelabel_button.isEnabled()

    panel.frames_dir = tmp_path
    panel._refresh_enabled()
    assert panel.prelabel_button.isEnabled()


def test_an_empty_prompt_blocks_everything(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel.frames_dir = tmp_path
    panel.prompt.setText("   ")
    panel._refresh_enabled()

    assert not panel.prelabel_button.isEnabled()


def test_the_colab_route_is_offered_exactly_when_it_is_needed(panel):
    assert panel.colab_hint.isVisibleTo(panel)

    panel.gpu = report(cuda=True, sam3="0.1.4")
    panel._refresh_enabled()
    assert not panel.colab_hint.isVisibleTo(panel)


def test_it_says_what_is_missing_rather_than_just_no(panel):
    panel.gpu = report(cuda=False, sam3=None)
    text = panel._gpu_text()

    assert "CUDA" in text
    assert "sam3" in text
    # and that drafting is still possible, which is the useful half
    assert "CPU" in text


def test_every_control_it_labels_has_a_tooltip(panel):
    """setToolTips skips missing keys silently, so check they are all there."""
    tips = toml.load(TOOLTIPS)
    wanted = {
        "sam3_weights",
        "sam3_prompt",
        "sam3_score_threshold",
        "sam3_max_instances",
        "sam3_frames",
        "sam3_prelabel",
        "sam3_overwrite",
        "sam3_export",
    }

    assert wanted <= set(tips)
    panel.setToolTips(tips)
    assert panel.prompt.toolTip()
    assert panel.export_button.toolTip()


# ------------------------------------------------------------ the export batch
# An export runs for hours, so what happens when it is interrupted matters more
# than what happens when it is not.


@pytest.fixture
def export_thread(monkeypatch, tmp_path):
    """A Sam3ExportThread with the model and the exporter stubbed out.

    run() is called directly rather than through start(): this is about the
    batch loop, not about threading.
    """
    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod
    from idtrackerai.segmentation_app.widgets.pipeline_threads import Sam3ExportThread

    exported = []

    def fake_export_video(video, output, *args, **kwargs):
        exported.append(Path(video).name)
        Path(output).write_bytes(b"contours")
        return {"video": Path(video).name, "frames": 1}

    monkeypatch.setattr(inference_mod, "export_video", fake_export_video)
    monkeypatch.setattr(inference_mod, "load_writer", lambda: (lambda *a, **k: None))
    # the checkpoint is not real, and the files written above are stand-ins:
    # an existing file counts as current unless a test says otherwise
    monkeypatch.setattr(inference_mod, "weights_sha256", lambda weights: "fake")
    monkeypatch.setattr(inference_mod, "stale_reason", lambda *a, **k: None)

    import idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor as predictor_mod

    monkeypatch.setattr(
        predictor_mod, "Sam3Predictor", lambda **kw: type("P", (), {"description": "fake"})()
    )

    thread = Sam3ExportThread()
    return thread, exported, tmp_path, monkeypatch


def make_clips(folder, names):
    clips = []
    for name in names:
        path = folder / name
        path.write_bytes(b"video")
        clips.append(path)
    return clips


def test_clips_that_already_have_contours_are_not_redone(export_thread):
    """Cancel three hours into five clips, come back, and keep the three."""
    thread, exported, tmp_path, _ = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    clips = make_clips(tmp_path, ["a.mp4", "b.mp4", "c.mp4"])
    (out / "a.h5").write_bytes(b"done earlier")

    thread.set_parameters(
        videos=clips, output_dir=out, weights=tmp_path / "sam3.pt",
        prompt="fish", device="cpu", score_threshold=0.5,
        max_instances=0, enhancement={},
    )
    thread.run()

    assert exported == ["b.mp4", "c.mp4"]
    assert [p.name for p in thread.skipped] == ["a.h5"]


def test_a_clip_exported_another_way_is_redone(export_thread):
    """A file made with another prompt or frame range is not adopted."""
    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod

    thread, exported, tmp_path, monkeypatch = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    clips = make_clips(tmp_path, ["a.mp4", "b.mp4"])
    (out / "a.h5").write_bytes(b"made with another prompt")
    (out / "b.h5").write_bytes(b"current")
    monkeypatch.setattr(
        inference_mod,
        "stale_reason",
        lambda output, *a, **k: "prompt differs" if output.name == "a.h5" else None,
    )
    thread.set_parameters(
        videos=clips, output_dir=out, weights=tmp_path / "sam3.pt",
        prompt="chicken", device="cpu", score_threshold=0.5,
        max_instances=0, enhancement={},
    )
    thread.run()

    assert exported == ["a.mp4"]
    assert [p.name for p in thread.skipped] == ["b.h5"]


def test_only_the_tracked_frames_are_exported(export_thread):
    """The session's tracking intervals reach the exporter, per clip."""
    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod

    thread, _, tmp_path, monkeypatch = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    seen = []

    def export_recording_intervals(video, output, args, *a, **k):
        seen.append(args.intervals)
        Path(output).write_bytes(b"c")
        return {"video": Path(video).name, "frames": 1}

    monkeypatch.setattr(inference_mod, "export_video", export_recording_intervals)
    monkeypatch.setattr(inference_mod, "video_frame_count", lambda video: 100)
    thread.set_parameters(
        videos=make_clips(tmp_path, ["a.mp4", "b.mp4"]), output_dir=out,
        weights=tmp_path / "sam3.pt", prompt="chicken", device="cpu",
        score_threshold=0.5, max_instances=0, enhancement={},
        tracking_intervals=[[30, 150]],
    )
    thread.run()

    assert seen == [[[30, 100]], [[0, 50]]]


def test_the_session_still_gets_every_clip_it_needs(export_thread):
    """Skipped is not the same as missing: the whole set must be adopted."""
    thread, _, tmp_path, monkeypatch = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    clips = make_clips(tmp_path, ["a.mp4", "b.mp4", "c.mp4"])
    (out / "a.h5").write_bytes(b"done earlier")

    thread.set_parameters(
        videos=clips, output_dir=out, weights=tmp_path / "sam3.pt",
        prompt="fish", device="cpu", score_threshold=0.5,
        max_instances=0, enhancement={},
    )
    thread.run()

    assert [p.name for p in thread.written] == ["a.h5", "b.h5", "c.h5"]


def test_re_export_is_available_when_it_is_what_you_meant(export_thread):
    thread, exported, tmp_path, _ = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    clips = make_clips(tmp_path, ["a.mp4", "b.mp4"])
    (out / "a.h5").write_bytes(b"done earlier")

    thread.set_parameters(
        videos=clips, output_dir=out, weights=tmp_path / "sam3.pt",
        prompt="fish", device="cpu", score_threshold=0.5,
        max_instances=0, enhancement={}, overwrite=True,
    )
    thread.run()

    assert exported == ["a.mp4", "b.mp4"]
    assert thread.skipped == []


def test_the_progress_bar_is_given_a_maximum(export_thread, qt_app):
    """Without it the dialog stays at 100 while the value counts frames."""
    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod

    thread, _, tmp_path, monkeypatch = export_thread
    out = tmp_path / "contours"
    out.mkdir()
    maxima = []
    thread.set_progress_max.connect(maxima.append)

    def export_reporting_progress(video, output, *args, progress=None, **kwargs):
        Path(output).write_bytes(b"c")
        for frame in range(1, 4):
            progress(frame, 3)
        return {"video": Path(video).name, "frames": 3}

    monkeypatch.setattr(inference_mod, "export_video", export_reporting_progress)
    thread.set_parameters(
        videos=make_clips(tmp_path, ["a.mp4"]), output_dir=out,
        weights=tmp_path / "sam3.pt", prompt="fish", device="cpu",
        score_threshold=0.5, max_instances=0, enhancement={},
    )
    thread.run()

    assert maxima == [3]


def test_a_finished_export_does_not_claim_the_session_adopted_it(panel, tmp_path):
    """Whether the contours fit is the source widget's call, not the panel's.

    A cancelled batch produces files that will be refused; announcing success
    here would contradict the warning the user is about to see.
    """
    from idtrackerai.segmentation_app.widgets import sam3_panel as panel_mod

    shown = []
    panel.export_thread.written = [tmp_path / "a.h5"]
    panel.export_thread.skipped = []
    emitted = []
    panel.contoursReady.connect(emitted.append)

    mp = pytest.MonkeyPatch()
    mp.setattr(
        panel_mod.QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[2])),
    )
    try:
        panel._export_finished()
    finally:
        mp.undo()

    assert emitted == [[tmp_path / "a.h5"]]
    assert shown, "the user should still be told the export finished"
    assert "tracks from them" not in shown[0]
    assert "written to" in shown[0]


def test_export_refuses_clips_that_would_share_a_contour_file(panel, tmp_path):
    """Contour files are named for the clip's file name, so two clips called
    cam1.mp4 would write one file, and the second would be skipped as 'done'."""
    from idtrackerai.segmentation_app.widgets import sam3_panel as panel_mod

    first, second = tmp_path / "a" / "cam1.mp4", tmp_path / "b" / "cam1.mp4"
    panel.gpu = report(cuda=True, sam3="0.1.4")
    panel.weights = tmp_path / "sam3.pt"
    panel.prompt.setText("fish")
    panel.contours_dir = tmp_path / "contours"
    panel.video_paths = [first, second]

    warned, started = [], []
    mp = pytest.MonkeyPatch()
    mp.setattr(
        panel_mod.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warned.append(a[2])),
    )
    mp.setattr(panel.export_thread, "start", lambda *a, **k: started.append(True))
    try:
        panel.run_export()
    finally:
        mp.undo()

    assert len(warned) == 1
    assert (
        "Two videos would write the same contour file (same file name in different"
        " folders). Export them separately." in warned[0]
    )
    assert str(first) in warned[0] and str(second) in warned[0]
    assert started == []
    assert not (tmp_path / "contours").exists()


# ------------------------------------------------------------------- the prompt
# Whatever the animal is, it is typed here or comes from the .toml. Nothing is
# assumed: a pre-filled "fish" would be run on chickens by anyone who did not
# notice it.


def test_the_prompt_starts_empty_and_says_what_goes_in_it(panel):
    assert panel.prompt.text() == ""
    assert panel.prompt.placeholderText() == "the animal, e.g. fish"


def ready_to_export(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel.frames_dir = tmp_path
    panel.gpu = report(cuda=True, sam3="0.1.4")
    panel.set_video_context([tmp_path / "clip.mp4"], tmp_path / "session")


def test_typing_a_prompt_is_what_enables_the_runs(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    assert not panel.export_button.isEnabled()
    assert not panel.prelabel_button.isEnabled()

    panel.prompt.setText("bird")  # nothing else: typing must be enough

    assert panel.export_button.isEnabled()
    assert panel.prelabel_button.isEnabled()


def test_clearing_the_prompt_disables_them_again(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.prompt.setText("rat")
    panel.prompt.setText("  ")

    assert not panel.export_button.isEnabled()
    assert not panel.prelabel_button.isEnabled()


def test_an_empty_prompt_never_reaches_a_run(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    started = []
    panel.export_thread.start = lambda *a, **k: started.append("export")
    panel.prelabel_thread.start = lambda *a, **k: started.append("prelabel")

    panel.run_export()
    panel.run_prelabel()

    assert started == []


def test_a_prompt_from_the_toml_fills_the_box(panel, tmp_path):
    panel.set_model_spec(
        {"backend": "sam3", "weights": str(tmp_path / "sam3.pt"), "prompt": "zebrafish"}
    )
    assert panel.prompt.text() == "zebrafish"


def test_the_export_explanation_reads_as_prose(panel):
    text = panel.export_explanation.text()
    assert "' +" not in text and "+ '" not in text
    assert "skipped" in text


# -------------------------------------------------------- weights from the hub


def test_hugging_face_can_stand_in_for_a_chosen_file(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.weights = None
    panel.prompt.setText("bird")
    assert not panel.export_button.isEnabled()

    panel.hf_button.click()

    assert str(panel.weights) == "huggingface"
    assert "Hugging Face" in panel.weights_label.text()
    assert panel.export_button.isEnabled()
    assert panel.model_spec()["weights"] == "huggingface"


def test_a_toml_asking_for_hugging_face_restores_it(panel):
    panel.set_model_spec(
        {"backend": "sam3", "weights": "huggingface", "prompt": "bird"}
    )
    assert str(panel.weights) == "huggingface"
    assert "Hugging Face" in panel.weights_label.text()


def test_the_export_thread_hashes_the_downloaded_file(export_thread):
    """Not the word: the contour file must record the checkpoint really used."""
    import types

    from idtrackerai.extra_tools.detectron2_pipeline import inference as inference_mod

    thread, exported, tmp_path, monkeypatch = export_thread
    cached = tmp_path / "cache" / "sam3.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"fetched")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, **kw: str(cached)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub)
    hashed = []
    monkeypatch.setattr(
        inference_mod, "weights_sha256", lambda weights: hashed.append(weights) or "x"
    )
    out = tmp_path / "contours"
    out.mkdir()
    thread.set_parameters(
        videos=make_clips(tmp_path, ["a.mp4"]), output_dir=out,
        weights=Path("huggingface"), prompt="bird", device="cpu",
        score_threshold=0.5, max_instances=0, enhancement={},
    )
    thread.run()

    assert hashed == [cached]
    assert exported == ["a.mp4"]


def test_a_refused_download_is_reported_not_raised(export_thread):
    thread, exported, tmp_path, monkeypatch = export_thread
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    failures = []
    thread.failed.connect(failures.append)
    thread.set_parameters(
        videos=make_clips(tmp_path, ["a.mp4"]), output_dir=tmp_path,
        weights=Path("huggingface"), prompt="bird", device="cpu",
        score_threshold=0.5, max_instances=0, enhancement={},
    )
    thread.run()

    assert exported == []
    assert failures and "huggingface_hub" in failures[0]


# --------------------------------------------------------- the contours folder


def test_the_contours_folder_is_shown_before_anything_is_chosen(panel, tmp_path):
    panel.set_video_context([tmp_path / "clip.mp4"], tmp_path / "session")
    assert panel.contours_field.text() == str(tmp_path / "session" / "sam3_contours")


def test_a_chosen_contours_folder_is_used_and_saved(panel, tmp_path):
    from idtrackerai.segmentation_app.widgets import sam3_panel as panel_mod

    chosen = tmp_path / "pen1_contours"
    mp = pytest.MonkeyPatch()
    mp.setattr(
        panel_mod.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(chosen)),
    )
    try:
        panel.contours_button.click()
    finally:
        mp.undo()
    panel.weights = tmp_path / "sam3.pt"
    panel.prompt.setText("bird")

    assert panel.export_dir() == chosen
    assert panel.contours_field.text() == str(chosen)
    assert panel.model_spec()["contours_dir"] == str(chosen)


def test_the_tomls_contours_folder_is_shown(panel, tmp_path):
    panel.set_model_spec(
        {
            "backend": "sam3", "weights": "huggingface", "prompt": "bird",
            "contours_dir": str(tmp_path / "from_toml"),
        }
    )
    assert panel.contours_field.text() == str(tmp_path / "from_toml")


# ------------------------------------------------------------ the command line
# What the panel would run, written out for a machine that has the GPU. It is
# only useful if it does exactly what the panel's own export would.


def command(panel) -> str:
    return panel.export_command.toPlainText()


def test_the_command_carries_the_prompt_as_typed(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.prompt.setText("bird")
    assert '--backend sam3 --prompt "bird"' in command(panel)

    panel.prompt.setText("lab rat")
    assert '--prompt "lab rat"' in command(panel)


def test_the_command_says_what_is_still_missing(panel):
    text = command(panel)
    assert '--prompt "<the animal>"' in text
    assert '--weights "<sam3.pt>"' in text


def test_the_command_exports_what_the_panel_would(panel, tmp_path):
    """The exporter's own defaults differ (0.7, 5 per frame), so both are
    written out whatever their value."""
    ready_to_export(panel, tmp_path)
    panel.prompt.setText("bird")
    text = command(panel)

    assert "inference.py" in text
    assert f'--weights "{tmp_path / "sam3.pt"}"' in text
    assert f'--videos "{tmp_path / "clip.mp4"}"' in text
    assert f'--output-dir "{tmp_path / "session" / "sam3_contours"}"' in text
    assert "--score-threshold 0.5" in text
    assert "--max-instances 0" in text

    panel.score_threshold.setValue(0.35)
    panel.max_instances.setValue(9)
    assert "--score-threshold 0.35" in command(panel)
    assert "--max-instances 9" in command(panel)


def test_the_command_fetches_from_hugging_face_when_chosen(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.hf_button.click()
    assert "--weights huggingface" in command(panel)


def test_the_command_carries_the_sessions_enhancement(panel, tmp_path):
    """SAM 3 has no training record for the exporter to take it from."""
    ready_to_export(panel, tmp_path)
    panel.enhancement_committed({"enhance": False})
    assert command(panel).endswith("--no-enhance")


def test_the_command_is_there_with_or_without_a_gpu(panel, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.gpu = report()
    panel._refresh_enabled()
    assert panel.export_command.isVisibleTo(panel)
    assert panel.copy_command_button.isVisibleTo(panel)


def test_copy_puts_the_command_on_the_clipboard(panel, qt_app, tmp_path):
    ready_to_export(panel, tmp_path)
    panel.prompt.setText("bird")
    panel.copy_command_button.click()
    assert qt_app.clipboard().text() == command(panel)


def test_the_new_controls_have_tooltips(panel):
    tips = toml.load(TOOLTIPS)
    assert {"sam3_huggingface", "sam3_contours_dir", "sam3_command"} <= set(tips)
    panel.setToolTips(tips)
    assert panel.hf_button.toolTip()
    assert panel.contours_button.toolTip()
    assert panel.export_command.toolTip()
