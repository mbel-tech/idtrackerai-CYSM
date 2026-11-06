"""The SAM 3 mode: segmenting from a word, with no model of your own.

Two things live here, because they are the two ways SAM 3 is worth using and
neither is worth a panel of its own:

**Export contours** produces the same sidecar the Detectron2 pipeline does, so
tracking reads it through the External contours path that is already tested.
Use it to get a clip tracked today, or for a recording you will never train a
model for.

**Draft annotations** runs SAM 3 over sampled frames and writes LabelMe files
for you to correct, which turns the slow part of preparing a Detectron2 dataset
from drawing into correcting.

Where the work happens
----------------------
SAM 3 is a 3.4 GB model and wants a CUDA GPU. Most machines running the
Segmentation App do not have one -- PyTorch has no ROCm build for Windows, so
an AMD card is not an alternative -- and this panel says so rather than
offering a run that would take days. Drafting a few hundred frames on CPU is
slow but survivable; exporting a whole video on CPU is not, so only the first
is offered without a GPU.

When there is no GPU here, the Colab route is the answer, and it is the same
bundle the Detectron2 stages already use.
"""

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import gpu as gpu_mod
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm
from idtrackerai.GUI_tools import WrappedLabel

from .pipeline_threads import GpuCheckThread, Sam3ExportThread, Sam3PrelabelThread


class Sam3Panel(QWidget):
    """Everything SAM 3 needs, and nothing the other modes already provide."""

    contoursReady = Signal(object)  # list[Path] written by an export

    def __init__(self, enhancement):
        super().__init__()
        self.enhancement = enhancement
        self.video_paths: list[Path] = []
        self.output_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.weights: Path | None = None
        # Where exports go. Set from a .toml's contours_dir; otherwise
        # sam3_contours beside the session output.
        self.contours_dir: Path | None = None
        self.number_of_animals = 0
        # returns the session's tracking intervals, so only those frames run
        self.tracking_intervals = lambda: None
        # [segmentation_model] keys with no control here, kept from the .toml
        self._loaded_spec: dict = {}

        # Unchecked until the probe comes back, so the panel opens instantly
        # and says it is still looking rather than freezing on a torch import.
        self.gpu = gpu_mod.GpuReport()

        self.prelabel_thread = Sam3PrelabelThread()
        self.export_thread = Sam3ExportThread()
        self.gpu_thread = GpuCheckThread()
        self.gpu_thread.reported.connect(self._gpu_reported)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._model_box())
        layout.addWidget(self._prelabel_box())
        layout.addWidget(self._export_box())
        layout.addStretch()

        self._wire_threads()
        self._refresh_enabled()
        self.check_gpu()

    # ------------------------------------------------------------ the model
    def _model_box(self) -> QWidget:
        box = QGroupBox("SAM 3")
        grid = QGridLayout()
        box.setLayout(grid)

        self.weights_label = WrappedLabel(framed=True)
        self.weights_label.setText("No sam3.pt chosen")
        self.weights_button = QPushButton("Choose sam3.pt")
        self.weights_button.clicked.connect(self.choose_weights)

        self.prompt = QLineEdit("fish")
        self.prompt.setPlaceholderText("the animal, e.g. fish")

        self.score_threshold = QDoubleSpinBox()
        self.score_threshold.setRange(0.0, 1.0)
        self.score_threshold.setSingleStep(0.05)
        self.score_threshold.setValue(0.5)

        self.max_instances = QSpinBox()
        self.max_instances.setRange(0, 1000)
        self.max_instances.setValue(0)
        self.max_instances.setSpecialValueText("all")

        self.gpu_status = WrappedLabel()
        self.gpu_status.setText(self._gpu_text())

        grid.addWidget(QLabel("Weights"), 0, 0)
        grid.addWidget(self.weights_label, 0, 1)
        grid.addWidget(self.weights_button, 0, 2)
        grid.addWidget(QLabel("Prompt"), 1, 0)
        grid.addWidget(self.prompt, 1, 1, 1, 2)
        grid.addWidget(QLabel("Confidence"), 2, 0)
        grid.addWidget(self.score_threshold, 2, 1)
        grid.addWidget(QLabel("Animals per frame"), 3, 0)
        grid.addWidget(self.max_instances, 3, 1)
        grid.addWidget(self.gpu_status, 4, 0, 1, 3)
        return box

    # -------------------------------------------------------- drafting labels
    def _prelabel_box(self) -> QWidget:
        box = QGroupBox("Draft annotations for correction")
        layout = QVBoxLayout()
        box.setLayout(layout)

        explanation = WrappedLabel()
        explanation.setText(
            "Outlines the animals in the sampled frames and writes LabelMe "
            "files beside them. Correct what it drew, and add what it missed, "
            "before building a dataset. Frames you have already annotated are "
            "left alone."
        )
        layout.addWidget(explanation)

        row = QHBoxLayout()
        self.frames_label = WrappedLabel(framed=True)
        self.frames_label.setText("No frames folder chosen")
        self.frames_button = QPushButton("Choose frames")
        self.frames_button.clicked.connect(self.choose_frames)
        row.addWidget(self.frames_label)
        row.addWidget(self.frames_button)
        layout.addLayout(row)

        self.overwrite = QCheckBox("Redraw frames that already have annotations")
        layout.addWidget(self.overwrite)

        self.prelabel_button = QPushButton("Draft annotations")
        self.prelabel_button.clicked.connect(self.run_prelabel)
        layout.addWidget(self.prelabel_button)
        return box

    # ------------------------------------------------------ exporting contours
    def _export_box(self) -> QWidget:
        box = QGroupBox("Export contours for tracking")
        layout = QVBoxLayout()
        box.setLayout(layout)

        self.export_explanation = WrappedLabel()
        self.export_explanation.setText(
            "Writes one contour file per clip, the same format the Detectron2 "
            "pipeline produces. When it finishes, this session switches to "
            "those contours automatically.' + N + N + '"
            "Clips that already have a contour file are skipped, so a run that "
            "was cancelled can be picked up where it stopped."
        )
        layout.addWidget(self.export_explanation)

        self.reexport = QCheckBox("Re-export clips that already have contours")
        layout.addWidget(self.reexport)

        self.export_button = QPushButton("Export contours")
        self.export_button.clicked.connect(self.run_export)
        layout.addWidget(self.export_button)

        self.colab_hint = WrappedLabel()
        self.colab_hint.setText(
            "Without a GPU here, run this stage in Colab instead: build the "
            "bundle with idtrackerai_d2_bundle, upload it with sam3.pt, and "
            "use section 8 of the notebook. Then come back and load the "
            "contour files through External contours."
        )
        layout.addWidget(self.colab_hint)
        return box

    # ------------------------------------------------------- GPU capability
    @property
    def has_gpu(self) -> bool:
        """Whether SAM 3 can run here: a CUDA GPU and the sam3 package.

        Not ``GpuReport.usable``, which additionally wants Detectron2. SAM 3
        needs neither Detectron2 nor a model of your own.
        """
        return self.gpu.sam3_usable

    def _gpu_text(self) -> str:
        if not self.gpu.checked:
            return "Checking what this machine can do..."
        if self.has_gpu:
            where = self.gpu.device_name or "a CUDA GPU"
            return f"SAM 3 can run here: {where}, sam3 {self.gpu.sam3_version}."
        return (
            "SAM 3 cannot run at full speed here: "
            + "; ".join(self.gpu.sam3_missing)
            + ". Drafting annotations will still work on the CPU, slowly. "
            "Exporting a whole video will not."
        )

    def check_gpu(self) -> None:
        if self.gpu_thread.isRunning():
            return
        self.gpu = gpu_mod.GpuReport()
        self.gpu_status.setText(self._gpu_text())
        self.gpu_thread.start()

    def _gpu_reported(self, report) -> None:
        self.gpu = report
        text = self._gpu_text()
        if not self.has_gpu:
            text += "\n\n" + gpu_mod.SAM3_INSTALL_HINT
        self.gpu_status.setText(text)
        self._refresh_enabled()

    # ----------------------------------------------------------------- context
    def set_video_context(self, video_paths, output_dir=None) -> None:
        self.video_paths = [Path(p) for p in video_paths or []]
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        self._refresh_enabled()

    def enhancement_committed(self, settings: dict) -> None:
        """The export applies the same enhancement the rest of the app does."""
        self._enhancement = dict(settings)

    def settings(self) -> dict:
        if not hasattr(self, "_enhancement"):
            self._enhancement = dict(self.enhancement.settings())
        return self._enhancement

    def _refresh_enabled(self) -> None:
        ready = self.weights is not None and bool(self.prompt.text().strip())
        self.prelabel_button.setEnabled(ready and self.frames_dir is not None)
        # A whole video on CPU is not a slow run, it is an impossible one.
        self.export_button.setEnabled(
            ready and bool(self.video_paths) and self.has_gpu
        )
        self.colab_hint.setVisible(not self.has_gpu)

    # ------------------------------------------------------------- user input
    def choose_weights(self) -> None:
        start = str(self.weights.parent if self.weights else Path.home())
        name, _ = QFileDialog.getOpenFileName(
            self, "Open the SAM 3 checkpoint", start, filter="Checkpoint (*.pt)"
        )
        if not name:
            return
        self.weights = Path(name)
        self.weights_label.setText(str(self.weights))
        self._refresh_enabled()

    # ----------------------------------------------------- the session's model
    def has_chosen_weights(self) -> bool:
        return self.weights is not None

    def export_dir(self) -> Path:
        return self.contours_dir or (self.output_dir or Path.cwd()) / "sam3_contours"

    def set_number_of_animals(self, n: int) -> None:
        self.number_of_animals = int(n or 0)

    def model_spec(self) -> dict | None:
        """The session's [segmentation_model] table, or None without a
        checkpoint. 'Animals per frame' at 'all' is written as 0, SAM 3's
        "keep everything", so tracking checks the files against that."""
        if self.weights is None or not self.prompt.text().strip():
            return None
        spec = dict(self._loaded_spec)
        spec.update(
            backend="sam3",
            weights=str(self.weights),
            prompt=self.prompt.text().strip(),
            score_threshold=round(self.score_threshold.value(), 4),
            max_instances=self.max_instances.value(),
            contours_dir=str(self.export_dir()),
        )
        return spec

    def set_model_spec(self, spec: dict | None) -> None:
        """Restores the panel from a session's [segmentation_model] table."""
        try:
            spec = sm.normalize_model(spec)
        except sm.SegmentationModelError as exc:
            logging.warning("Ignoring [segmentation_model]: %s", exc)
            spec = None
        if spec is None or spec["backend"] != "sam3":
            self._loaded_spec = {}
            return
        self._loaded_spec = {
            key: value
            for key, value in spec.items()
            if value is not None and value != sm.DEFAULTS.get(key)
        }
        self.weights = Path(spec["weights"])
        self.weights_label.setText(str(self.weights))
        self.prompt.setText(spec["prompt"])
        self.score_threshold.setValue(spec["score_threshold"])
        if spec["max_instances"] is not None:
            self.max_instances.setValue(spec["max_instances"])
        else:
            self.max_instances.setValue(self.number_of_animals)
        if spec.get("contours_dir"):
            self.contours_dir = Path(spec["contours_dir"])
        self._refresh_enabled()

    def choose_frames(self) -> None:
        start = str(self.frames_dir or self.output_dir or Path.cwd())
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of sampled frames", start
        )
        if not folder:
            return
        self.frames_dir = Path(folder)
        self.frames_label.setText(str(self.frames_dir))
        self._refresh_enabled()

    # ----------------------------------------------------------------- running
    def run_prelabel(self) -> None:
        if self.frames_dir is None or self.weights is None:
            return
        if not self.has_gpu and QMessageBox.question(
            self,
            "No GPU on this machine",
            "SAM 3 will run on the CPU, which takes minutes per frame rather "
            "than a fraction of one. For a few hundred frames that is hours.\n\n"
            "Draft them anyway?",
        ) != QMessageBox.StandardButton.Yes:
            return

        self.prelabel_thread.set_parameters(
            frames_dir=self.frames_dir,
            weights=self.weights,
            prompt=self.prompt.text().strip(),
            label=self.prompt.text().strip(),
            device="cuda" if self.has_gpu else "cpu",
            score_threshold=self.score_threshold.value(),
            max_instances=self.max_instances.value(),
            overwrite=self.overwrite.isChecked(),
        )
        self.prelabel_thread.start()

    def run_export(self) -> None:
        if self.weights is None or not self.video_paths:
            return
        # contour files are named for the clip's file name, so two clips with one
        # would write the same file and the second be skipped as already done
        shared = sampling_mod.repeated_stems(self.video_paths)
        if shared:
            lines = "\n".join(
                f"{stem}: {', '.join(str(p) for p in paths)}"
                for stem, paths in shared.items()
            )
            QMessageBox.warning(
                self,
                "Clips share a file name",
                "Two videos would write the same contour file (same file name in"
                f" different folders). Export them separately.\n\n{lines}",
            )
            return
        output_dir = self.export_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_thread.set_parameters(
            videos=self.video_paths,
            output_dir=output_dir,
            weights=self.weights,
            prompt=self.prompt.text().strip(),
            device="cuda" if self.has_gpu else "cpu",
            score_threshold=self.score_threshold.value(),
            max_instances=self.max_instances.value(),
            enhancement=self.settings(),
            overwrite=self.reexport.isChecked(),
            tracking_intervals=self.tracking_intervals(),
        )
        self.export_thread.start()

    def _prelabel_finished(self) -> None:
        summary = self.prelabel_thread.summary
        if summary is None:
            return
        message = (
            f"{summary['written']} annotation file(s) written, "
            f"{summary['polygons']} outline(s) drawn."
        )
        if summary["skipped_existing"]:
            message += (
                f"\n\n{summary['skipped_existing']} frame(s) were already "
                "annotated and were left untouched."
            )
        if summary["empty"]:
            message += (
                f"\n\nSAM 3 found nothing in {len(summary['empty'])} frame(s); "
                "those still need annotating by hand."
            )
        message += "\n\nThese are drafts. Correct them in LabelMe before training."
        QMessageBox.information(self, "Annotations drafted", message)

    def _export_finished(self) -> None:
        """Reports what happened, once something else has decided what that is.

        Whether the session can actually use these files is not this panel's
        call: the source widget validates them against the video and refuses a
        set that does not fit, which is exactly what a cancelled batch leaves
        behind. So the outcome is reported by whoever adopts them, and this
        says only what is true either way.
        """
        written = self.export_thread.written
        if not written:
            return

        skipped = self.export_thread.skipped
        detail = f"{len(written) - len(skipped)} contour file(s) written to\n{written[0].parent}"
        if skipped:
            detail += (
                f"\n\n{len(skipped)} clip(s) already had contours and were "
                "left alone."
            )
        QMessageBox.information(self, "Export finished", detail)
        # Adopting them, and saying whether that worked, belongs to whoever
        # receives this.
        self.contoursReady.emit(written)

    # ---------------------------------------------------------------- threads
    def _wire_threads(self) -> None:
        for thread, finished in (
            (self.prelabel_thread, self._prelabel_finished),
            (self.export_thread, self._export_finished),
        ):
            thread.started.connect(lambda t=thread: self._thread_started(t))
            thread.finished.connect(self._thread_finished)
            thread.finished.connect(finished)
            thread.set_progress_value.connect(self._set_progress_value)
            thread.set_progress_max.connect(self._set_progress_max)
            thread.failed.connect(self._thread_failed)
        self.export_thread.set_progress_label.connect(self._set_progress_label)

    def _thread_started(self, thread) -> None:
        self.progress = QProgressDialog("Running SAM 3...", "Cancel", 0, 100, self)
        self.progress.setMinimumDuration(200)
        self.progress.setModal(True)
        self.progress.setAutoReset(False)
        self.progress.canceled.connect(thread.quit)

    def _thread_finished(self) -> None:
        if hasattr(self, "progress"):
            self.progress.reset()

    def _set_progress_value(self, value: int) -> None:
        if hasattr(self, "progress"):
            self.progress.setValue(value)

    def _set_progress_max(self, value: int) -> None:
        if hasattr(self, "progress"):
            self.progress.setMaximum(value)

    def _set_progress_label(self, text: str) -> None:
        if hasattr(self, "progress"):
            self.progress.setLabelText(text)

    def _thread_failed(self, message: str) -> None:
        QMessageBox.critical(self, "SAM 3 failed", message)

    # --------------------------------------------------------------- tooltips
    def setToolTips(self, tips: dict) -> None:
        pairs = [
            (self.weights_button, "sam3_weights"),
            (self.weights_label, "sam3_weights"),
            (self.prompt, "sam3_prompt"),
            (self.score_threshold, "sam3_score_threshold"),
            (self.max_instances, "sam3_max_instances"),
            (self.frames_button, "sam3_frames"),
            (self.prelabel_button, "sam3_prelabel"),
            (self.overwrite, "sam3_overwrite"),
            (self.export_button, "sam3_export"),
            (self.reexport, "sam3_reexport"),
        ]
        for widget, key in pairs:
            if key in tips:
                widget.setToolTip(tips[key])

    # ---------------------------------------------------------------- closing
    def close(self) -> bool:
        """Stops any running work before the app exits."""
        for thread in (self.prelabel_thread, self.export_thread, self.gpu_thread):
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        return super().close()
