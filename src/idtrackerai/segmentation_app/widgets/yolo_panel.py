"""The YOLO mode: contours from a YOLO segmentation model you trained.

Two ways in, because there are two things a YOLO run can leave you with:

**Export contours** runs the model's ``best.pt`` over this session's clips and
writes the same sidecar the Detectron2 and SAM 3 routes write, so tracking
reads it through the External contours path that is already tested.

**Import label files** takes the outlines of a run that was already made
elsewhere (``save_txt=True``) and writes the same files from them, without
running the model again.

What neither can use is a table of centre points or boxes: idtracker.ai tells
animals apart from the pixels inside each outline, so it needs the outlines.

Where the work happens
----------------------
YOLO is light next to SAM 3: on a CUDA GPU it keeps up with the video, and on
a CPU it still manages a few frames a second. So the export is offered
wherever the ultralytics package is installed, with a warning rather than a
refusal when there is no GPU, and the same command is shown for running it in
Colab or on another machine.
"""

import logging
import sys
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import gpu as gpu_mod
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm
from idtrackerai.extra_tools.detectron2_pipeline import yolo_import
from idtrackerai.GUI_tools import WrappedLabel

from .colab_notebook_export import ColabNotebookExport
from .pipeline_threads import GpuCheckThread, YoloExportThread


class YoloPanel(ColabNotebookExport, QWidget):
    """Everything the YOLO route needs, and nothing the other modes provide."""

    contoursReady = Signal(object)  # list[Path] written by an export or import

    def __init__(self, enhancement):
        super().__init__()
        self.enhancement = enhancement
        self.video_paths: list[Path] = []
        self.output_dir: Path | None = None
        self.weights: Path | None = None
        # Where contours go. Set from a .toml's contours_dir; otherwise
        # yolo_contours beside the session output.
        self.contours_dir: Path | None = None
        self.number_of_animals = 0
        # returns the session's tracking intervals, so only those frames run
        self.tracking_intervals = lambda: None
        # [segmentation_model] keys with no control here, kept from the .toml
        self._loaded_spec: dict = {}

        self.gpu = gpu_mod.GpuReport()
        self.export_thread = YoloExportThread()
        self.gpu_thread = GpuCheckThread()
        self.gpu_thread.reported.connect(self._gpu_reported)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._model_box())
        layout.addWidget(self._export_box())
        layout.addWidget(self._import_box())
        layout.addStretch()

        self._wire_threads()
        self._refresh_enabled()
        self.check_gpu()

    # ------------------------------------------------------------ the model
    def _model_box(self) -> QWidget:
        box = QGroupBox("YOLO")
        grid = QGridLayout()
        box.setLayout(grid)

        self.weights_label = WrappedLabel(framed=True)
        self.weights_label.setText("No best.pt chosen")
        self.weights_button = QPushButton("Choose best.pt")
        self.weights_button.clicked.connect(self.choose_weights)

        self.score_threshold = QDoubleSpinBox()
        self.score_threshold.setRange(0.0, 1.0)
        self.score_threshold.setSingleStep(0.05)
        self.score_threshold.setValue(0.5)
        self.score_threshold.valueChanged.connect(
            lambda _v: self._update_export_command()
        )

        # 0 follows the session, so a pen whose count changes by day gets the
        # right cap each day without anyone retyping it here.
        self.max_instances = QSpinBox()
        self.max_instances.setRange(0, 1000)
        self.max_instances.setValue(0)
        self.max_instances.setSpecialValueText("number of animals")
        self.max_instances.valueChanged.connect(
            lambda _v: self._update_export_command()
        )

        # 0 leaves it to the model: the size it was trained at.
        self.imgsz = QSpinBox()
        self.imgsz.setRange(0, 8192)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(0)
        self.imgsz.setSpecialValueText("as trained")
        self.imgsz.valueChanged.connect(lambda _v: self._update_export_command())

        self.gpu_status = WrappedLabel()
        self.gpu_status.setText(self._gpu_text())

        grid.addWidget(QLabel("Weights"), 0, 0)
        grid.addWidget(self.weights_label, 0, 1)
        grid.addWidget(self.weights_button, 0, 2)
        grid.addWidget(QLabel("Confidence"), 1, 0)
        grid.addWidget(self.score_threshold, 1, 1)
        grid.addWidget(QLabel("Animals per frame"), 2, 0)
        grid.addWidget(self.max_instances, 2, 1)
        grid.addWidget(QLabel("Image size"), 3, 0)
        grid.addWidget(self.imgsz, 3, 1)
        grid.addWidget(self.gpu_status, 4, 0, 1, 3)
        return box

    # ------------------------------------------------------ exporting contours
    def _export_box(self) -> QWidget:
        box = QGroupBox("Export contours for tracking")
        layout = QVBoxLayout()
        box.setLayout(layout)

        explanation = WrappedLabel()
        explanation.setText(
            "Runs the model over this session's clips and writes one contour "
            "file per clip, the same format the Detectron2 and SAM 3 routes "
            "produce. When it finishes, this session switches to those "
            "contours automatically.\n\n"
            "Clips that already have a contour file are skipped, so a run that "
            "was cancelled can be picked up where it stopped."
        )
        layout.addWidget(explanation)

        row = QHBoxLayout()
        row.addWidget(QLabel("Contours folder"))
        self.contours_field = QLineEdit()
        self.contours_field.setReadOnly(True)
        self.contours_button = QPushButton("Choose folder...")
        self.contours_button.clicked.connect(self.choose_contours_dir)
        row.addWidget(self.contours_field)
        row.addWidget(self.contours_button)
        layout.addLayout(row)

        self.reexport = QCheckBox("Re-export clips that already have contours")
        layout.addWidget(self.reexport)

        self.export_button = QPushButton("Export contours")
        self.export_button.clicked.connect(self.run_export)
        layout.addWidget(self.export_button)

        command_label = WrappedLabel()
        command_label.setText(
            "Or run the same export from a terminal, on a machine with a GPU:"
        )
        layout.addWidget(command_label)
        self.export_command = QPlainTextEdit()
        self.export_command.setReadOnly(True)
        self.export_command.setMaximumHeight(96)
        layout.addWidget(self.export_command)
        self.copy_command_button = QPushButton("Copy command")
        self.copy_command_button.clicked.connect(self.copy_export_command)
        layout.addWidget(self.copy_command_button)

        notebook_label = WrappedLabel()
        notebook_label.setText(
            "Or run it on Colab: one notebook with this session's settings "
            "in it, which only needs the Drive folder of the videos and weights."
        )
        layout.addWidget(notebook_label)
        layout.addWidget(self.make_notebook_button())

        self.colab_hint = WrappedLabel()
        self.colab_hint.setText(
            "Without a GPU here the export runs at a few frames per second. "
            "For whole videos, run the command above on a machine that has "
            "one, or use Save Colab notebook: it writes this session as a "
            "notebook that segments and tracks on a Colab GPU."
        )
        layout.addWidget(self.colab_hint)
        return box

    # ------------------------------------------------------ importing outlines
    def _import_box(self) -> QWidget:
        box = QGroupBox("Or import outlines YOLO already wrote")
        layout = QVBoxLayout()
        box.setLayout(layout)

        explanation = WrappedLabel()
        explanation.setText(
            "For a prediction that was already run on these clips with "
            "save_txt: choose its labels folder, holding one "
            "<clip>_<frame>.txt per frame, and the outlines in it become this "
            "session's contours. The files must come from a segmentation "
            "model; boxes and centre points cannot be tracked from."
        )
        layout.addWidget(explanation)
        self.import_button = QPushButton("Import label files...")
        self.import_button.clicked.connect(self.choose_labels)
        layout.addWidget(self.import_button)
        return box

    # ------------------------------------------------------- GPU capability
    @property
    def has_gpu(self) -> bool:
        """Whether YOLO runs at full speed here: a CUDA GPU and ultralytics."""
        return self.gpu.yolo_usable

    @property
    def can_run(self) -> bool:
        """Whether YOLO can run here at all. A CPU is slow, not impossible."""
        return self.gpu.ultralytics_version is not None

    def _gpu_text(self) -> str:
        if not self.gpu.checked:
            return "Checking what this machine can do..."
        if self.has_gpu:
            where = self.gpu.device_name or "a CUDA GPU"
            return (
                f"YOLO can run here: {where},"
                f" ultralytics {self.gpu.ultralytics_version}."
            )
        text = "YOLO cannot run at full speed here: " + "; ".join(
            self.gpu.yolo_missing
        )
        if self.can_run:
            return text + ". The export will run on the CPU, slowly."
        return text + ". Label files can still be imported."

    def check_gpu(self) -> None:
        if self.gpu_thread.isRunning():
            return
        self.gpu = gpu_mod.GpuReport()
        self.gpu_status.setText(self._gpu_text())
        self.gpu_thread.start()

    def _gpu_reported(self, report) -> None:
        self.gpu = report
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
        self._update_export_command()

    def settings(self) -> dict:
        if not hasattr(self, "_enhancement"):
            self._enhancement = dict(self.enhancement.settings())
        return self._enhancement

    def set_number_of_animals(self, n: int) -> None:
        self.number_of_animals = int(n or 0)
        self._update_export_command()

    def effective_max_instances(self) -> int:
        """The cap an export uses: the one set here, else the session's
        number of animals, else 0 -- keep everything the model reports."""
        return self.max_instances.value() or self.number_of_animals

    def _refresh_enabled(self) -> None:
        ready = self.weights is not None and bool(self.video_paths)
        self.export_button.setEnabled(ready and self.can_run)
        self.import_button.setEnabled(bool(self.video_paths))
        self.colab_hint.setVisible(not self.has_gpu)
        text = self._gpu_text()
        if self.gpu.checked and not self.has_gpu:
            text += "\n\n" + gpu_mod.YOLO_INSTALL_HINT
        self.gpu_status.setText(text)
        self.contours_field.setText(str(self.export_dir()))
        self._update_export_command()

    # ------------------------------------------------------------- user input
    def _set_weights(self, weights: Path | None) -> None:
        self.weights = weights
        self.weights_label.setText(
            "No best.pt chosen" if weights is None else str(weights)
        )
        self._refresh_enabled()

    def choose_weights(self) -> None:
        start = str(self.weights.parent if self.weights else Path.home())
        name, _ = QFileDialog.getOpenFileName(
            self, "Open the YOLO segmentation weights", start, filter="Weights (*.pt)"
        )
        if name:
            self._set_weights(Path(name))

    def choose_contours_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder for the contour files", str(self.export_dir())
        )
        if not folder:
            return
        self.contours_dir = Path(folder)
        self._refresh_enabled()

    # ----------------------------------------------------- the session's model
    def has_chosen_weights(self) -> bool:
        return self.weights is not None

    def export_dir(self) -> Path:
        return self.contours_dir or (self.output_dir or Path.cwd()) / "yolo_contours"

    def model_spec(self) -> dict | None:
        """The session's [segmentation_model] table, or None without weights.

        'Animals per frame' and 'Image size' are written only when set here:
        left out, tracking takes the cap from number_of_animals and accepts
        the size the model was trained at.
        """
        if self.weights is None:
            return None
        spec = dict(self._loaded_spec)
        for key in ("max_instances", "imgsz"):
            spec.pop(key, None)
        spec.update(
            backend="yolo",
            weights=str(self.weights),
            score_threshold=round(self.score_threshold.value(), 4),
            contours_dir=str(self.export_dir()),
        )
        if self.max_instances.value():
            spec["max_instances"] = self.max_instances.value()
        if self.imgsz.value():
            spec["imgsz"] = self.imgsz.value()
        return spec

    def set_model_spec(self, spec: dict | None) -> None:
        """Restores the panel from a session's [segmentation_model] table."""
        try:
            spec = sm.normalize_model(spec)
        except sm.SegmentationModelError as exc:
            logging.warning("Ignoring [segmentation_model]: %s", exc)
            spec = None
        if spec is None or spec["backend"] != "yolo":
            self._loaded_spec = {}
            return
        self._loaded_spec = {
            key: value
            for key, value in spec.items()
            if value is not None and value != sm.DEFAULTS.get(key)
        }
        self.score_threshold.setValue(spec["score_threshold"])
        self.max_instances.setValue(spec["max_instances"] or 0)
        self.imgsz.setValue(spec["imgsz"] or 0)
        if spec.get("contours_dir"):
            self.contours_dir = Path(spec["contours_dir"])
        self._set_weights(Path(spec["weights"]))  # refreshes everything else

    # ----------------------------------------------------------------- running
    def _clips_sharing_a_name(self) -> bool:
        """Contour files are named for the clip, so two clips with one name
        would write the same file and the second be skipped as already done."""
        shared = sampling_mod.repeated_stems(self.video_paths)
        if not shared:
            return False
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
        return True

    def run_export(self) -> None:
        if self.weights is None or not self.video_paths:
            return
        if self._clips_sharing_a_name():
            return
        if not self.has_gpu and QMessageBox.question(
            self,
            "No GPU on this machine",
            "YOLO will run on the CPU, at a few frames per second. A short "
            "clip is fine; a whole recording takes hours.\n\nExport anyway?",
        ) != QMessageBox.StandardButton.Yes:
            return
        output_dir = self.export_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_thread.set_parameters(
            videos=self.video_paths,
            output_dir=output_dir,
            weights=self.weights,
            device="cuda" if self.has_gpu else "cpu",
            score_threshold=self.score_threshold.value(),
            max_instances=self.effective_max_instances(),
            imgsz=self.imgsz.value() or None,
            classes=self._loaded_spec.get("classes"),
            on_overlap=self._loaded_spec.get("on_overlap", "merge"),
            enhancement=self.settings(),
            overwrite=self.reexport.isChecked(),
            tracking_intervals=self.tracking_intervals(),
        )
        self.export_thread.start()

    def choose_labels(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "The labels folder of a YOLO segmentation prediction",
            str(self.output_dir or Path.cwd()),
        )
        if not folder:
            return
        problem = self.import_labels(Path(folder))
        if problem:
            QMessageBox.warning(self, "Could not import these label files", problem)

    def import_labels(self, folder: Path) -> str | None:
        """Writes this session's contour files from a labels folder.

        Returns what went wrong, or None after handing the files over through
        ``contoursReady``. Nothing is handed over unless every clip imported:
        a session tracked from some clips' contours would be missing animals.
        """
        if not self.video_paths:
            return "Open the session's video first."
        if self._clips_sharing_a_name():
            return None
        output_dir = self.export_dir()
        written: list[Path] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for video in self.video_paths:
                output = output_dir / f"{video.stem}.h5"
                yolo_import.import_video(
                    video,
                    folder,
                    output,
                    score_threshold=self.score_threshold.value(),
                    max_instances=self.effective_max_instances() or None,
                    classes=self._loaded_spec.get("classes"),
                    weights_name=self.weights.name if self.weights else None,
                )
                written.append(output)
        except yolo_import.YoloImportError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            logging.exception("YOLO label import failed")
            return str(exc)
        finally:
            QApplication.restoreOverrideCursor()
        self.contoursReady.emit(written)
        return None

    # ------------------------------------------------------- the command line
    @staticmethod
    def _exporter() -> Path:
        """The exporter's path, run through this interpreter, as the other
        panels do: its console script is often not on PATH."""
        return Path(gpu_mod.__file__).with_name("inference.py")

    def _update_export_command(self) -> None:
        """What 'Export contours' would do, as a command for another machine.

        Confidence and animals per frame are always written, because the
        exporter's own defaults (0.7, and 5 per frame) are not this panel's.
        The enhancement is written too: off, unless it is on in this session.
        """
        if not hasattr(self, "export_command"):
            return  # still being built
        weights = '"<best.pt>"' if self.weights is None else f'"{self.weights}"'
        videos = (
            " ".join(f'"{video}"' for video in self.video_paths)
            or '"<the session\'s videos>"'
        )
        parts = [
            f'"{sys.executable}" "{self._exporter()}"',
            "--backend yolo",
            f"--weights {weights}",
            f"--videos {videos}",
            f'--output-dir "{self.export_dir()}"',
            f"--score-threshold {round(self.score_threshold.value(), 4):g}",
            f"--max-instances {self.effective_max_instances()}",
        ]
        if self.imgsz.value():
            parts.append(f"--imgsz {self.imgsz.value()}")
        parts.extend(fp.settings_to_flags(self.settings()))
        self.export_command.setPlainText(" ".join(parts))

    def copy_export_command(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(self.export_command.toPlainText())
        self.copy_command_button.setText("Copied")
        QTimer.singleShot(
            1500, lambda: self.copy_command_button.setText("Copy command")
        )

    def _export_finished(self) -> None:
        """Says what was written. Whether the session can use the files is
        decided, and reported, by whoever adopts them."""
        written = self.export_thread.written
        if not written:
            return
        skipped = self.export_thread.skipped
        detail = (
            f"{len(written) - len(skipped)} contour file(s) written to\n"
            f"{written[0].parent}"
        )
        if skipped:
            detail += (
                f"\n\n{len(skipped)} clip(s) already had contours and were "
                "left alone."
            )
        QMessageBox.information(self, "Export finished", detail)
        self.contoursReady.emit(written)

    # ---------------------------------------------------------------- threads
    def _wire_threads(self) -> None:
        thread = self.export_thread
        thread.started.connect(self._thread_started)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(self._export_finished)
        thread.set_progress_value.connect(self._set_progress_value)
        thread.set_progress_max.connect(self._set_progress_max)
        thread.set_progress_label.connect(self._set_progress_label)
        thread.failed.connect(self._thread_failed)

    def _thread_started(self) -> None:
        self.progress = QProgressDialog("Running YOLO...", "Cancel", 0, 100, self)
        self.progress.setMinimumDuration(200)
        self.progress.setModal(True)
        self.progress.setAutoReset(False)
        self.progress.canceled.connect(self.export_thread.quit)

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
        QMessageBox.critical(self, "YOLO failed", message)

    # --------------------------------------------------------------- tooltips
    def setToolTips(self, tips: dict) -> None:
        pairs = [
            (self.weights_button, "yolo_weights"),
            (self.weights_label, "yolo_weights"),
            (self.score_threshold, "yolo_score_threshold"),
            (self.max_instances, "yolo_max_instances"),
            (self.imgsz, "yolo_imgsz"),
            (self.export_button, "yolo_export"),
            (self.reexport, "yolo_reexport"),
            (self.contours_button, "yolo_contours_dir"),
            (self.contours_field, "yolo_contours_dir"),
            (self.export_command, "yolo_command"),
            (self.copy_command_button, "yolo_command"),
            (self.notebook_button, "yolo_notebook"),
            (self.import_button, "yolo_import"),
        ]
        for widget, key in pairs:
            if key in tips:
                widget.setToolTip(tips[key])

    # ---------------------------------------------------------------- closing
    def close(self) -> bool:
        """Stops any running work before the app exits."""
        for thread in (self.export_thread, self.gpu_thread):
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        return super().close()
