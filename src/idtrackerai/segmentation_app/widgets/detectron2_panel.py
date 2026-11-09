"""The guided panel for preparing a Detectron2 model.

Six steps: choose the enhancement, sample frames, annotate them, build the
dataset, train, export contours. The first four run anywhere. The last two
need a CUDA GPU, and the panel checks whether this machine has one rather than
assuming it does not: Colab is the fallback, not the route.

Training runs here as a managed subprocess, so its output is visible and it
can be stopped, without importing detectron2 into the GUI process. The export
is handed over as a command instead, because it takes days over a collection
and that does not belong behind a window the user cannot close.

Each step records what it produced, so the app can be closed in the middle of
a job that takes days. Status is re-derived from disk on every load rather than
trusted from the state file, so a deleted folder shows as incomplete.
"""

import importlib.util
import logging
import sys
from pathlib import Path

from qtpy.QtCore import (  # type: ignore[reportPrivateImportUsage]
    QProcess,
    QProcessEnvironment,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import bundle as bundle_mod
from idtrackerai.extra_tools.detectron2_pipeline import dataset as dataset_mod
from idtrackerai.extra_tools.detectron2_pipeline import gpu as gpu_mod
from idtrackerai.extra_tools.detectron2_pipeline import install_gpu as install_mod
from idtrackerai.extra_tools.detectron2_pipeline import prep_state as prep_state_mod
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline import segmentation_model as sm
from idtrackerai.extra_tools.detectron2_pipeline.errors import SamplingError
from idtrackerai.GUI_tools import WrappedLabel

from .enhancement_widget import EnhancementWidget
from .pipeline_threads import (
    DatasetThread,
    FrameCountThread,
    GpuCheckThread,
    SamplingThread,
)


class Detectron2Panel(QWidget):
    """Walks the user from raw video to a training-ready COCO dataset."""

    needToDraw = Signal()

    def __init__(self, enhancement: EnhancementWidget):
        super().__init__()
        self.enhancement = enhancement
        self.state: prep_state_mod.PrepState | None = None
        self.number_of_animals = 0
        # [segmentation_model] keys this panel has no control for, kept from
        # the loaded .toml so saving it again does not drop them
        self._loaded_spec: dict = {}

        # What the Segmentation App currently has open, kept apart from the
        # annotation list so the panel can tell "the user has not touched this"
        # from "the user built this deliberately".
        self._open_videos: list[Path] = []
        self._counts: dict[Path, int] = {}
        self._count_keys: dict[Path, tuple] = {}
        self._pending_counts: set[Path] = set()
        # What each listed clip's frames are called; see _refresh_names
        self._names: dict[Path, str] = {}
        self._names_problem = ""  # why the output folder cannot be sampled into

        self.sampling_thread = SamplingThread()
        self.dataset_thread = DatasetThread()
        self.count_thread = FrameCountThread()
        self.count_thread.counted.connect(self._counts_ready)
        # anything queued while that run was in flight goes now
        self.count_thread.finished.connect(self._drain_count_queue)

        # Counting frames opens every file, so the preview waits for the
        # spinbox to settle rather than recomputing on each keystroke.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self.update_preview)

        self.gpu = gpu_mod.GpuReport()
        self.gpu_thread = GpuCheckThread()
        self.gpu_thread.reported.connect(self._gpu_reported)
        self.train_process: QProcess | None = None

        self.steps = QToolBox()
        self.steps.addItem(self._enhancement_page(), "1. Enhancement")
        self.steps.addItem(self._sampling_page(), "2. Sample frames")
        self.steps.addItem(self._annotate_page(), "3. Annotate")
        self.steps.addItem(self._dataset_page(), "4. Build dataset")
        self.steps.addItem(self._train_page(), "5. Train")
        self.steps.addItem(self._export_page(), "6. Export contours")

        # No standing hand-off paragraph any more. It said "the GPU stages
        # happen on Colab", which steps 5 and 6 now say for themselves and
        # with live information; keeping it cost two lines of height that the
        # sampling step needs for its own button.
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.steps)

        self._wire_threads()
        self.steps.currentChanged.connect(lambda _i: self._fit_to_open_step())
        self.enhancement.profileSaved.connect(self._profile_saved)
        self.enhancement_committed(self.enhancement.settings())
        self._update_export_command()
        self._fit_to_open_step()
        self.check_gpu()
        # Last, because setText is called while the pages are built and from
        # saved state, and the handler needs every page to exist
        self.frames_dir.textChanged.connect(self._output_folder_changed)

    def _fit_to_open_step(self) -> None:
        """Ask for the height the open step actually needs.

        QToolBox reports the same modest sizeHint whatever it holds, so the
        layout happily gave the panel 248px for a step needing 280 and the
        step then scrolled internally -- hiding its own button. The left
        column scrolls, so asking for the real height costs nothing worse
        than a scrollbar on the column.
        """
        page = self.steps.currentWidget()
        if page is None:
            return
        # one header per step, plus what the open one needs
        headers = 34 * self.steps.count()
        self.steps.setMinimumHeight(headers + page.sizeHint().height() + 8)

    # ------------------------------------------------------------------ pages
    def _enhancement_page(self) -> QWidget:
        """The enhancement lives above, as its own control.

        It applies to thresholding as well as to this pipeline, so it is not
        owned by this panel. This page just says where it is and what the
        current choice means for the frames about to be sampled.
        """
        page = QWidget()
        self.enhancement_status = WrappedLabel()
        layout = QVBoxLayout(page)
        layout.addWidget(self.enhancement_status)
        return page

    def _sampling_page(self) -> QWidget:
        page = QWidget()

        # The clips to draw frames from are this panel's own list, not the
        # tracking session's. A model should see every recording from a rig,
        # while a session is the one recording being tracked; opening fifty
        # clips just to annotate them would make one absurd concatenated
        # session out of fifty separate experiments.
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # Three rows is enough to show that a list exists and to scroll it.
        # The page has to leave room for the button that acts on the list, and
        # a step whose own action needs scrolling to reach is a bad step.
        self.video_list.setMaximumHeight(58)
        self.add_videos_button = QPushButton("Add videos...")
        self.add_videos_button.clicked.connect(self.add_videos)
        self.add_folder_button = QPushButton("Add folder...")
        self.add_folder_button.clicked.connect(self.add_video_folder)
        # Off by default: a folder of clips and a folder of whole experiments
        # look the same until you open them, and taking every video beneath the
        # wrong one is a bigger mistake than having to tick a box.
        self.include_subfolders = QCheckBox("Include subfolders")
        self.remove_videos_button = QPushButton("Remove")
        self.remove_videos_button.clicked.connect(self.remove_selected_videos)
        self.reset_videos_button = QPushButton("Use open videos")
        self.reset_videos_button.clicked.connect(self.reset_videos_to_open)
        self.video_summary = WrappedLabel()

        video_buttons = QHBoxLayout()
        video_buttons.setContentsMargins(0, 0, 0, 0)
        for button in (self.add_videos_button, self.add_folder_button,
                       self.remove_videos_button, self.reset_videos_button):
            video_buttons.addWidget(button)

        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, 100000)
        self.n_frames.setValue(600)
        self.n_frames.valueChanged.connect(self._schedule_preview)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.frames_dir = QLineEdit()
        self.frames_dir.setPlaceholderText("chosen when you run this step")
        self.frames_dir.setReadOnly(True)
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_frames_dir)

        self.sample_button = QPushButton("Sample frames")
        self.sample_button.clicked.connect(self.run_sampling)
        self.sample_status = WrappedLabel()
        self.sample_preview = WrappedLabel(framed=True)

        # Both numbers share a row: they are small, and the page has to fit
        # the button that acts on them without scrolling.
        self.seed_label = QLabel("Selection number")
        counts_row = QHBoxLayout()
        counts_row.setContentsMargins(0, 0, 0, 0)
        counts_row.addWidget(self.n_frames, 1)
        counts_row.addSpacing(8)
        counts_row.addWidget(self.seed_label)
        counts_row.addWidget(self.seed, 1)
        counts_holder = QWidget()
        counts_holder.setLayout(counts_row)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Frames to sample", counts_holder)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.frames_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        self.video_list_label = QLabel("Videos to sample from")

        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.video_list_label)
        layout.addWidget(self.video_list)
        layout.addLayout(video_buttons)
        layout.addWidget(self.include_subfolders)
        layout.addWidget(self.video_summary)
        layout.addLayout(form)
        layout.addWidget(self.sample_preview)
        layout.addWidget(self.sample_button)
        layout.addWidget(self.sample_status)
        return page

    def _annotate_page(self) -> QWidget:
        page = QWidget()
        self.class_name = QLineEdit("fish")
        self.launch_button = QPushButton("Open in LabelMe")
        self.launch_button.clicked.connect(self.launch_labelme)
        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self.open_frames_folder)
        self.refresh_button = QPushButton("Refresh count")
        self.refresh_button.clicked.connect(self.refresh)
        # Where the Colab route begins. Annotating is the last step that wants
        # eyes on the footage, so this is the moment someone without a CUDA
        # card needs the notebook -- not buried in the docs.
        self.bundle_button = QPushButton("Get Colab bundle...")
        self.bundle_button.clicked.connect(self.save_colab_bundle)
        self.annotate_status = WrappedLabel()

        form = QFormLayout()
        form.addRow("Class name", self.class_name)

        row = QHBoxLayout()
        row.addWidget(self.launch_button)
        row.addWidget(self.open_folder_button)
        row.addWidget(self.refresh_button)
        row.addWidget(self.bundle_button)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(self.annotate_status)
        return page

    def _dataset_page(self) -> QWidget:
        page = QWidget()
        self.expected_instances = QSpinBox()
        self.expected_instances.setRange(0, 1000)
        self.expected_instances.setSpecialValueText("any")
        self.val_fraction = QDoubleSpinBox()
        self.val_fraction.setRange(0.0, 0.9)
        self.val_fraction.setSingleStep(0.05)
        self.val_fraction.setValue(0.15)
        self.dataset_dir = QLineEdit()
        self.dataset_dir.setReadOnly(True)
        self.dataset_dir.setPlaceholderText("chosen when you run this step")
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_dataset_dir)

        # How clips are grouped decides whether the validation score means
        # anything. Segments of one recording show the same animals minutes
        # apart, so splitting between them measures memorisation. The grouping
        # is guessed from file names, which cannot always be right, so it is
        # shown rather than applied quietly.
        self.group_by = QComboBox()
        self.group_by.addItem("By recording (groups segments together)", "recording")
        self.group_by.addItem("By file (each clip on its own)", "file")
        self.group_by.addItem("Random (not recommended)", "random")
        self.group_by.currentIndexChanged.connect(self._group_by_changed)
        self.grouping_status = WrappedLabel(framed=True)
        self.edit_grouping_button = QPushButton("Edit grouping...")
        self.edit_grouping_button.clicked.connect(self.edit_grouping)

        self.build_button = QPushButton("Build dataset")
        self.build_button.clicked.connect(self.run_dataset)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMaximumHeight(160)
        self.report_view.setPlaceholderText("the validation report appears here")

        form = QFormLayout()
        form.addRow("Animals per frame", self.expected_instances)
        form.addRow("Validation fraction", self.val_fraction)
        group_row = QHBoxLayout()
        group_row.setContentsMargins(0, 0, 0, 0)
        group_row.addWidget(self.group_by, 1)
        group_row.addWidget(self.edit_grouping_button)
        group_holder = QWidget()
        group_holder.setLayout(group_row)
        form.addRow("Train/validation split", group_holder)
        row = QHBoxLayout()
        row.addWidget(self.dataset_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addWidget(self.grouping_status)
        layout.addWidget(self.build_button)
        layout.addWidget(self.report_view)
        return page

    def _group_by_changed(self) -> None:
        if self.state is not None:
            self.state.group_by = self.group_by.currentData()
            self.state.save()
        self._update_grouping_status()
        self._update_video_summary()

    def edit_grouping(self) -> None:
        """Correct the guessed grouping, clip by clip.

        The rule reads file names, so it can only ever guess. Two clips that
        happen to be named like segments of one recording, or segments named
        so that nothing marks them as such, both end up wrong, and nothing in
        the name can settle it. This is where the person who ran the
        experiment says which is which.
        """
        videos = self.video_paths
        if not videos:
            QMessageBox.information(
                self, "No videos", "Add the videos this corpus draws from first."
            )
            return

        overrides = dict(self.state.group_overrides) if self.state else {}
        mode = self.group_by.currentData()
        mode = "file" if mode == "random" else mode

        dialog = QDialog(self)
        dialog.setWindowTitle("Grouping for the train/validation split")
        explain = WrappedLabel()
        explain.setText(
            "Clips sharing a group never end up on opposite sides of the "
            "split. Edit the group of any clip that was guessed wrongly; "
            "clearing a cell restores the guess."
        )
        names = self.clip_names()
        table = QTableWidget(len(videos), 2)
        table.setHorizontalHeaderLabels(["Clip", "Group"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for row, video in enumerate(videos):
            clip = names[video]
            name = QTableWidgetItem(clip)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name)
            table.setItem(
                row, 1,
                QTableWidgetItem(dataset_mod.group_of(clip, overrides, mode)),
            )
        table.resizeColumnsToContents()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.addWidget(explain)
        layout.addWidget(table)
        layout.addWidget(buttons)
        dialog.resize(520, 420)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Only what differs from the guess is stored, so improving the rule
        # later still benefits anyone who did not have to correct it.
        new_overrides: dict[str, str] = {}
        for row, video in enumerate(videos):
            clip = names[video]
            chosen = (table.item(row, 1).text() or "").strip()
            if chosen and chosen != dataset_mod.group_of(clip, None, mode):
                new_overrides[clip] = chosen
        if self.state is not None:
            self.state.group_overrides = new_overrides
            self.state.save()
        self._update_grouping_status()
        self._update_video_summary()

    def _update_grouping_status(self) -> None:
        """Shows the grouping that was inferred, before it is acted on."""
        mode = self.group_by.currentData()
        if mode == "random":
            self.grouping_status.setText(
                "Frames will be split at random. Frames from one clip are near "
                "duplicates, so validation will flatter the model."
            )
            return

        videos = self.video_paths
        if not videos:
            self.grouping_status.setText("Add videos to see how they group.")
            return

        names = self.clip_names()
        groups = dataset_mod.group_videos(
            [names[v] for v in videos],
            self.state.group_overrides if self.state else None,
            mode,
        )
        shown = ", ".join(
            f"{name} ({len(members)})" for name, members in sorted(groups.items())[:6]
        )
        text = f"{len(videos)} clips in {len(groups)} groups: {shown}"
        text += ", ..." if len(groups) > 6 else "."
        if mode == "recording" and len(groups) == 1 < len(videos):
            text += (
                " That puts every clip in one group, which leaves nothing to "
                "split on, so it will fall back to splitting by file."
            )
        elif len(groups) < 2:
            text += " At least two groups are needed to split on them."
        self.grouping_status.setText(text)

    # -------------------------------------------------------- steps 5 and 6
    def _train_page(self) -> QWidget:
        """Training, run here when this machine can, on Colab when it cannot.

        Colab exists because most people tracking animals have no CUDA card,
        not because the work has to leave the machine. When the card is there,
        uploading tens of gigabytes of video to a hosted runtime to answer a
        question this computer could answer is the wrong default.
        """
        page = QWidget()
        self.gpu_status = WrappedLabel(framed=True)
        self.gpu_status.setText(self.gpu.summary())

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(40)
        self.model_dir = QLineEdit()
        self.model_dir.setReadOnly(True)
        self.model_dir.setPlaceholderText("chosen when you run this step")
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_model_dir)

        self.train_button = QPushButton("Train here")
        self.train_button.clicked.connect(self.run_training)
        self.train_button.setEnabled(False)
        self.stop_train_button = QPushButton("Stop")
        self.stop_train_button.clicked.connect(self.stop_training)
        self.stop_train_button.setEnabled(False)
        self.recheck_button = QPushButton("Check again")
        self.recheck_button.clicked.connect(self.check_gpu)
        self.install_button = QPushButton("Install what is missing")
        self.install_button.clicked.connect(self.install_gpu_stack)
        self.install_button.setVisible(False)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(110)
        self.train_log.setPlaceholderText("training output appears here")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Epochs", self.epochs)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.model_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Model folder", holder)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.stop_train_button)
        buttons.addWidget(self.install_button)
        buttons.addWidget(self.recheck_button)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.gpu_status)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.train_log)
        return page

    def _export_page(self) -> QWidget:
        """The export is handed over rather than run here.

        One clip of 30 000 frames takes about an hour; a collection takes
        days. That does not belong behind a window the user cannot close, so
        the app builds the command and they run it where they like. It is
        resumable, so it can be stopped and restarted as often as needed.
        """
        page = QWidget()
        self.weights_hint = WrappedLabel()
        self.weights_hint.setText(
            "Already have trained weights? Choose them here; steps 1 to 5 are"
            " only for training a new model. They are saved in the session"
            " .toml as its [segmentation_model]."
        )
        self.weights_path = QLineEdit()
        self.weights_path.setPlaceholderText("<model folder>/model_final.pth")
        self.weights_path.editingFinished.connect(self._update_export_command)
        weights_browse = QPushButton("Choose file...")
        weights_browse.clicked.connect(self._choose_weights)
        self.score_threshold = QDoubleSpinBox()
        self.score_threshold.setRange(0.05, 1.0)
        self.score_threshold.setSingleStep(0.05)
        self.score_threshold.setDecimals(2)
        self.score_threshold.setValue(sm.DEFAULTS["score_threshold"])
        self.score_threshold.valueChanged.connect(
            lambda _v: self._update_export_command()
        )

        self.export_status = WrappedLabel()
        self.export_command = QPlainTextEdit()
        self.export_command.setReadOnly(True)
        self.export_command.setMaximumHeight(96)
        self.copy_command_button = QPushButton("Copy command")
        self.copy_command_button.clicked.connect(self.copy_export_command)
        self.contours_dir = QLineEdit()
        self.contours_dir.setReadOnly(True)
        contours_browse = QPushButton("Choose folder...")
        contours_browse.clicked.connect(self._choose_contours_dir)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        for label, field, button in (
            ("Weights", self.weights_path, weights_browse),
            ("Contours folder", self.contours_dir, contours_browse),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(field)
            row.addWidget(button)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
        form.addRow("Score threshold", self.score_threshold)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.weights_hint)
        layout.addWidget(self.export_status)
        layout.addLayout(form)
        layout.addWidget(self.export_command)
        layout.addWidget(self.copy_command_button)
        return page

    def save_colab_bundle(self) -> None:
        """Write the Colab bundle, and say what else has to go on Drive.

        Offered here because annotating is the last step that wants eyes on
        the footage. Everything after it needs a CUDA card, and for anyone
        without one that means Colab -- so this is the moment they need the
        notebook, not a line in the documentation.
        """
        frames = self.state.frames_path if self.state else None
        default = Path(self.frames_dir.text() or ".").parent / "colab_bundle.zip"
        name, _ = QFileDialog.getSaveFileName(
            self, "Save the Colab bundle", str(default), filter="Zip (*.zip)"
        )
        if not name:
            return

        try:
            written = bundle_mod.build(Path(name))
        except OSError as exc:
            QMessageBox.warning(self, "Could not write the bundle", str(exc))
            return

        size_kb = written.stat().st_size / 1024
        upload = [f"{written}          (this bundle)"]
        if frames is not None and frames.is_dir():
            upload.append(f"{frames}          (your frames and annotations)")
        else:
            upload.append("the folder holding your annotated frames")

        QMessageBox.information(
            self,
            "Colab bundle written",
            f"{written.name} ({size_kb:.0f} KB)\n\n"
            "Put these in one folder on Google Drive:\n\n  "
            + "\n  ".join(upload)
            + "\n\nThe videos go on Drive too, anywhere you like - the "
            "notebook asks where. Then open "
            "colab_detectron2_pipeline.ipynb from inside the bundle.\n\n"
            "It picks up at step 4 and runs to the end: dataset, training, "
            "contour export, and the parameter files for tracking.\n\n"
            "Tip: 'Save parameters' writes a .toml the notebook will inherit, "
            "so your animal count, ROI and area thresholds carry over.",
        )

    # ----------------------------------------------------- GPU capability
    def check_gpu(self) -> None:
        if self.gpu_thread.isRunning():
            return
        self.gpu = gpu_mod.GpuReport()  # unchecked: shows "Checking..."
        self.gpu_status.setText(self.gpu.summary())
        self.recheck_button.setEnabled(False)
        self.gpu_thread.start()

    def _gpu_reported(self, report) -> None:
        self.gpu = report
        self.recheck_button.setEnabled(True)
        text = report.summary()
        if not report.usable:
            text += "\n\n" + gpu_mod.INSTALL_HINT
        self.gpu_status.setText(text)
        self._refresh_gpu_steps()

    def _refresh_gpu_steps(self) -> None:
        usable = self.gpu.usable
        running = self.train_process is not None
        self.train_button.setEnabled(
            usable and not running and bool(self.state and self.state.dataset_path)
        )
        self.stop_train_button.setEnabled(running)
        # offered only when there is something to install and nothing running
        self.install_button.setVisible(self.gpu.checked and not usable)
        self.install_button.setEnabled(not running)
        mark = "" if self.gpu.checked else " - checking"
        if self.gpu.checked and not usable:
            mark = " - needs Colab"
        self.steps.setItemText(4, f"5. Train{mark}")
        self._update_export_command()

    # ----------------------------------------------------------------- context
    def set_video_context(self, video_paths, output_dir=None) -> None:
        """Called when a video is loaded; finds and reloads any prior work."""
        previously_open = list(self._open_videos)
        self._open_videos = [Path(p) for p in video_paths]
        if not self._open_videos:
            return
        try:
            root = prep_state_mod.prep_root(self._open_videos, output_dir)
        except ValueError:
            return

        self.state = prep_state_mod.PrepState.load(root)
        self.state.refresh()

        # The annotation list belongs to the user, so opening another video
        # must not wipe it. It is re-seeded from the open videos only when it
        # was never curated -- empty, or still exactly what was seeded last
        # time -- and otherwise left alone.
        stored = self.state.video_paths
        if stored and stored != previously_open:
            self.set_video_paths(stored)
        else:
            self.set_video_paths(self._open_videos)

        index = self.group_by.findData(self.state.group_by)
        if index >= 0:
            self.group_by.blockSignals(True)
            self.group_by.setCurrentIndex(index)
            self.group_by.blockSignals(False)

        if self.state.enhancement:
            self.enhancement.setSettings(self.state.enhancement)
        if self.state.frames_path:
            self.frames_dir.setText(str(self.state.frames_path))
        else:
            self.frames_dir.setText(str(root / "annotate"))
        if self.state.dataset_path:
            self.dataset_dir.setText(str(self.state.dataset_path))
        else:
            self.dataset_dir.setText(str(root / "dataset"))

        self.refresh()

    def refresh(self) -> None:
        """Re-derives every step's status from disk and updates the headers."""
        if self.state is None:
            return
        self.state.refresh()
        s = self.state

        self.steps.setItemText(
            0, "1. Enhancement" + (" - done" if s.profile else "")
        )
        if s.sampling_done:
            # The folder is on screen already, in the Output folder field. A
            # full absolute path here wraps to three lines and pushes this
            # step's own button off the bottom of the page.
            self.sample_status.setText(f"{s.n_sampled} frames sampled.")
            self.sample_status.setToolTip(str(s.frames_path))
            self.steps.setItemText(1, f"2. Sample frames - done ({s.n_sampled})")
        else:
            self.sample_status.setText("No frames sampled yet.")
            self.steps.setItemText(1, "2. Sample frames")

        if s.n_sampled:
            self.annotate_status.setText(
                f"{s.annotated} of {s.n_sampled} frames annotated."
            )
            mark = " - done" if s.annotation_done else ""
            self.steps.setItemText(2, f"3. Annotate{mark} ({s.annotated}/{s.n_sampled})")
        else:
            self.annotate_status.setText("Sample some frames first.")
            self.steps.setItemText(2, "3. Annotate")

        if s.dataset_done:
            stale = " - stale, rebuild" if s.dataset_is_stale() else " - done"
            self.steps.setItemText(
                3, f"4. Build dataset{stale} ({s.n_train}/{s.n_val})"
            )
        else:
            self.steps.setItemText(3, "4. Build dataset")

        self._refresh_gpu_steps()
        self.launch_button.setEnabled(bool(s.frames_path and s.n_sampled))
        self.open_folder_button.setEnabled(bool(s.frames_path and s.frames_path.is_dir()))
        self.build_button.setEnabled(s.annotated > 0)

    # ----------------------------------------------------------------- step 1
    def enhancement_committed(self, settings: dict) -> None:
        """Record what the frames will be prepared with, and say so."""
        self.enhancement_status.setText(
            "Set with the Frame enhancement control above, which applies to "
            "thresholding too.\n\nFrames sampled for annotation will be "
            f"written with: {fp.describe(settings)}"
        )
        if self.state is None:
            return
        self.state.enhancement = settings
        self.state.save()

    def _profile_saved(self, path) -> None:
        if self.state is None:
            return
        self.state.set_profile(Path(path))
        self.state.save()
        self.refresh()

    # ------------------------------------------------------ the video list
    @property
    def video_paths(self) -> list[Path]:
        """The clips frames will be drawn from, in list order."""
        return [
            Path(self.video_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.video_list.count())
        ]

    def set_video_paths(self, paths) -> None:
        """Replaces the list, dropping duplicates but keeping the order given.

        One file under two spellings counts once, as it does in a sampling run.
        """
        with sampling_mod.locations_cached():  # each path is resolved once
            listed = [Path(path) for path in paths]
            unique = [listed[i] for i in sampling_mod._first_spellings(listed)]

            self.video_list.clear()
            for path in unique:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.video_list.addItem(item)
            self._refresh_names()
        self._relabel_videos()

        if self.state is not None:
            self.state.set_videos(unique)
            self.state.save()
        self._start_counting(unique)
        self._update_video_summary()
        self._update_grouping_status()
        self._schedule_preview()
        self._update_export_command()  # which clips share a file name has changed

    def clip_names(self) -> dict[Path, str]:
        """What each listed clip's frames are called: the cached ``_refresh_names``."""
        return self._names

    def _refresh_names(self) -> None:
        """Works the names out again, from the list and the output folder.

        The frames folder's manifest decides, because frames that exist keep
        their names. With no folder chosen yet nothing is recorded: an empty
        text would otherwise be read as the working directory, whose manifest
        belongs to something else.
        """
        videos = self.video_paths
        folder = self.frames_dir.text()
        recorded = sampling_mod.recorded_clips(Path(folder)) if folder else []
        self._names_problem = ""
        try:
            self._names = sampling_mod.assign_names(videos, recorded)
        except SamplingError as exc:
            # The folder records different clips under one name. A run says so
            # and stops; until then the list should still work, and the preview
            # says what the run will say.
            self._names_problem = str(exc)
            self._names = sampling_mod.assign_names(videos)

    def _relabel_videos(self) -> None:
        """Shows each clip under its name, and which ones cannot be found."""
        for row in range(self.video_list.count()):
            item = self.video_list.item(row)
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            name = self._names[path]
            label = path.name if name == path.stem else f"{name}  ({path.name})"
            if path.is_file():
                item.setText(label)
                item.setToolTip(str(path))
            else:
                # kept rather than dropped: an unplugged drive should not
                # quietly empty a list the user spent time building
                item.setText(f"{label}  (missing)")
                item.setToolTip(f"{path} - not found right now")

    def _output_folder_changed(self, _text: str) -> None:
        """The frames folder decides which names are already taken."""
        self._refresh_names()
        self._relabel_videos()
        self._update_video_summary()
        self._update_grouping_status()
        self._schedule_preview()

    def add_videos(self) -> None:
        start = str(self.video_paths[-1].parent) if self.video_paths else ""
        names, _ = QFileDialog.getOpenFileNames(
            self, "Videos to draw annotation frames from", start,
            "Videos (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv *.m4v);;All files (*)",
        )
        if names:
            self.set_video_paths(self.video_paths + [Path(n) for n in names])

    def add_video_folder(self) -> None:
        start = str(self.video_paths[-1].parent) if self.video_paths else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of videos to draw annotation frames from", start
        )
        if not folder:
            return
        recursive = self.include_subfolders.isChecked()
        # a deep tree on a network drive takes a moment to walk
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # drops the macOS "._" companions that otherwise arrive as
            # unreadable videos, and hidden folders such as .Trash
            found = sampling_mod.videos_in_folder(Path(folder), recursive)
        finally:
            QApplication.restoreOverrideCursor()
        if not found:
            hint = (
                " or its subfolders"
                if recursive
                else ". Tick “Include subfolders” to look inside them too"
            )
            QMessageBox.information(
                self, "No videos there", f"No video files were found in {folder}{hint}."
            )
            return
        self.set_video_paths(self.video_paths + found)

    def remove_selected_videos(self) -> None:
        rows = {index.row() for index in self.video_list.selectedIndexes()}
        if not rows:
            QMessageBox.information(
                self, "Nothing selected", "Select the videos to remove first."
            )
            return
        keep = [p for i, p in enumerate(self.video_paths) if i not in rows]
        self.set_video_paths(keep)

    def reset_videos_to_open(self) -> None:
        self.set_video_paths(self._open_videos)

    def _start_counting(self, videos) -> None:
        """Queues anything new or changed for counting.

        Videos are queued rather than counted directly, because adding a
        second folder while the first is still being counted would otherwise
        mark the new clips as seen and never count them, leaving the preview
        waiting for numbers that were never coming.
        """
        for video in videos:
            try:
                stat = video.stat()
                key = (stat.st_mtime, stat.st_size)
            except OSError:
                self._counts[video] = 0
                self._count_keys.pop(video, None)
                continue
            if self._count_keys.get(video) != key:
                self._count_keys[video] = key
                self._pending_counts.add(video)
        self._drain_count_queue()

    def _drain_count_queue(self) -> None:
        if not self._pending_counts or self.count_thread.isRunning():
            return
        batch = sorted(self._pending_counts)
        self._pending_counts.clear()
        self.count_thread.set_parameters(batch)
        self.count_thread.start()

    def _counts_ready(self, counts: dict) -> None:
        self._counts.update(counts)
        self._update_video_summary()
        self.update_preview()

    def _update_video_summary(self) -> None:
        videos = self.video_paths
        if not videos:
            self.video_summary.setText(
                "No videos yet. Add the clips you want the model trained on - "
                "every recording from this setup, not just the one being tracked."
            )
            return

        names = self.clip_names()
        groups = dataset_mod.group_videos(
            [names[v] for v in videos],
            self.state.group_overrides if self.state else None,
            self.state.group_by if self.state else "recording",
        )
        known = [self._counts.get(v) for v in videos]
        counted = [c for c in known if c]
        def plural(n: int, noun: str) -> str:
            return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

        parts = [plural(len(videos), "video"), plural(len(groups), "recording")]
        if len(counted) == len(videos):
            parts.append(f"{sum(counted):,} frames")
        else:
            parts.append(f"counting frames ({len(counted)}/{len(videos)})")

        missing = [v for v in videos if not v.is_file()]
        if missing:
            parts.append(f"{len(missing)} not found")
        text = ", ".join(parts) + "."

        unreadable = [v for v in videos if v.is_file() and self._counts.get(v) == 0]
        if unreadable:
            text += (
                f" {len(unreadable)} could not be read and will be skipped: "
                + ", ".join(v.name for v in unreadable[:3])
                + ("..." if len(unreadable) > 3 else "")
            )
        self.video_summary.setText(text)

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def update_preview(self) -> None:
        """Says what would be sampled, before anything is written."""
        videos = self.video_paths
        if not videos:
            self.sample_preview.setText("")
            return
        if self._names_problem:
            self.sample_preview.setText(f"Cannot plan this yet: {self._names_problem}")
            return

        counts = [self._counts.get(v) for v in videos]
        if any(c is None for c in counts):
            self.sample_preview.setText("Counting frames...")
            return

        try:
            plans = sampling_mod.plan_sampling(
                videos,
                self.n_frames.value(),
                self.seed.value(),
                counts=counts,
                names=self.clip_names(),
            )
        except Exception as exc:  # noqa: BLE001 - a preview must never raise
            self.sample_preview.setText(f"Cannot plan this yet: {exc}")
            return

        taking = [p for p in plans if p.requested]
        empty = [p for p in plans if p.readable and not p.requested]
        shares = sorted(p.requested for p in taking)
        text = (
            f"{sum(p.requested for p in plans)} frames from {len(taking)} of "
            f"{len(plans)} videos"
        )
        if shares:
            text += (
                f", {shares[0]} to {shares[-1]} each"
                if shares[0] != shares[-1]
                else f", {shares[0]} each"
            )
        text += "."
        if empty:
            # the whole point of a collection is that every clip is seen, so
            # this is worth saying loudly rather than leaving to be discovered
            text += (
                f" {len(empty)} videos would get no frames at all - raise the"
                " count to at least one per video."
            )
        renamed = sum(1 for p in plans if p.name != p.video.stem)
        if renamed:
            text += (
                f" {renamed} {'clip' if renamed == 1 else 'clips'} will be named"
                " by folder, because file names are shared."
            )
        self.sample_preview.setText(text)

    # ----------------------------------------------------------------- step 2
    def _choose_frames_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the sampled frames", self.frames_dir.text()
        )
        if name:
            self.frames_dir.setText(name)

    def run_sampling(self) -> None:
        videos = self.video_paths
        if not videos:
            QMessageBox.warning(
                self, "No videos",
                "Add the videos to draw annotation frames from first.",
            )
            return
        output = Path(self.frames_dir.text())

        gone = [v for v in videos if not v.is_file()]
        if gone:
            QMessageBox.warning(
                self, "Videos not found",
                f"{len(gone)} of the listed videos cannot be found right now, "
                "so no frames can be taken from them:\n\n"
                + "\n".join(str(v) for v in gone[:8])
                + ("\n..." if len(gone) > 8 else "")
                + "\n\nReconnect the drive, or remove them from the list.",
            )
            return

        existing = list(output.glob("*.png")) + list(output.glob("*.jpg")) if output.is_dir() else []
        if existing:
            answer = QMessageBox.question(
                self,
                "Folder is not empty",
                f"{output} already holds {len(existing)} image(s).\n\n"
                "Sampling again adds to them, and any annotations you have "
                "already made are kept. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        counts = [self._counts.get(v) for v in videos]
        self.sampling_thread.set_parameters(
            videos,
            output,
            self.n_frames.value(),
            self.enhancement.settings(),
            seed=self.seed.value(),
            counts=None if any(c is None for c in counts) else counts,
        )
        self.sample_button.setEnabled(False)
        self.sampling_thread.start()

    def _sampling_finished(self) -> None:
        self.sample_button.setEnabled(True)
        try:
            # The run recorded its names in the manifest, which now decides them.
            # The list can be edited while it works, so everything that shows a
            # name is brought up to date, not only the cache.
            self._output_folder_changed(self.frames_dir.text())
        except Exception:  # noqa: BLE001 - the run's result matters more than this
            logging.exception("Could not refresh the clip names after sampling")
        result = self.sampling_thread.result
        if result is None:
            self.refresh()
            return
        if self.state is not None:
            self.state.set_frames_dir(result.output)
            self.state.set_profile(result.profile_path)
            self.state.n_sampled = result.written
            self.state.sampling_complete = result.complete
            self.state.enhancement = self.enhancement.settings()
            self.state.save()
        if result.unreadable:
            # computed by the sampler and, until now, thrown away
            QMessageBox.warning(
                self,
                "Some videos were skipped",
                f"{len(result.unreadable)} video(s) could not be read and "
                "contributed no frames:\n\n"
                + "\n".join(result.unreadable[:10])
                + ("\n..." if len(result.unreadable) > 10 else ""),
            )
        if not result.complete:
            QMessageBox.information(
                self,
                "Sampling cancelled",
                f"{result.written} frames were written and kept. Run the step "
                "again to add more.",
            )
        self.refresh()
        self.steps.setCurrentIndex(2)

    # ----------------------------------------------------------------- step 3
    def launch_labelme(self) -> None:
        folder = self.state.frames_path if self.state else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return

        if importlib.util.find_spec("labelme") is None:
            QMessageBox.warning(
                self,
                "LabelMe is not installed",
                "Annotating needs LabelMe, which is an optional extra. Install "
                "it into this environment with:\n\n"
                f'  "{sys.executable}" -m pip install "idtrackerai-CYSM[annotate]"',
            )
            return

        arguments = ["-m", "labelme", str(folder)]
        label = self.class_name.text().strip()
        if label:
            # Pre-loading the label and validating it exactly turns the
            # label-typo problem the converter can only report into one that
            # cannot happen.
            arguments += ["--labels", label, "--validate-label", "exact"]

        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        # LabelMe uses PySide6; this application pins QT_API for its own qtpy.
        # Do not let ours leak into it.
        environment.remove("QT_API")
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(arguments)

        started, _pid = process.startDetached()
        if not started:
            QMessageBox.critical(
                self,
                "Could not start LabelMe",
                "Tried to run:\n\n"
                f"  {sys.executable} {' '.join(arguments)}\n\n"
                "LabelMe is installed but would not start, so this usually means "
                "the environment is broken. Check that "
                "'python -m labelme --help' works.",
            )
            return
        logging.info("Launched LabelMe on %s", folder)

    def open_frames_folder(self) -> None:
        folder = self.state.frames_path if self.state else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ----------------------------------------------------------------- step 4
    def _choose_dataset_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the dataset", self.dataset_dir.text()
        )
        if name:
            self.dataset_dir.setText(name)

    def run_dataset(self) -> None:
        frames = self.state.frames_path if self.state else None
        if frames is None or not frames.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return

        expected = self.expected_instances.value()
        mode = self.group_by.currentData()
        request = dataset_mod.DatasetRequest(
            input_dir=frames,
            output_dir=Path(self.dataset_dir.text()),
            val_fraction=self.val_fraction.value(),
            single_class=self.class_name.text().strip() or None,
            expected_instances=expected or None,
            seed=self.seed.value(),
            split_by="random" if mode == "random" else "video",
            group_by="file" if mode == "file" else "recording",
            group_overrides=dict(self.state.group_overrides) if self.state else None,
        )
        self.dataset_thread.set_parameters(request)
        self.build_button.setEnabled(False)
        self.dataset_thread.start()

    def _dataset_finished(self) -> None:
        self.build_button.setEnabled(True)
        report = self.dataset_thread.result
        if report is None:
            self.refresh()
            return
        if self.state is not None:
            self.state.set_dataset_dir(report.output_dir)
            self.state.n_train = report.train_images
            self.state.n_val = report.val_images
            self.state.save()
        self.report_view.setPlainText(self._format_report(report))
        self.refresh()

    @staticmethod
    def _format_report(report) -> str:
        lines = [
            f"Categories: {[c['name'] for c in report.categories]}",
            f"Train: {report.train_images} images, {report.train_annotations} instances",
            f"Val:   {report.val_images} images, {report.val_annotations} instances",
        ]
        if report.train_videos or report.val_videos:
            lines.append(f"  train groups: {report.train_videos}")
            lines.append(f"  val groups:   {report.val_videos}")
        if report.enhancement:
            lines.append(f"Enhancement carried forward: {fp.describe(report.enhancement)}")
        else:
            lines.append(
                "WARNING: no enhancement record; training will not know how "
                "these frames were prepared."
            )
        if report.problems:
            lines.append(f"\n{len(report.problems)} problem(s):")
            lines += [f"  {p}" for p in report.problems[:20]]
        if report.notes:
            lines.append(f"\n{len(report.notes)} note(s):")
            lines += [f"  {n}" for n in report.notes[:20]]
        return "\n".join(lines)

    def install_gpu_stack(self) -> None:
        """Install PyTorch and Detectron2 for this machine, after showing how.

        Neither can be an ordinary dependency: PyPI's torch wheel for Windows
        is CPU-only, so declaring it would quietly stop idtracker.ai's own
        identity tracking using the GPU, and Detectron2 is not on PyPI at all
        and builds from source. So it is offered here, where the person has
        just been told something is missing, rather than left to the docs.

        The plan is shown first. This downloads gigabytes and compiles C++;
        nobody should discover that from a progress bar.
        """
        if self.train_process is not None:
            QMessageBox.information(
                self, "Busy", "Wait for the current run to finish first."
            )
            return

        # check_index=False keeps this instant: picking the CUDA index makes a
        # network request, and the run itself does that properly.
        plan = install_mod.make_plan(check_index=False)
        answer = QMessageBox.question(
            self,
            "Install PyTorch and Detectron2?",
            plan.as_text()
            + "\n\nThis downloads several gigabytes and compiles C++; it can "
            "take a long while. The exact CUDA version is confirmed when it "
            "runs.\n\nGo ahead?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        arguments = [
            "-m", "idtrackerai.extra_tools.detectron2_pipeline.install_gpu",
            "--yes",
        ]
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._training_output)
        process.finished.connect(self._install_finished)
        process.errorOccurred.connect(self._training_error)

        self.steps.setCurrentIndex(4)
        self.train_log.setPlainText("Installing. This will take a while.\n")
        self.train_process = process
        self._refresh_gpu_steps()
        process.start()

    def _install_finished(self, code: int, _status) -> None:
        self.train_process = None
        if code == 0:
            self.train_log.appendPlainText(
                "\nInstall finished. Checking what it can do..."
            )
        else:
            self.train_log.appendPlainText(
                f"\nThe install did not finish (exit code {code}). The output "
                "above says where it stopped; a missing C++ compiler is the "
                "usual reason."
            )
        # either way, report what the machine can actually do now
        self.check_gpu()

    # ------------------------------------------------------------- step 5 run
    def _choose_model_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the trained model", self.model_dir.text()
        )
        if name:
            self.model_dir.setText(name)
            self._update_export_command()

    def _choose_weights(self) -> None:
        start = self.weights_path.text() or self.model_dir.text()
        name, _ = QFileDialog.getOpenFileName(
            self, "Trained Detectron2 weights", start, "PyTorch weights (*.pth *.pt)"
        )
        if name:
            self.weights_path.setText(name)
            self._update_export_command()

    def weights(self) -> Path | None:
        """The weights to export with: the file chosen in step 6, else the
        model step 5 trains into its folder, else None."""
        if self.weights_path.text().strip():
            return Path(self.weights_path.text().strip())
        if self.model_dir.text():
            return Path(self.model_dir.text()) / "model_final.pth"
        return None

    def set_number_of_animals(self, n: int) -> None:
        """The export's detection cap. Without it the exporter keeps at most
        5 per frame, and an arena holding more loses animals silently."""
        self.number_of_animals = int(n or 0)
        self._update_export_command()

    def has_chosen_weights(self) -> bool:
        """True when weights were picked in step 6 or came from the .toml,
        as opposed to merely implied by step 5's model folder."""
        return bool(self.weights_path.text().strip())

    def model_spec(self) -> dict | None:
        """The session's [segmentation_model] table, or None without weights."""
        weights = self.weights()
        if weights is None:
            return None
        spec = dict(self._loaded_spec)
        spec.update(
            backend="detectron2",
            weights=str(weights),
            score_threshold=round(self.score_threshold.value(), 4),
        )
        if self.contours_dir.text():
            spec["contours_dir"] = self.contours_dir.text()
        return spec

    def set_model_spec(self, spec: dict | None) -> None:
        """Restores step 6 from a session's [segmentation_model] table."""
        try:
            spec = sm.normalize_model(spec)
        except sm.SegmentationModelError as exc:
            logging.warning("Ignoring [segmentation_model]: %s", exc)
            spec = None
        if spec is None or spec["backend"] != "detectron2":
            self._loaded_spec = {}
            return
        # keep only what was given, so defaults are not written back as if chosen
        self._loaded_spec = {
            key: value
            for key, value in spec.items()
            if value is not None and value != sm.DEFAULTS.get(key)
        }
        self.weights_path.setText(spec["weights"])
        self.score_threshold.setValue(spec["score_threshold"])
        if spec.get("contours_dir"):
            self.contours_dir.setText(spec["contours_dir"])
        self.steps.setCurrentIndex(5)
        self._update_export_command()

    def _choose_contours_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the exported contours", self.contours_dir.text()
        )
        if name:
            self.contours_dir.setText(name)
            self._update_export_command()

    @staticmethod
    def _script(name: str) -> Path:
        """A pipeline script's path, run through this interpreter.

        Called by path rather than by console-script name, because the scripts
        are installed but their directory is often not on PATH on Windows, and
        never is inside a venv that was not activated.
        """
        return Path(gpu_mod.__file__).with_name(name)

    def run_training(self) -> None:
        dataset = self.state.dataset_path if self.state else None
        if dataset is None or not (dataset / "train.json").is_file():
            QMessageBox.warning(
                self, "No dataset", "Build the dataset in step 4 first."
            )
            return
        if not self.model_dir.text():
            QMessageBox.warning(
                self, "No model folder", "Choose where the weights should go."
            )
            return

        arguments = [
            str(self._script("training.py")),
            "--dataset", str(dataset),
            "--output", self.model_dir.text(),
            "--epochs", str(self.epochs.value()),
        ]
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._training_output)
        process.finished.connect(self._training_finished)
        process.errorOccurred.connect(self._training_error)

        self.train_log.setPlainText(
            "$ " + sys.executable + " " + " ".join(arguments) + "\n"
        )
        self.train_process = process
        self._refresh_gpu_steps()
        process.start()

    def _training_output(self) -> None:
        if self.train_process is None:
            return
        chunk = bytes(self.train_process.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        self.train_log.appendPlainText(chunk.rstrip())

    def _training_error(self, error) -> None:
        self.train_log.appendPlainText(f"\nCould not run training: {error}")
        if error == QProcess.ProcessError.FailedToStart:
            # a process that never started never emits finished, so without
            # this the Train and Install buttons would stay disabled for good
            self.train_process = None
            self._refresh_gpu_steps()

    def _training_finished(self, code: int, _status) -> None:
        self.train_process = None
        self._refresh_gpu_steps()
        if code == 0:
            self.train_log.appendPlainText("\nTraining finished.")
            self.steps.setItemText(4, "5. Train - done")
            self.steps.setCurrentIndex(5)
        else:
            self.train_log.appendPlainText(f"\nTraining stopped (exit code {code}).")

    def stop_training(self) -> None:
        if self.train_process is None:
            return
        self.train_log.appendPlainText("\nStopping...")
        self.train_process.kill()

    # ------------------------------------------------------------- step 6 run
    VIDEO_SUFFIXES = sampling_mod.VIDEO_SUFFIXES

    def _export_target(self, videos) -> tuple[str, str]:
        """The ``--videos`` argument, and a phrase describing what it covers.

        A glob is used only when it means exactly this list: one folder, and
        every video in it. Anything else is listed clip by clip, because a
        command that quietly exports the wrong set is worse than a long one.
        """
        if not videos:
            return '"<folder>/*.mp4"', "no videos listed yet"

        parents = {video.parent for video in videos}
        if len(parents) == 1:
            folder = parents.pop()
            suffix = videos[0].suffix.lower()
            same_suffix = all(v.suffix.lower() == suffix for v in videos)
            try:
                on_disk = {
                    p for p in folder.iterdir()
                    if p.suffix.lower() in self.VIDEO_SUFFIXES
                    and not sampling_mod._is_sidecar(p)
                }
            except OSError:
                on_disk = set()
            if same_suffix and on_disk and on_disk == set(videos):
                return (
                    f'"{folder / ("*" + suffix)}"',
                    f"every {suffix} file in {folder.name}",
                )

        listed = " ".join(f'"{video}"' for video in videos)
        where = (
            "across several folders" if len(parents) > 1
            else "a subset of that folder"
        )
        return listed, f"the {len(videos)} clips listed above, {where}"

    def _update_export_command(self) -> None:
        videos = self.video_paths
        weights = self.weights() or Path("<model folder>") / "model_final.pth"
        contours = self.contours_dir.text() or "<contours folder>"

        # A glob of the first clip's folder is only right when the list IS
        # that folder. On a curated list it exports clips that were removed,
        # and on a list spanning folders it silently leaves whole folders out
        # -- which would surface a day and a half into an export.
        target, scope = self._export_target(videos)
        command = (
            f'"{sys.executable}" "{self._script("inference.py")}"'
            f' --videos {target}'
            f' --weights "{weights}"'
            f' --output-dir "{contours}"'
        )
        if abs(self.score_threshold.value() - sm.DEFAULTS["score_threshold"]) > 1e-9:
            command += f" --score-threshold {self.score_threshold.value():g}"
        if self.number_of_animals > 0:
            command += f" --max-instances {self.number_of_animals}"
        self.export_command.setPlainText(command)
        self._export_scope = scope

        where = (
            "This machine can run it."
            if self.gpu.usable
            else "This machine cannot run it; the Colab notebook can."
        )
        # the exporter names a contour file for the clip's file name and stops
        # when two clips would write one; say so now, not after the copy
        shared = sampling_mod.repeated_stems(videos)
        clashing = sum(len(paths) for paths in shared.values())
        warning = (
            f"{clashing} clips share a file name, so the exporter will refuse "
            "this list; export them in separate runs. "
            if clashing
            else ""
        )
        self.export_status.setText(
            warning
            + f"This exports {scope}. Running the model over every frame takes "
            "roughly an hour per 30 000-frame clip, so a collection takes "
            "days. It is not run from here: copy the command and run it in a "
            "terminal you can leave open. It writes one file per clip and "
            "skips clips it has already done, so it can be stopped and "
            f"restarted freely. {where}"
        )

    def copy_export_command(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.export_command.toPlainText())
            self.copy_command_button.setText("Copied")
            QTimer.singleShot(
                1500, lambda: self.copy_command_button.setText("Copy command")
            )

    # ---------------------------------------------------------------- threads
    def _wire_threads(self) -> None:
        for thread, finished in (
            (self.sampling_thread, self._sampling_finished),
            (self.dataset_thread, self._dataset_finished),
        ):
            thread.started.connect(lambda t=thread: self._thread_started(t))
            thread.finished.connect(self._thread_finished)
            thread.finished.connect(finished)
            thread.set_progress_value.connect(self._set_progress_value)
            thread.set_progress_max.connect(self._set_progress_max)
            thread.failed.connect(self._thread_failed)

    def _thread_started(self, thread) -> None:
        # Created here rather than before start(), as BkgWidget does, so a run
        # that finishes instantly never flashes a dialog.
        self.progress = QProgressDialog("Working...", "Cancel", 0, 100, self)
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

    def _thread_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Step failed", message)

    # --------------------------------------------------------------- tooltips
    def setToolTips(self, tips: dict) -> None:
        """Explain every control, since none of this is self-evident."""
        pairs = [
            (self.video_list, "d2_video_list"),
            (self.video_list_label, "d2_video_list"),
            (self.add_videos_button, "d2_add_videos"),
            (self.add_folder_button, "d2_add_folder"),
            (self.include_subfolders, "d2_include_subfolders"),
            (self.remove_videos_button, "d2_remove_videos"),
            (self.reset_videos_button, "d2_reset_videos"),
            (self.video_summary, "d2_video_list"),
            (self.n_frames, "d2_n_frames"),
            (self.seed, "d2_seed"),
            (self.seed_label, "d2_seed"),
            (self.sample_preview, "d2_preview"),
            (self.frames_dir, "d2_frames_folder"),
            (self.sample_button, "d2_n_frames"),
            (self.class_name, "d2_class_name"),
            (self.launch_button, "d2_labelme"),
            (self.open_folder_button, "d2_open_folder"),
            (self.refresh_button, "d2_refresh"),
            (self.bundle_button, "d2_bundle"),
            (self.expected_instances, "d2_expected"),
            (self.val_fraction, "d2_val_fraction"),
            (self.group_by, "d2_group_by"),
            (self.edit_grouping_button, "d2_edit_grouping"),
            (self.grouping_status, "d2_group_by"),
            (self.dataset_dir, "d2_dataset_folder"),
            (self.build_button, "d2_build"),
            (self.gpu_status, "d2_gpu"),
            (self.epochs, "d2_epochs"),
            (self.model_dir, "d2_model_folder"),
            (self.train_button, "d2_train"),
            (self.stop_train_button, "d2_train"),
            (self.recheck_button, "d2_gpu"),
            (self.install_button, "d2_install"),
            (self.train_log, "d2_train"),
            (self.contours_dir, "d2_contours_folder"),
            (self.weights_path, "d2_weights"),
            (self.weights_hint, "d2_weights"),
            (self.score_threshold, "d2_score_threshold"),
            (self.export_command, "d2_export"),
            (self.copy_command_button, "d2_export"),
            (self.export_status, "d2_export"),
            (self.enhancement_status, "enhancement"),
        ]
        for widget, key in pairs:
            if key in tips:
                widget.setToolTip(tips[key])


    # ----------------------------------------------------------------- closing
    def close(self) -> bool:
        """Stops any running work and flushes the state before the app exits."""
        if self.train_process is not None:
            self.train_process.kill()
            self.train_process.waitForFinished(3000)
            self.train_process = None
        for thread in (self.sampling_thread, self.dataset_thread,
                       self.count_thread, self.gpu_thread):
            if thread.isRunning():
                thread.quit()
                if not thread.wait(5000):
                    # quit() only asks, and the GPU check in particular does
                    # not listen. A QThread destroyed while it still runs
                    # aborts the whole process.
                    thread.terminate()
                    thread.wait(2000)
        if self.state is not None:
            self.state.save()
        return super().close()
