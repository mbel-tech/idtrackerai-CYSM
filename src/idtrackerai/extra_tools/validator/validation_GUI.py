import logging
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import sleep

import numpy as np
import toml
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtGui import QAction, QCloseEvent, QColor, QIcon, QKeyEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Blob, Fragment, ListOfBlobs, ListOfFragments, Session
from idtrackerai.base.postprocess import produce_output_dict
from idtrackerai.GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    GUIBase,
    IdLabels,
    LabelRangeSlider,
    LightPopUp,
    QHLine,
    TransparentDisabledOverlay,
    VideoPlayer,
    build_ROI_patches_from_list,
    get_icon,
    open_session,
)
from idtrackerai.utils import save_trajectories, track

from .widgets import (
    AdditionalInfo,
    ErrorsExplorer,
    IdGroups,
    Interpolator,
    LengthCalibrator,
    MarkMetadata,
    SetupPoints,
    VideoInfo,
    find_selected_blob,
    paintBlobs,
    paintTrails,
)

SELECT_POINT_DIST = 300

AUTOSAVE_INTERVAL_MS = 5 * 60 * 1000
"""How often the Validator saves by itself.

Validation is hours of unrepeatable manual work and nothing else writes it to
disk, so leaving it entirely to the user to remember Ctrl+S is a poor trade.
"""

MERGE_PROPAGATION_FRAMES = 100
"""How far a propagated merge reaches.

A merge is geometric, not an identity edit, so there is no fragment to bound
it: it would otherwise run to the end of the video, merging blobs long after
the two parts stopped belonging to the same animal.
"""

UNDO_DEPTH = 20
"""Edits kept on the undo stack. Snapshots live in RAM, hence a limit."""


class WarningRedirector(logging.Handler):
    def __init__(self, parent: GUIBase) -> None:
        super().__init__()
        self.parent = parent
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        QMessageBox.warning(
            self.parent, "idtracker.ai validator warning", record.getMessage()
        )


class IdentitiesSnapshot:
    """The user-generated identities and centroids of a set of blobs.

    This is everything an identity edit can change, so restoring it undoes one.
    The blobs themselves are referenced, not copied; only the two small lists
    each one carries are duplicated.
    """

    def __init__(self, blobs: Iterable[Blob]) -> None:
        self.entries = [
            (
                blob,
                (
                    None
                    if blob.user_generated_identities is None
                    else list(blob.user_generated_identities)
                ),
                (
                    None
                    if blob.user_generated_centroids is None
                    else list(blob.user_generated_centroids)
                ),
            )
            for blob in blobs
        ]

    def restore(self) -> None:
        for blob, identities, centroids in self.entries:
            blob.user_generated_identities = (
                None if identities is None else list(identities)
            )
            blob.user_generated_centroids = (
                None if centroids is None else list(centroids)
            )


class FramesSnapshot:
    """Which blobs were in which frames, plus what those blobs held.

    Merging and extending add and remove whole blobs, so restoring identities
    is not enough: the frame's blob list has to come back as well. Taking both
    keeps a single snapshot type sufficient for any structural edit.
    """

    def __init__(self, blobs: ListOfBlobs, frames: Iterable[int]) -> None:
        self.blobs = blobs
        self.frames = {frame: list(blobs.blobs_in_video[frame]) for frame in frames}
        self.identities = IdentitiesSnapshot(
            blob for blobs_in_frame in self.frames.values() for blob in blobs_in_frame
        )

    def restore(self) -> None:
        for frame, blobs_in_frame in self.frames.items():
            self.blobs.blobs_in_video[frame] = list(blobs_in_frame)
        self.identities.restore()


class UndoManager:
    """A bounded stack of undoable Validator edits.

    In-memory and per-run: closing the Validator discards it. It exists because
    a mis-aimed propagated edit used to mean reloading the session and losing
    everything done since the last save.
    """

    def __init__(self, gui: "ValidationGUI") -> None:
        self.gui = gui
        self.stack: list[tuple[str, IdentitiesSnapshot | FramesSnapshot]] = []

    def push_identities(self, description: str, blobs: Iterable[Blob]) -> None:
        """Records an edit that only changes identities and centroids."""
        self._push(description, IdentitiesSnapshot(blobs))

    def push_frames(self, description: str, frames: Iterable[int]) -> None:
        """Records an edit that adds or removes blobs in `frames`."""
        self._push(description, FramesSnapshot(self.gui.blobs, frames))

    def _push(
        self, description: str, snapshot: IdentitiesSnapshot | FramesSnapshot
    ) -> None:
        self.stack.append((description, snapshot))
        del self.stack[:-UNDO_DEPTH]

    def clear(self) -> None:
        """Forgets every edit; snapshots of a previous session are meaningless."""
        self.stack.clear()

    def undo(self) -> None:
        if not self.stack:
            self.gui.light_popup.info("Undo", "Nothing to undo")
            return

        description, snapshot = self.stack.pop()
        snapshot.restore()
        # An edit can have been propagated anywhere along a fragment, and the
        # snapshot does not record how far, so the whole video is recomputed.
        self.gui.update_trajectories_range(0, self.gui.n_frames)
        self.gui.light_popup.info("Undo", f"Undid: {description}")


class DblClickDialog(QDialog):
    class Answers(Enum):
        Cancel = 0
        ChangeId = 1
        Interpolate = 2
        Reset = 3
        Remove = 4
        Swap = 5
        Extend = 6
        Merge = 7

    def __init__(self, parent: QWidget, n_animals: int) -> None:
        super().__init__(parent)
        self.spinbox = QSpinBox()
        self.spinbox.setMaximum(n_animals)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.spinbox.setSpecialValueText("Null identity")

        self.propagate = QCheckBox("Propagate action")
        self.propagate.setChecked(True)
        spin_row = QHBoxLayout()
        spin_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spin_row.addWidget(QLabel("New identity:"))
        spin_row.addWidget(self.spinbox)

        cancel_btn = QPushButton(get_icon("cancel"), "Cancel")
        change_id_btn = QPushButton(get_icon("ok"), "Change id")
        remove_btn = QPushButton(QIcon.fromTheme("user-trash"), "Remove\ncentroid")
        reset_id_btn = QPushButton(get_icon("refresh"), "Reset id")
        self.interp_btn = QPushButton("Start interpolation")
        self.interp_btn.setIcon(get_icon("run"))
        first_btn_row = QHBoxLayout()
        first_btn_row.addWidget(remove_btn)
        first_btn_row.addWidget(reset_id_btn)
        first_btn_row.addWidget(change_id_btn)
        second_btn_row = QHBoxLayout()
        second_btn_row.addWidget(self.interp_btn)
        second_btn_row.addWidget(cancel_btn)
        change_id_btn.setDefault(True)

        swap_btn = QPushButton(get_icon("refresh"), "Swap with...")
        swap_btn.setToolTip(
            "Exchange this identity with another one, instead of renaming it"
        )
        extend_btn = QPushButton(get_icon("add"), "Extend...")
        extend_btn.setToolTip(
            "Repeat this centroid over the next frames, for an animal that"
            " stopped being detected while barely moving"
        )
        merge_btn = QPushButton(get_icon("add"), "Merge with...")
        merge_btn.setToolTip(
            "Rebuild one blob from this one and another, for an animal that"
            " segmentation split in two"
        )

        first_btn_row.addWidget(swap_btn)
        second_btn_row.addWidget(extend_btn)
        second_btn_row.addWidget(merge_btn)

        swap_btn.clicked.connect(lambda: self.done(self.Answers.Swap.value))
        extend_btn.clicked.connect(lambda: self.done(self.Answers.Extend.value))
        merge_btn.clicked.connect(lambda: self.done(self.Answers.Merge.value))

        cancel_btn.clicked.connect(lambda: self.done(self.Answers.Cancel.value))
        change_id_btn.clicked.connect(lambda: self.done(self.Answers.ChangeId.value))
        reset_id_btn.clicked.connect(lambda: self.done(self.Answers.Reset.value))
        remove_btn.clicked.connect(lambda: self.done(self.Answers.Remove.value))
        self.interp_btn.clicked.connect(
            lambda: self.done(self.Answers.Interpolate.value)
        )

        main_layout.addLayout(spin_row)
        main_layout.addWidget(self.propagate, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(first_btn_row)
        main_layout.addWidget(QHLine())
        main_layout.addLayout(second_btn_row)

    def exec_with_description(
        self, default: int | None
    ) -> tuple[Answers, int | None, bool]:
        if default is not None:
            self.spinbox.setValue(default)
        self.interp_btn.setEnabled(bool(default is not None and default > 0))

        self.spinbox.setFocus()
        answer = self.Answers(super().exec())

        return answer, self.spinbox.value(), self.propagate.isChecked()


class LoadSessionObjects(QThread):
    """Independent thread to load lists of Blobs/Fragments
    because they take too long for large sessions."""

    blobs: ListOfBlobs | None = None
    fragments: list[Fragment] | None = None
    loaded_from: Path
    "Original file of loaded ListOfBlobs. Validated list will be saved in the same location"

    def __init__(self, session: Session, parent: QWidget) -> None:
        super().__init__(parent)
        self.session = session

    def run(self) -> None:
        # when loading light session from CLI, the main windows remains out of focus.
        # This sleeps fixes it, not beautiful but it works...
        sleep(0.1)
        for path in (
            self.session.preprocessing_folder / "list_of_blobs_validated.pickle",
            self.session.preprocessing_folder / "list_of_blobs_no_gaps.pickle",
            self.session.blobs_path,
        ):
            try:
                self.blobs = ListOfBlobs.load(path)
                self.loaded_from = path
                break
            except FileNotFoundError:
                pass
        else:
            logging.warning(
                "List of blobs not found in %s", self.session.blobs_path.parent
            )
            self.blobs = None
        try:
            self.fragments = ListOfFragments.load(
                self.session.fragments_path, reconnect=False
            ).fragments
            for index, fragment in enumerate(self.fragments):
                if fragment.identifier != index:
                    logging.warning(
                        "Loading an old session, invalid list of fragments format"
                    )
                    raise FileExistsError
        except FileNotFoundError:
            self.fragments = None


class SaveSessionObjects(QThread):
    """Independent thread to save lists of Blobs/Fragments
    because they take too long for large sessions."""

    loaded_from: Path
    "Original file of loaded ListOfBlobs. Validated list will be saved in the same location"

    def __init__(
        self,
        session: Session,
        blobs: ListOfBlobs,
        parent: QWidget,
        loaded_from: Path,
        verbose: bool = True,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.loaded_from = loaded_from
        self.blobs = blobs
        self.verbose = verbose
        self.success = True
        self.error_message: str | None = None
        self.backup_path: Path | None = None
        "Where the session went instead, if the normal location was unwritable."

    def run(self) -> None:
        # when loading light session from CLI, the main windows remains out of focus.
        # This sleeps fixes it, not beautiful but it works...
        sleep(0.1)
        try:
            self.session.save()
            self.blobs.save(self.loaded_from, verbose=self.verbose)
        except Exception as exc:  # noqa: BLE001 - reported, then salvaged below
            self.success = False
            self.error_message = str(exc)
            logging.error("Failed to save the session: %s", exc)
            self.save_backup()

    def save_backup(self) -> None:
        """Writes the session somewhere local after the real save failed.

        Session folders live on network drives and synced folders that go away
        without warning. Hours of validation are held only in memory, so a
        failed save has to land somewhere rather than raise and lose them.
        """
        backup_dir = Path.home() / "idtrackerai_backups" / self.session.name
        original_folder = self.session.session_folder
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            # Session.save() takes no path; redirecting it means moving the
            # folder it derives the path from, and putting it back afterwards.
            self.session.session_folder = backup_dir
            self.session.save()
            self.blobs.save(backup_dir / "list_of_blobs.pickle", verbose=self.verbose)
        except Exception as exc:  # noqa: BLE001 - nothing left to fall back to
            logging.error("The backup save also failed: %s", exc)
        else:
            self.backup_path = backup_dir
            logging.info("Session backed up to %s", backup_dir)
        finally:
            self.session.session_folder = original_folder


class ValidationGUI(GUIBase):
    blobs: ListOfBlobs

    _unsaved_changes = False
    _edit_generation = 0
    "Bumped by every edit, so a save can tell whether anything changed meanwhile."
    _save_generation = 0
    _save_silent = False

    @property
    def unsaved_changes(self) -> bool:
        return self._unsaved_changes

    @unsaved_changes.setter
    def unsaved_changes(self, value: bool) -> None:
        self._unsaved_changes = value
        if value:
            self._edit_generation += 1

    def __init__(self) -> None:
        super().__init__()

        # TODO logging.getLogger().addHandler(WarningRedirector(self))
        self.light_popup = LightPopUp()
        self.undo_manager = UndoManager(self)
        self.saving_session_thread: SaveSessionObjects | None = None
        self.save_thread: SaveTrajectoriesThread | None = None
        self.setWindowTitle("Validator")
        self.documentation_url = "https://idtracker.ai/latest/user_guide/validator.html"

        self.video_player = VideoPlayer(self)
        self.widgets_to_close.append(self.video_player)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.double_click_event.connect(self.double_click_on_canvas)

        def new_changes() -> None:
            self.unsaved_changes = True

        self.id_groups = IdGroups(self)
        self.id_groups.needToDraw.connect(self.video_player.update)
        self.id_groups.unsaved_changes.connect(new_changes)

        self.errorsExplorer = ErrorsExplorer()
        self.errorsExplorer.go_to_error.connect(self.go_to_error)

        self.mark_metadata = MarkMetadata(self)
        self.mark_metadata.needToDraw.connect(self.video_player.update)

        self.interpolator = Interpolator()
        self.interpolator.undo_manager = self.undo_manager
        self.interpolator.need_to_draw.connect(self.video_player.update)
        self.interpolator.update_trajectories.connect(self.update_trajectories_range)
        self.interpolator.go_to_frame.connect(self.video_player.setCurrentFrame)
        self.interpolator.preload_frames.connect(self.video_player.preload_frames)
        self.interpolator.interpolation_accepted.connect(
            self.errorsExplorer.accepted_interpolation
        )
        self.errorsExplorer.interpolate_all_flagged.connect(
            self.interpolator.batch_interpolate
        )

        self.id_labels = IdLabels()
        self.id_labels.needToDraw.connect(self.video_player.update)
        self.id_labels.needToDraw.connect(new_changes)
        self.id_labels.presenceIntervalRequested.connect(self.set_identity_presence)

        self.setup_points = SetupPoints()
        self.setup_points.needToDraw.connect(self.video_player.update)
        self.setup_points.needToDraw.connect(new_changes)

        self.length_calibrator = LengthCalibrator()
        self.length_calibrator.needToDraw.connect(self.video_player.update)
        self.length_calibrator.needToDraw.connect(new_changes)

        self.video_player.canvas.click_event.connect(self.setup_points.click_event)
        self.video_player.canvas.click_event.connect(self.length_calibrator.click_event)
        self.video_player.canvas.move_event.connect(self.length_calibrator.move_event)
        self.video_player.canvas.click_event.connect(self.interpolator.click_event)

        self.video_info = VideoInfo()

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setContentsMargins(8, 0, 0, 0)

        self.additional_info = AdditionalInfo(self)

        tabs = QTabWidget()
        tabs.setMovable(True)
        tabs.addTab(self.id_groups, "Groups")
        tabs.addTab(self.id_labels, "Labels")
        tabs.addTab(self.setup_points, "Setup Points")
        tabs.addTab(self.length_calibrator, "Length Calibration")
        tabs.addTab(self.mark_metadata, "Mark Metadata")
        tabs.addTab(self.video_info, "Video Info")
        TransparentDisabledOverlay(
            "Disable the Interpolator to\nenable these extra tools", tabs
        )

        self.additional_info.metadata_visibility.toggled.connect(
            lambda checked: tabs.setTabVisible(
                tabs.indexOf(self.mark_metadata), checked
            )
        )
        tabs.setTabVisible(
            tabs.indexOf(self.mark_metadata),
            self.additional_info.metadata_visibility.isChecked(),
        )

        right_splitter.setMinimumWidth(250)
        tabs.currentChanged.connect(self.video_player.update)
        right_splitter.addWidget(tabs)
        right_splitter.setHandleWidth(16)
        right_splitter.addWidget(self.additional_info)
        self.interpolator.enabled_changed.connect(
            lambda enabled: tabs.setEnabled(not enabled)
        )

        self.respect_tracking_intervals = QCheckBox("Respect tracking intervals")
        self.respect_tracking_intervals.setChecked(True)
        self.respect_tracking_intervals.stateChanged.connect(
            self.respect_tracking_intervals_changed
        )

        left_splitter = QVBoxLayout()
        left_splitter.addWidget(self.errorsExplorer)
        left_splitter.addWidget(self.respect_tracking_intervals)
        left_splitter.addWidget(QHLine())
        left_splitter.addWidget(self.interpolator)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.video_player.layout().setContentsMargins(8, 0, 8, 0)
        left_widget = QWidget()
        left_widget.setLayout(left_splitter)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.video_player)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes((1, 10, 1))
        self.centralWidget().layout().addWidget(splitter)
        self.centralWidget().setEnabled(False)
        self.centralWidget().layout().setContentsMargins(8, 0, 8, 8)

        self.selected_id: int | None = None
        self.selected_blob: Blob | None = None
        self.selection_last_location: tuple[float, float] | None = None

        self.video_player.painting_time.connect(self.paint)
        self.current_frame_number = -1
        self.trajectories: np.ndarray
        """Float, positions of animals"""
        self.unidentified: np.ndarray
        """Bool, there is some identity without centroid"""
        self.duplicated: np.ndarray
        """Bool, some centroid have the same identity"""

        session_menu = self.menuBar().addMenu("Session")
        assert session_menu is not None

        open_action = QAction("Open session", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setIcon(QIcon.fromTheme("document-open"))
        open_action.triggered.connect(self.open_file_dialog)
        session_menu.addAction(open_action)

        self.reset_action = QAction("Reset session...", self)
        self.reset_action.setShortcut("Ctrl+R")
        self.reset_action.setEnabled(False)
        self.reset_action.setIcon(get_icon("refresh"))
        self.reset_action.triggered.connect(self.reset_session)
        session_menu.addAction(self.reset_action)

        self.save_action = QAction("Save session", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.setIcon(QIcon.fromTheme("document-save"))
        self.save_action.triggered.connect(self.save_session)
        session_menu.addAction(self.save_action)

        drawing_flags = self.menuBar().addMenu("Draw")

        self.view_labels = QAction("Labels", self)
        self.view_labels.setShortcut("Alt+L")
        self.view_contours = QAction("Contours", self)
        self.view_contours.setShortcut("Alt+C")
        self.view_centroids = QAction("Centroids", self)
        self.view_centroids.setShortcut("Alt+P")
        self.view_bboxes = QAction("Bounding boxes", self)
        self.view_bboxes.setShortcut("Alt+B")
        self.view_trails = QAction("Trails", self)
        self.view_trails.setShortcut("Alt+T")
        self.view_ROIs = QAction("Regions of interest", self)
        self.view_ROIs.setShortcut("Alt+R")

        drawing_flags.addActions(
            (
                self.view_labels,
                self.view_contours,
                self.view_centroids,
                self.view_bboxes,
                self.view_trails,
                self.view_ROIs,
            )
        )

        for action in drawing_flags.actions():
            action.setCheckable(True)
            action.toggled.connect(self.video_player.update)

        find_identity_action = QAction("Find identity", self)
        find_identity_action.setIcon(QIcon.fromTheme("edit-find"))
        find_identity_action.setShortcut("Ctrl+F")
        find_identity_action.triggered.connect(self.find_identity)

        drawing_flags.addSeparator()
        drawing_flags.addAction(find_identity_action)

        # Defaults
        self.view_labels.setChecked(True)
        self.view_contours.setChecked(True)
        self.view_centroids.setChecked(True)
        self.view_bboxes.setChecked(False)
        self.view_trails.setChecked(True)
        self.view_ROIs.setChecked(False)

        tooltips = toml.load(Path(__file__).parent / "tooltips.toml")

        self.additional_info.metadata_visibility.setToolTip(
            tooltips["metadata_visibility"]
        )
        self.interpolator.apply_btn.setToolTip(tooltips["apply_interpolation"])
        self.interpolator.abort_btn.setToolTip(tooltips["abort_interpolation"])
        self.errorsExplorer.jumps_th.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.reset_jumps.setToolTip(tooltips["reset_jumps"])
        self.errorsExplorer.jumps_th_label.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.jumps_th.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.update_btn.setToolTip(tooltips["update_errors"])
        self.errorsExplorer.autoselect_errors.setToolTip(tooltips["autoselect_error"])
        self.respect_tracking_intervals.setToolTip(tooltips["tracking_intervals_check"])
        self.interpolator.interpolation_order_box.setToolTip(
            tooltips["interpolation_order"]
        )
        self.interpolator.interpolation_order_label.setToolTip(
            tooltips["interpolation_order"]
        )
        for index in range(self.interpolator.input_size_row.count()):
            self.interpolator.input_size_row.itemAt(index).widget().setToolTip(
                tooltips["input_size"]
            )

        self.unsaved_changes = False

    def open_file_dialog(self) -> None:
        file_dialog = QFileDialog()
        settings_key = f"{self.__class__.__name__}_filedialog_state"
        state = self.settings.value(settings_key, b"")
        if state:
            file_dialog.restoreState(state)
        file_paths = file_dialog.getExistingDirectory(
            self, "Open session directory", options=QFileDialog.Option.ShowDirsOnly
        )
        self.settings.setValue(settings_key, file_dialog.saveState())
        self.open_session(file_paths)

    def find_identity(self) -> None:
        """Displays a QInputDialog to select an identity to, then, find
        its blob, select it and center the video canvas to its centroid"""
        to_find, success = QInputDialog.getText(
            self,
            "",
            "Identity to find:",
            text=str(self.selected_id) if self.selected_id not in (None, -1) else "",
            flags=Qt.WindowType.SplashScreen,
        )
        to_find = to_find.strip()

        if not success or not to_find:
            return

        if to_find.isdigit():
            identity_to_find = int(to_find)
        else:
            try:
                identity_to_find = self.id_labels.labels.index(to_find)
            except ValueError:
                QMessageBox.warning(
                    self, "Find error", f"Identity not recognized: {to_find!r}"
                )
                return

        for blob in self.blobs.blobs_in_video[self.current_frame_number]:
            for identity, centroid in blob.final_ids_and_centroids:
                if identity == identity_to_find:
                    self.selected_blob = blob
                    self.selected_id = identity
                    self.selection_last_location = centroid
                    self.current_frame_number = -1  # this makes info_widget to update
                    self.video_player.center_canvas_at(*centroid, self.max_zoom)
                    return

        QMessageBox.warning(
            self, "Find error", f"Identity {identity_to_find} not found in this frame"
        )

    def set_identity_presence(self, identity: int) -> None:
        """Asks for the frame range in which `identity` is present.

        Everything outside it stops counting as a "Miss id" error. See
        `Session.presence_intervals`.
        """
        intervals = self.session.presence_intervals.get(
            str(identity), [[0, self.n_frames - 1]]
        )
        start, end = intervals[0]

        text, ok = QInputDialog.getText(
            self,
            f"Set presence for identity {identity}",
            f"Frame range in which identity {identity} is present (start-end):",
            text=f"{start}-{end}",
        )
        if not ok or not text:
            return

        try:
            start_text, end_text = text.split("-")
            start, end = int(start_text), int(end_text)
            if not 0 <= start < end < self.n_frames:
                raise ValueError(text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid frame range",
                f"Write two frame numbers as start-end, between 0 and"
                f" {self.n_frames - 1}, for instance 0-{self.n_frames - 1}.",
            )
            return

        # Rebound rather than mutated: Session is a plain class and the empty
        # default dict is shared between instances.
        self.session.presence_intervals = {
            **self.session.presence_intervals,
            str(identity): [[start, end]],
        }
        self.unsaved_changes = True
        self.errorsExplorer.update_list_of_errors()
        logging.info("Identity %s is present from frame %s to %s", identity, start, end)

    def blobs_reachable_by_propagation(self, blob: Blob) -> list[Blob]:
        """The blobs a propagated identity edit on `blob` can reach.

        Propagation stops at the edges of a fragment, so that is the whole
        extent of what needs snapshotting before one. A blob added by the user,
        or a session whose fragments never loaded, has no fragment to follow.
        """
        if self.fragments is None or blob.fragment_identifier < 0:
            return [blob]

        fragment = self.fragments[blob.fragment_identifier]
        return [
            candidate
            # end_frame is exclusive
            for frame in range(fragment.start_frame, fragment.end_frame)
            for candidate in self.blobs.blobs_in_video[frame]
            if candidate.fragment_identifier == fragment.identifier
        ]

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        if a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.id_groups.uncheck_edit_buttons()
            self.setup_points.add.setChecked(False)
        elif a0.key() == Qt.Key.Key_Z and a0.modifiers() == (
            Qt.KeyboardModifier.ControlModifier
        ):
            self.undo_manager.undo()
        elif a0.key() == Qt.Key.Key_I:
            self.start_interpolation_from_selection()

    def start_interpolation_from_selection(self) -> None:
        """Starts interpolating whatever is currently selected.

        The selected error first: working through the error list is the common
        path, and it already knows the identity and the whole extent of the
        gap. Otherwise the selected identity at the current frame, which the
        interpolator then expands by itself.
        """
        # selected_error only exists once a row has been clicked.
        if hasattr(self.errorsExplorer, "selected_error"):
            kind, identity, start, length = self.errorsExplorer.selected_error
            if kind in ("Jump", "Miss id") and identity > 0:
                self.interpolator.set_interpolation_params(
                    identity, start, start + length
                )
                return

        if self.selected_id is not None and self.selected_id > 0:
            self.interpolator.set_interpolation_params(
                self.selected_id,
                self.current_frame_number,
                self.current_frame_number + 1,
            )

    def manageDropedPaths(self, paths: Sequence[str]) -> None:
        if not paths:
            return
        if len(paths) > 1:
            QMessageBox.warning(
                self,
                "Error with dragged files",
                "Cannot open more than one session at once",
            )
            return
        self.open_session(paths[0])

    def go_to_error(
        self, kind: str, start: int, length: int, where: np.ndarray, identity: int
    ) -> None:
        if where.ndim == 2:
            # Set the zoom to capture all positions of 'where'
            xmax, ymax = np.nanmax(where, axis=0)
            xmin, ymin = np.nanmin(where, axis=0)
            zoom_scale = max(self.max_zoom, 1.8 * (xmax - xmin), 1.8 * (ymax - ymin))
            self.video_player.center_canvas_at(
                0.5 * (xmax + xmin), 0.5 * (ymin + ymax), zoom_scale=zoom_scale
            )
            where = where[0]
        else:
            self.video_player.center_canvas_at(*where, zoom_scale=self.max_zoom)
        self.selection_last_location = None if np.isnan(where).any() else tuple(where)

        self.selected_id = identity
        if kind in ("Jump", "Miss id"):
            self.interpolator.set_interpolation_params(identity, start, start + length)
        else:
            self.interpolator.setActivated(False)
        self.video_player.setCurrentFrame(
            start - 1 if start > 0 and kind in ("Jump", "Miss id") else start, True
        )

    def reset_session(self) -> None:
        start, finish = self.reset_session_dialog.exec()
        logging.info(f"Resetting session from {start} to {finish}")
        if start is None:
            return

        for blobs_in_frame in track(
            self.blobs.blobs_in_video[start:finish], "Resetting user corrections"
        ):
            for blob in blobs_in_frame:
                if blob.added_by_user:
                    self.blobs.blobs_in_video[blob.frame_number].remove(blob)
                else:
                    blob.user_generated_identities = None  # type: ignore
                    blob.user_generated_centroids = None  # type: ignore

        self.errorsExplorer.non_accepted_jumps[start:finish] = True
        self.update_trajectories_range(start, finish)

    def autosave(self) -> None:
        if not self.unsaved_changes:
            return
        if self.saving_session_thread is not None and (
            self.saving_session_thread.isRunning()
        ):
            return
        if self.save_thread is not None and self.save_thread.isRunning():
            return

        logging.info("Autosaving session")
        self.save_session(silent=True)

    def save_session(self, wait: bool = False, silent: bool = False) -> None:
        """Writes the session, the blobs and the trajectories to disk.

        `silent` skips the modal progress dialogs, which is what makes the
        autosave usable: a dialog stealing focus every five minutes in the
        middle of a click-heavy job would be worse than not autosaving.
        """
        self._save_generation = self._edit_generation
        self._save_silent = silent
        self.session.identities_labels = self.id_labels.get_labels()[1:]
        self.session.identities_colors = [
            c.name() for c in self.id_labels.get_colors()[0][1:]
        ]
        self.session.identities_groups = self.id_groups.get_groups()
        self.session.setup_points = self.setup_points.get_points()
        self.session.length_calibrations = self.length_calibrator.get_calibrations()
        self.session.last_validated = datetime.now()

        saving_thread = SaveSessionObjects(
            self.session, self.blobs, self, self.blobs_path, verbose=not silent
        )
        self.saving_session_thread = saving_thread

        if silent:
            # ListOfBlobs.save temporarily disconnects the blob graph, so it
            # cannot run on a worker thread while the user keeps editing it.
            # Doing it here, on the GUI thread, rules out concurrent edits.
            saving_thread.run()
            self.report_save_failure()
            self._start_save_trajectories(silent=True)
            return

        saving_thread.finished.connect(self.report_save_failure)
        saving_thread.finished.connect(
            lambda: self._start_save_trajectories(silent=silent)
        )

        if not silent:
            progress_bar = QProgressDialog(
                "Saving session, please wait...",
                "Close app",
                0,
                0,
                self,
                Qt.WindowType.SplashScreen,
            )
            progress_bar.canceled.connect(saving_thread.terminate)
            progress_bar.canceled.connect(sys.exit)
            progress_bar.setModal(True)
            saving_thread.finished.connect(progress_bar.cancel)
            progress_bar.show()

        saving_thread.start()
        if wait:
            saving_thread.wait()

    def report_save_failure(self) -> None:
        """Tells the user when a save failed, and where the work went instead.

        A failed save used to raise on a worker thread and be swallowed, so the
        session looked saved when nothing had been written.
        """
        thread = self.saving_session_thread
        if thread is None or thread.success:
            return

        if self._save_silent:
            # The autosave timer must never pop a blocking dialog: log it, flag
            # it in the title, and wait longer before the next attempt.
            logging.error(
                "Autosave failed: %s. %s",
                thread.error_message,
                f"A copy was saved at {thread.backup_path}"
                if thread.backup_path
                else "No backup could be written either",
            )
            self.setWindowTitle("Validator (AUTOSAVE FAILED, save manually)")
            self.autosave_timer.setInterval(
                min(2 * self.autosave_timer.interval(), 60 * 60 * 1000)
            )
            return

        if thread.backup_path is None:
            QMessageBox.critical(
                self,
                "Session not saved",
                f"The session could not be saved:\n{thread.error_message}\n\n"
                "Writing a backup copy failed as well. Check the free space and"
                " the permissions of your home folder, and do not close the"
                " Validator: the work is still in memory.",
            )
            return

        QMessageBox.warning(
            self,
            "Session saved elsewhere",
            f"The session could not be saved in its own folder:\n"
            f"{thread.error_message}\n\nIt was saved here instead:\n"
            f"{thread.backup_path}\n\nCheck that the drive holding the session"
            " is still connected.",
        )

    def _start_save_trajectories(self, silent: bool = False):
        if self.saving_session_thread is not None and (
            not self.saving_session_thread.success
        ):
            return

        self.save_thread = SaveTrajectoriesThread(
            self.blobs.blobs_in_video, self.session, self.fragments
        )
        self.save_thread.finished.connect(self.finish_saving)

        if not silent:
            progress = QProgressDialog(
                "Computing trajectories",
                "Abort",
                0,
                self.session.number_of_frames + 1,
                self,
                Qt.WindowType.SplashScreen,
            )
            progress.setMinimumDuration(1500)
            progress.setModal(True)
            progress.canceled.connect(self.save_thread.quit)
            self.save_thread.progress_changed.connect(progress.setValue)

        self.save_thread.start()

    def finish_saving(self) -> None:
        if self.save_thread is not None and self.save_thread.success:
            # Edits made while saving are not in what was written
            if self._edit_generation == self._save_generation:
                self.unsaved_changes = False
            self.autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
            self.setWindowTitle("Validator")

    def check_unsaved_changes(self) -> None | QMessageBox.StandardButton:
        if not self.unsaved_changes:
            return None
        return QMessageBox.question(
            self,
            "Save changes?",
            "There are unsaved changes. Changes which are not saved will be"
            " permanently lost.",
            QMessageBox.StandardButton.Cancel
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Save,
        )

    def ask_for_session(self) -> None:
        answer = QMessageBox.information(
            self,
            "Open session",
            "Please, open a session to validate",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
        )
        if answer is QMessageBox.StandardButton.Open:
            self.open_file_dialog()

    def open_session(self, session: Session | Path | str | None) -> None:
        if not session:
            return

        match self.check_unsaved_changes():
            case QMessageBox.StandardButton.Cancel:
                return
            case QMessageBox.StandardButton.Save:
                self.save_session()
            case QMessageBox.StandardButton.Discard:
                pass  # override unsaved changes
            case None:
                pass  # there are not unsaved changes
            case other:
                raise ValueError(other)

        if not isinstance(session, Session):
            session = open_session(self, self.settings, session)
            if session is None:
                return
        self.session = session

        if (
            hasattr(session, "timers")
            and "Tracking session" in session.timers
            and not session.timers["Tracking session"].finished
        ):
            answer = QMessageBox.question(
                self,
                "Loading session warning",
                f"The session you are trying to load ({session}) has not finished,"
                " unexpected behavior can happen. Do you want to continue?",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return

        loading_thread = LoadSessionObjects(session, self)
        progress_bar = QProgressDialog(
            "Loading session, please wait...",
            "Close app",
            0,
            0,
            self,
            Qt.WindowType.SplashScreen,
        )
        progress_bar.canceled.connect(loading_thread.terminate)
        progress_bar.canceled.connect(sys.exit)
        progress_bar.setModal(True)

        loading_thread.finished.connect(lambda: self._finalize_loading(loading_thread))
        loading_thread.finished.connect(progress_bar.cancel)
        loading_thread.start()
        progress_bar.show()

    def _finalize_loading(self, loading_thread: LoadSessionObjects) -> None:
        if loading_thread.blobs is None:
            QMessageBox.warning(
                self, "Loading session error", "List of blobs not found"
            )
            return
        session = self.session

        self.blobs_path = loading_thread.loaded_from
        self.undo_manager.clear()

        # remove selection
        self.selected_blob = None
        self.selected_id = None
        self.selection_last_location = None

        self.id_groups.load_groups(session.identities_groups)
        self.id_labels.load(
            session.identities_labels or [str(i + 1) for i in range(session.n_animals)],
            session.identities_colors,
        )
        self.blobs = loading_thread.blobs
        self.fragments = loading_thread.fragments

        self.additional_info.fragments = self.fragments

        self.video_player.update_video_paths(
            session.video_paths,
            session.number_of_frames,
            (session.width, session.height),
            session.frames_per_second,
        )
        self.n_animals = session.n_animals
        self.n_frames = session.number_of_frames
        self.generate_trajectories(self.blobs.blobs_in_video)
        try:
            self.max_zoom = 2 * session.median_body_length
        except AttributeError:
            logging.warning('No "median_body_length" found in session')
            self.max_zoom = 50 * np.nanmedian(
                np.sqrt((np.diff(self.trajectories, axis=0) ** 2).sum(-1))
            )
        self.centralWidget().setEnabled(True)
        self.dbl_click_dialog = DblClickDialog(self, session.n_animals)

        self.respect_tracking_intervals.setVisible(
            not (
                session.tracking_intervals is None
                or (session.tracking_intervals[0] == [0, session.number_of_frames])
            )
        )

        self.video_info.set_data(session)
        self.setup_points.load_points(session.setup_points)
        self.length_calibrator.load(
            session.length_calibrations
            if hasattr(session, "length_calibrations")
            else None
        )

        tracking_intervals = (
            session.tracking_intervals
            if self.respect_tracking_intervals.isChecked()
            else None
        )
        self.errorsExplorer.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.blobs,
            tracking_intervals,
            session=self.session,
        )
        self.interpolator.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.blobs,
            tracking_intervals,
        )
        self.video_player.update()
        self.unsaved_changes = False

        if hasattr(session, "roi_list") and session.roi_list:
            self.view_ROIs.setEnabled(True)
            self.view_ROIs.setChecked(True)
            self.ROI_patches = build_ROI_patches_from_list(
                session.roi_list, session.width, session.height
            )
        else:
            self.view_ROIs.setChecked(False)
            self.view_ROIs.setEnabled(False)

        self.setWindowTitle("Validator | " + session.session_folder.name)

        self.save_action.setEnabled(True)
        self.reset_action.setEnabled(True)

        self.reset_session_dialog = ResetSessionDialog(self, session.number_of_frames)

    def respect_tracking_intervals_changed(self, checked: bool) -> None:
        "Transfer the status information of respect_tracking_intervals to the widgets that need it"
        tracking_intervals = self.session.tracking_intervals if checked else None
        self.errorsExplorer.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.blobs,
            tracking_intervals,
            session=self.session,
        )
        self.interpolator.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.blobs,
            tracking_intervals,
        )

    def click_on_canvas(self, event: CanvasMouseEvent) -> None:
        self.selected_blob, self.selected_id, self.selection_last_location = clicked_id(
            self.blobs.blobs_in_video[self.current_frame_number], event
        )
        self.id_groups.selected_id(self.selected_id)
        self.current_frame_number = -1  # this makes info_widget to update
        self.video_player.update()

    def double_click_on_canvas(self, event: CanvasMouseEvent) -> None:
        if (
            self.selected_blob is None
            or self.id_groups.is_active()
            or event.button != Qt.MouseButton.LeftButton
        ):
            return
        self.selected_blob, self.selected_id, self.selection_last_location = clicked_id(
            self.blobs.blobs_in_video[self.current_frame_number], event
        )
        if self.selected_blob is None:
            return
        if self.selection_last_location is None:
            # clicked on a blob without centroids
            answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(0)
            if answer == DblClickDialog.Answers.ChangeId:
                self.selected_blob.add_centroid(event.xy_data, new_id)
                self.update_trajectories_range(self.current_frame_number)
            return

        # clicked on a blob with centroid
        answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(
            self.interpolator.animal_id + 1
            if self.interpolator.isEnabled()
            else self.selected_id
        )

        if answer == DblClickDialog.Answers.Reset:
            new_id = None
            answer = DblClickDialog.Answers.ChangeId

        if answer == DblClickDialog.Answers.Remove:
            new_id = -1
            answer = DblClickDialog.Answers.ChangeId

        if answer == DblClickDialog.Answers.ChangeId:
            self.undo_manager.push_identities(
                f"Change identity {self.selected_id} to {new_id}",
                (
                    self.blobs_reachable_by_propagation(self.selected_blob)
                    if propagate
                    else [self.selected_blob]
                ),
            )
            self.selected_blob.update_identity(
                self.selected_id, new_id, self.selection_last_location
            )
            # If we are interpolating, do not update list of errors, let the interpolation finish
            update_errors = not self.interpolator.isEnabled()
            if propagate:
                lower, upper = self.selected_blob.propagate_identity(
                    self.selected_id, new_id, self.selection_last_location
                )
                if lower != upper:
                    self.light_popup.info(
                        "Identification change",
                        f"Identification propagated from frame {lower} to frame"
                        f" {upper}",
                    )
                self.update_trajectories_range(
                    lower, upper + 1, update_errors=update_errors
                )
            else:
                self.update_trajectories_range(
                    self.current_frame_number, update_errors=update_errors
                )
            return
        if (
            answer == DblClickDialog.Answers.Interpolate
            and self.selected_id is not None
            and self.selected_id > 0
        ):
            self.interpolator.set_interpolation_params(
                self.selected_id,
                self.current_frame_number,
                self.current_frame_number + 1,
            )
            return

        if answer == DblClickDialog.Answers.Swap:
            self.swap_selected_identity()
            return

        if answer == DblClickDialog.Answers.Extend:
            self.extend_selected_centroid()
            return

        if answer == DblClickDialog.Answers.Merge:
            self.merge_selected_blob(propagate)

    def swap_selected_identity(self) -> None:
        """Exchanges the selected identity with another, along the fragment.

        Two animals that cross and come out the wrong way round need both
        identities moved. Renaming one of them, which is all "Change id"
        offers, leaves the other duplicated for the rest of the fragment.
        """
        if self.selected_id is None or self.selected_id <= 0:
            return

        target_id, ok = QInputDialog.getInt(
            self,
            "Swap identities",
            f"Exchange identity {self.selected_id} with:",
            1,  # value
            1,  # minimum
            self.n_animals,  # maximum
        )
        if not ok or target_id == self.selected_id:
            return

        self.undo_manager.push_identities(
            f"Swap identities {self.selected_id} and {target_id}",
            self.blobs_reachable_by_propagation(self.selected_blob),
        )
        lower, upper = self.selected_blob.propagate_swap_identity(
            self.selected_id, target_id, self.selection_last_location
        )
        self.update_trajectories_range(lower, upper + 1)
        if lower != upper:
            self.light_popup.info(
                "Identities swapped",
                f"Swap propagated from frame {lower} to frame {upper}",
            )

    def extend_selected_centroid(self) -> None:
        """Repeats the selected centroid over the following frames.

        An animal that stops moving can stop being segmented, leaving a gap
        that interpolation fills with a straight line between two points that
        are in the same place anyway. Stamping the centroid says the same thing
        in one step.
        """
        if self.selected_id is None or self.selected_id <= 0:
            return

        n_frames, ok = QInputDialog.getInt(
            self,
            "Extend centroid",
            "Repeat this centroid over the next N frames:",
            10,  # value
            1,  # minimum
            self.n_frames - self.current_frame_number,  # maximum
        )
        if not ok:
            return

        frames = range(
            self.current_frame_number,
            min(self.n_frames, self.current_frame_number + n_frames),
        )
        # add_centroid can create a blob as well as edit one, so the blob lists
        # go into the snapshot too.
        self.undo_manager.push_frames(
            f"Extend identity {self.selected_id} over {len(frames)} frames", frames
        )

        for frame in frames:
            self.blobs.add_centroid(
                frame, self.selected_id, self.selection_last_location
            )
        self.update_trajectories_range(frames.start, frames.stop)

    def merge_selected_blob(self, propagate: bool) -> None:
        """Merges the selected blob with the one carrying another identity.

        Blobs are found by identity rather than by object, because the selected
        blob does not exist as the same object in later frames; its identity
        does. With `propagate` the merge repeats for as long as both identities
        keep appearing on separate blobs, bounded by
        `MERGE_PROPAGATION_FRAMES`.
        """
        target_id, ok = QInputDialog.getInt(
            self,
            "Merge blobs",
            "Merge this blob with the blob of identity:",
            1,  # value
            1,  # minimum
            self.n_animals,  # maximum
        )
        if not ok:
            return

        identities_to_merge = {target_id}
        if self.selected_id is not None and self.selected_id > 0:
            identities_to_merge.add(self.selected_id)

        start_frame = self.current_frame_number
        end_frame = (
            min(start_frame + MERGE_PROPAGATION_FRAMES, self.blobs.number_of_frames)
            if propagate
            else start_frame + 1
        )

        merges: list[tuple[int, list[int]]] = []
        for frame in range(start_frame, end_frame):
            indices = [
                index
                for index, blob in enumerate(self.blobs.blobs_in_video[frame])
                if identities_to_merge.intersection(blob.all_final_identities)
            ]
            if len(indices) >= 2:
                merges.append((frame, indices))
            elif propagate:
                # The parts stopped being separate; anything further along is
                # a different situation.
                break

        if not merges:
            self.light_popup.info(
                "Nothing to merge",
                "Merging needs two separate blobs in the same frame, each carrying"
                f" one of the identities {sorted(identities_to_merge)}. No frame"
                " in this range has that.",
            )
            return

        new_identity = (
            self.selected_id
            if self.selected_id is not None and self.selected_id > 0
            else target_id
        )
        self.undo_manager.push_frames(
            f"Merge identities {sorted(identities_to_merge)} in"
            f" {len(merges)} frame(s)",
            [frame for frame, _ in merges],
        )
        for frame, indices in merges:
            self.blobs.merge_blobs(frame, indices, new_identity)

        self.update_trajectories_range(start_frame, merges[-1][0] + 1)
        if len(merges) > 1:
            self.light_popup.info(
                "Blobs merged", f"Merged blobs in {len(merges)} frames."
            )

    def paint(self, painter: CanvasPainter, frame_number: int) -> None:
        blobs_in_frame = self.blobs.blobs_in_video[frame_number]
        if self.id_groups.is_active():
            cmap, cmap_alpha = self.id_groups.get_cmaps(self.session.n_animals)
        else:
            cmap, cmap_alpha = self.id_labels.get_colors()
        update_info_widget = frame_number != self.current_frame_number
        self.current_frame_number = frame_number

        if self.selected_blob is not None and self.selected_blob not in blobs_in_frame:
            self.selected_blob, self.selection_last_location = find_selected_blob(
                blobs_in_frame, self.selected_id, self.selection_last_location
            )

        if self.view_trails.isChecked():
            paintTrails(self.current_frame_number, painter, self.trajectories, cmap)

        if self.view_ROIs.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 0, 0, 50))
            painter.drawPath(self.ROI_patches)

        paintBlobs(
            self.view_contours.isChecked(),
            self.view_centroids.isChecked(),
            self.view_bboxes.isChecked(),
            self.view_labels.isChecked(),
            painter,
            blobs_in_frame,
            cmap,
            cmap_alpha,
            self.selected_blob,
            self.selection_last_location,
            self.id_labels.get_labels(),
            self.mark_metadata(blobs_in_frame, self.fragments),
        )

        if self.setup_points.isVisible():
            self.setup_points.paint_on_canvas(painter)

        if self.length_calibrator.isVisible():
            self.length_calibrator.paint_on_canvas(painter)

        if self.interpolator.isEnabled():
            self.interpolator.paint_on_canvas(painter, frame_number)

        if update_info_widget:
            self.additional_info.set_data(self.selected_blob)

    def closeEvent(self, event: QCloseEvent) -> None:
        match self.check_unsaved_changes():
            case QMessageBox.StandardButton.Discard:
                return super().closeEvent(event)
            case QMessageBox.StandardButton.Save:
                self.save_session(wait=True)
                return super().closeEvent(event)
            case QMessageBox.StandardButton.Cancel:
                return event.ignore()
            case None:  # there are not unsaved changes
                return super().closeEvent(event)
            case other:
                raise ValueError(other)

    def update_trajectories_range(
        self, start: int, finish: int | None = None, update_errors: bool = True
    ) -> None:
        finish = start + 1 if finish is None else finish
        logging.debug(f"Updating trajectories from frame {start} to {finish}")
        ids_in_frame = set()
        self.trajectories[start:finish] = np.nan
        self.duplicated[start:finish] = False
        self.unidentified[start:finish] = False
        for frame in range(start, finish):
            ids_in_frame.clear()
            for blob in self.blobs.blobs_in_video[frame]:
                for identity, centroid in blob.final_ids_and_centroids:
                    if identity not in (None, 0):
                        self.trajectories[frame, identity - 1] = centroid
                        if identity in ids_in_frame:
                            self.duplicated[frame, identity - 1] = True
                        ids_in_frame.add(identity)
                    else:
                        self.unidentified[frame] = True
        self.interpolator.trajectories_have_been_updated()
        if update_errors:
            self.errorsExplorer.update_list_of_errors()
        self.video_player.update()
        self.unsaved_changes = True

    def generate_trajectories(self, blobs_in_video: list[list[Blob]]) -> None:
        number_of_frames = len(blobs_in_video)
        self.trajectories = np.full((number_of_frames, self.n_animals, 2), np.nan)
        self.unidentified = np.zeros(number_of_frames, bool)
        self.duplicated = np.zeros((number_of_frames, self.n_animals), bool)
        ids_in_frame: set[int] = set()

        progress_bar = QProgressDialog(
            "Analyzing trajectories",
            "Close app",
            0,
            number_of_frames - 1,
            self,
            Qt.WindowType.SplashScreen,
        )
        progress_bar.setMinimumDuration(1000)
        progress_bar.canceled.connect(sys.exit)
        progress_bar.setModal(True)

        for frame, blobs_in_frame in enumerate(blobs_in_video):
            progress_bar.setValue(frame)
            ids_in_frame.clear()
            for blob in blobs_in_frame:
                for identity, centroid in blob.final_ids_and_centroids:
                    if identity not in (None, 0):
                        self.trajectories[frame, identity - 1] = centroid
                        if identity in ids_in_frame:
                            self.duplicated[frame, identity - 1] = True
                        ids_in_frame.add(identity)
                    else:
                        self.unidentified[frame] = True


def clicked_id(
    blobs: list[Blob], click: CanvasMouseEvent
) -> tuple[Blob | None, int | None, tuple[float, float] | None]:
    candidates: list[tuple[Blob, int | None, tuple[float, float], float]] = []

    for blob in blobs:
        if blob.contains_point(click.xy_data):
            for identity, centroid in blob.final_ids_and_centroids:
                dist = click.sq_distance_to(centroid)
                candidates.append((blob, identity, centroid, dist))
            if not candidates:  # blob with no centroids
                return blob, -1, None
            break

    if candidates:
        return min(candidates, key=lambda x: x[-1])[:-1]

    for blob in blobs:
        for identity, centroid in blob.final_ids_and_centroids:
            dist = click.sq_distance_to(centroid)
            if dist < (SELECT_POINT_DIST * click.zoom):
                candidates.append((blob, identity, centroid, dist))

    if candidates:
        return min(candidates, key=lambda x: x[-1])[:-1]

    return None, -1, None


class SaveTrajectoriesThread(QThread):
    progress_changed = Signal(int)

    def __init__(
        self,
        blobs_in_video: list[list[Blob]],
        session: Session,
        list_of_fragments: list[Fragment] | None,
    ) -> None:
        super().__init__()
        self.blobs_in_video = blobs_in_video
        self.fragments = list_of_fragments
        self.session = session
        self.success = False
        self.finished.connect(
            lambda: self.progress_changed.emit(len(self.blobs_in_video) + 1)
        )

    def run(self) -> None:
        self.abort = False

        trajectories = produce_output_dict(
            self.blobs_in_video,
            self.session,
            progress_bar=self.progress_changed,
            abort=lambda: self.abort,
        )
        if self.abort:
            return
        if not trajectories:  # empty dict
            logging.error("Trajectories could not be produced")
            return

        save_trajectories(
            self.session.trajectories_folder,
            trajectories,
            self.session.trajectories_formats,
        )

        self.progress_changed.emit(self.session.number_of_frames)
        self.success = True

    def quit(self) -> None:
        self.abort = True


class ResetSessionDialog(QDialog):
    """Pop up to select the range of the user corrections reset.
    Reset is activated in the menu bar / Session / Reset session"""

    class Answers(Enum):
        Cancel = 0
        RangeReset = 1
        AllReset = 2

    def __init__(self, parent, n_frames: int) -> None:
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(
            QLabel(
                '<span style="font-weight:600">Resetting removes all user corrections'
            ),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            QLabel("Range of frames to reset:"), alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.double_slider = LabelRangeSlider(0, n_frames, parent)
        layout.addWidget(self.double_slider)
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton(get_icon("cancel"), "Cancel")
        range_btn = QPushButton(get_icon("refresh"), "Reset range")
        all_btn = QPushButton(get_icon("refresh"), "Reset all")
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(range_btn)
        btn_layout.addWidget(all_btn)
        layout.addLayout(btn_layout)

        cancel_btn.clicked.connect(lambda: self.done(self.Answers.Cancel.value))
        range_btn.clicked.connect(lambda: self.done(self.Answers.RangeReset.value))
        all_btn.clicked.connect(lambda: self.done(self.Answers.AllReset.value))

        cancel_btn.setAutoDefault(False)
        range_btn.setAutoDefault(False)
        all_btn.setAutoDefault(False)

    def exec(self) -> tuple[int | None, int | None]:  # type: ignore
        match self.Answers(super().exec()):
            case self.Answers.Cancel:
                return None, None
            case self.Answers.RangeReset:
                return self.double_slider.value()
            case self.Answers.AllReset:
                return self.double_slider.minimum(), self.double_slider.maximum()
