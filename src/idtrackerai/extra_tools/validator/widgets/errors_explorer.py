import logging
import warnings

import numpy as np
from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from idtrackerai import ListOfBlobs, Session
from idtrackerai.GUI_tools import get_icon, key_event_modifier

from .gaps import find_gap_anchors, gap_is_unambiguous

FLAG_COLUMN = 4


class CustomTableWidget(QTableWidget):
    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class CustomTableWidgetItem(QTableWidgetItem):
    def __init__(self, value: str | int):
        super().__init__("" if value == -1 else str(value))
        self.setData(  # convert numpy ints into Python ints
            Qt.ItemDataRole.UserRole, value if isinstance(value, str) else int(value)
        )
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        return self.data(Qt.ItemDataRole.UserRole) < other.data(
            Qt.ItemDataRole.UserRole
        )


class ErrorsExplorer(QWidget):
    go_to_error = Signal(str, int, int, np.ndarray, int)
    # kind, start, end, where, id
    interpolate_all_flagged = Signal(list)
    # list of (identity, start, length), identity counted from 1

    def __init__(self):
        super().__init__()
        self.table = CustomTableWidget(1, 5)
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        horizontalHeader.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        horizontalHeader.setSectionResizeMode(
            FLAG_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        horizontalHeader.setMinimumSectionSize(10)
        verticalHeader = self.table.verticalHeader()
        verticalHeader.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        verticalHeader.setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["Type", "Id", "Start", "Length", "Flag"])
        self.table.setSortingEnabled(True)
        self.table.currentCellChanged.connect(self.cell_clicked)
        self.table.cellDoubleClicked.connect(self.cell_clicked)

        long_jumps_row = QHBoxLayout()

        long_jumps_row.addWidget(QLabel("Min duration"))
        self.min_duration = QSpinBox()
        self.min_duration.setValue(0)
        self.min_duration.setSuffix(" frames")
        self.min_duration.setSpecialValueText("all")
        self.min_duration.setToolTip(
            "Hide errors shorter than this. A one-frame gap in a long fragment"
            " is usually not worth a correction."
        )
        self.min_duration.valueChanged.connect(self.update_list_of_errors)
        long_jumps_row.addWidget(self.min_duration)
        long_jumps_row.addSpacing(20)

        self.jumps_th_label = QToggleSwitch("Jumps threshold")
        self.jumps_th_label.setChecked(True)
        long_jumps_row.addWidget(self.jumps_th_label)
        self.jumps_th = QSpinBox()
        self.jumps_th.setValue(10)
        self.jumps_th.setSpecialValueText("disabled")
        self.jumps_th.setSuffix(" std")
        self.jumps_th.setPrefix("avg + ")
        self.jumps_th.valueChanged.connect(self.update_list_of_errors)
        long_jumps_row.addWidget(self.jumps_th)
        self.reset_jumps = QToolButton()
        self.reset_jumps.setText("Reset")
        self.reset_jumps.clicked.connect(lambda: self.non_accepted_jumps.fill(True))
        self.reset_jumps.clicked.connect(self.update_list_of_errors)
        self.jumps_th_label.toggled.connect(self.update_list_of_errors)
        self.jumps_th_label.toggled.connect(self.jumps_th.setEnabled)
        self.jumps_th_label.toggled.connect(self.reset_jumps.setEnabled)
        long_jumps_row.addWidget(self.reset_jumps)

        pair_row = QHBoxLayout()
        self.filter_pair_btn = QCheckBox("Filter pair")
        self.filter_pair_btn.setToolTip(
            "Show only errors belonging to these two identities. Two animals"
            " that keep swapping with each other are worked on as a pair."
            "\n\nThis also hides every No-id error, which carries no identity."
        )
        self.filter_pair_btn.stateChanged.connect(self.update_list_of_errors)
        pair_row.addWidget(self.filter_pair_btn)

        self.id_1 = QSpinBox()
        self.id_1.setPrefix("ID ")
        self.id_1.setMinimum(1)
        self.id_1.setValue(1)
        self.id_1.valueChanged.connect(self.update_list_of_errors)
        pair_row.addWidget(self.id_1)

        self.id_2 = QSpinBox()
        self.id_2.setPrefix("ID ")
        self.id_2.setMinimum(1)
        self.id_2.setValue(2)
        self.id_2.valueChanged.connect(self.update_list_of_errors)
        pair_row.addWidget(self.id_2)
        pair_row.addStretch()

        flag_row = QHBoxLayout()
        flag_row.addWidget(QLabel("Vicinity"))
        self.vicinity_threshold = QSpinBox()
        self.vicinity_threshold.setRange(1, 9999)
        self.vicinity_threshold.setValue(50)
        self.vicinity_threshold.setSuffix(" px")
        self.vicinity_threshold.setToolTip(
            "A gap counts as unambiguous when no other animal comes closer than"
            " this to the straight line across it, in any frame of the gap."
        )
        flag_row.addWidget(self.vicinity_threshold)

        self.flag_unambiguous_btn = QPushButton("Flag unambiguous")
        self.flag_unambiguous_btn.setToolTip(
            'Check the "Flag" box of every Miss id gap that no other animal'
            " comes near. It only checks boxes, never clears them, and the"
            " flags are lost whenever the list of errors is rebuilt."
        )
        self.flag_unambiguous_btn.clicked.connect(self.flag_unambiguous_gaps)
        flag_row.addWidget(self.flag_unambiguous_btn)

        self.interpolate_flagged_btn = QPushButton("Interpolate all flagged")
        self.interpolate_flagged_btn.setToolTip(
            "Fill every flagged Miss id gap with a straight line between the"
            " last position before it and the first one after it. Ctrl+Z"
            " undoes all of them at once."
        )
        self.interpolate_flagged_btn.clicked.connect(self.interpolate_flagged)
        flag_row.addWidget(self.interpolate_flagged_btn)
        flag_row.addStretch()

        self.flag_summary = QLabel()

        self.autoselect_errors = QCheckBox("Autoselect first error")

        layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(layout)
        errors_header = QHBoxLayout()
        self.update_btn = QToolButton()
        self.update_btn.setIcon(get_icon("refresh"))
        self.update_btn.setShortcut(Qt.Key.Key_U)
        self.update_btn.clicked.connect(self.update_list_of_errors)

        self.next_error_btn = QToolButton()
        self.next_error_btn.setText("Next [N]")
        self.next_error_btn.setShortcut(Qt.Key.Key_N)
        self.next_error_btn.setToolTip("Select the next error in the list")
        self.next_error_btn.clicked.connect(self.select_next_error)

        self.left_label = QLabel()
        errors_header.addWidget(self.left_label)
        errors_header.addWidget(self.update_btn)
        errors_header.addWidget(self.next_error_btn)
        layout.addLayout(errors_header)
        layout.addWidget(self.autoselect_errors)
        layout.addWidget(self.table)
        layout.addLayout(long_jumps_row)
        layout.addLayout(pair_row)
        layout.addLayout(flag_row)
        layout.addWidget(self.flag_summary)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.trajectories: np.ndarray
        self.unidentified: np.ndarray
        self.duplicated: np.ndarray
        self.non_accepted_jumps: np.ndarray
        self.in_tracking_interval: np.ndarray
        self.session: Session | None = None

    def cell_clicked(self, row: int, col: int):
        if row < 0 or col < 0:
            return

        kind = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        identity = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        start: int = self.table.item(row, 2).data(Qt.ItemDataRole.UserRole)
        length = self.table.item(row, 3).data(Qt.ItemDataRole.UserRole)
        self.selected_error = kind, identity, start, length

        where = self.trajectories[start]
        if kind in ("Jump", "Miss id"):
            if start > 0:
                where = self.trajectories[start - 1 : start + length + 1, identity - 1]
            else:
                where = self.trajectories[start : start + length + 1, identity - 1]
        elif kind == "No id":
            for blob in self.list_of_blobs.blobs_in_video[start]:
                for blob_id, centroid in blob.final_ids_and_centroids:
                    if blob_id in (None, 0):
                        where = np.asarray(centroid)
                        break
        elif kind == "Dupl":
            duplicated_centroids = []
            for blob in self.list_of_blobs.blobs_in_video[start]:
                for blob_id, centroid in blob.final_ids_and_centroids:
                    if blob_id == identity:
                        duplicated_centroids.append(centroid)
            where = np.asarray(duplicated_centroids)
        else:
            raise ValueError(kind)
        logging.debug(f"Error selected {kind=}, {start=}, {length=}, {identity=}")
        self.go_to_error.emit(kind, start, length, where, identity)

    def set_references(
        self,
        traj: np.ndarray,
        unidentified: np.ndarray,
        duplicated: np.ndarray,
        blobs: ListOfBlobs,
        tracking_intervals: list[list[int]] | None,
        session: Session | None = None,
    ):
        self.session = session
        self.trajectories = traj
        self.unidentified = unidentified
        self.duplicated = duplicated
        self.list_of_blobs = blobs
        self.non_accepted_jumps = np.ones((traj.shape[0] - 1, traj.shape[1]), bool)
        if tracking_intervals is None:
            self.in_tracking_interval = np.ones(len(traj), bool)
        else:
            self.in_tracking_interval = np.zeros(len(traj), bool)
            for start, end in tracking_intervals:
                self.in_tracking_interval[start:end] = True

        self.update_list_of_errors()

    def accepted_interpolation(self):
        "Connected from Interpolator.interpolation_accepted"
        if not hasattr(self, "selected_error"):
            return
        kind, identity, start, length = self.selected_error
        start -= 1
        if kind == "Jump":
            self.non_accepted_jumps[start : start + length, identity - 1] = False

    def getErrors(self) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
        # TODO Add more errors (super-crossings)
        if np.isnan(self.trajectories).all():
            # not finished session
            return {}
        missing = (
            np.isnan(self.trajectories[..., 0]) & self.in_tracking_interval[:, None]
        )
        return {
            "Miss id": get_list_of_Trues_for_id(self.mask_absent_identities(missing)),
            "No id": [(-1,) + get_list_of_Trues(self.unidentified)],
            "Dupl": get_list_of_Trues_for_id(self.duplicated),
            "Jump": self.get_impossible_jumps(),
        }

    def mask_absent_identities(self, missing: np.ndarray) -> np.ndarray:
        """Clears "missing" frames in which an identity is not expected at all.

        An animal that enters the arena at frame 4000 has no trajectory before
        it, and flagging all 4000 of those frames as errors buries the real
        ones. The Validator lets the user declare the interval in which each
        identity is present, and those declarations are applied here.

        `missing` is `(n_frames, n_animals)`; it is modified in place and
        returned.
        """
        if self.session is None or not self.session.presence_intervals:
            return missing

        for identity_idx in range(missing.shape[1]):
            intervals = self.session.presence_intervals.get(str(identity_idx + 1))
            if not intervals:
                continue
            present = np.zeros(missing.shape[0], bool)
            for start, end in intervals:
                # Inclusive at both ends, as the user types them.
                present[start : end + 1] = True
            missing[:, identity_idx] &= present

        return missing

    def keep_error(self, identity: int, length: int) -> bool:
        """Whether an error survives the min-duration and pair filters."""
        if length < self.min_duration.value():
            return False
        if self.filter_pair_btn.isChecked():
            return identity in (self.id_1.value(), self.id_2.value())
        return True

    def select_next_error(self):
        """Selects the next row, wrapping round at the end of the table."""
        if self.table.rowCount() == 0:
            return
        next_row = (self.table.currentRow() + 1) % self.table.rowCount()
        self.table.selectRow(next_row)
        self.cell_clicked(next_row, 0)

    def update_list_of_errors(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for error_kind, errors_for_id in self.getErrors().items():
            for identity, starts, lengths in errors_for_id:
                for start, length in zip(starts, lengths):
                    if not self.keep_error(identity, length):
                        continue
                    self.table.insertRow(0)
                    self.table.setItem(0, 0, CustomTableWidgetItem(error_kind))
                    self.table.setItem(0, 1, CustomTableWidgetItem(identity))
                    self.table.setItem(0, 2, CustomTableWidgetItem(start))
                    self.table.setItem(0, 3, CustomTableWidgetItem(length))
                    self.table.setItem(0, FLAG_COLUMN, self.make_flag_item(error_kind))
        self.left_label.setText(f"List of errors ({self.table.rowCount()} errors)")
        self.flag_summary.setText("")
        self.table.setSortingEnabled(True)
        if self.autoselect_errors.isChecked():
            self.table.selectRow(0)

    @staticmethod
    def make_flag_item(error_kind: str) -> QTableWidgetItem:
        """The "Flag" cell of a row: a checkbox for Miss id, empty for the rest.

        Only a Miss id gap can be filled with a straight line, so only those
        rows get something to check.
        """
        item = QTableWidgetItem()
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if error_kind == "Miss id":
            flags |= Qt.ItemFlag.ItemIsUserCheckable
            item.setCheckState(Qt.CheckState.Unchecked)
        item.setFlags(flags)
        return item

    def miss_id_gap(self, row: int) -> tuple[int, int, int] | None:
        """`(identity, start, length)` of a row, or None if it is not a Miss id."""
        if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) != "Miss id":
            return None
        return (
            self.table.item(row, 1).data(Qt.ItemDataRole.UserRole),
            self.table.item(row, 2).data(Qt.ItemDataRole.UserRole),
            self.table.item(row, 3).data(Qt.ItemDataRole.UserRole),
        )

    def flag_unambiguous_gaps(self) -> None:
        """Checks the Flag box of every Miss id gap that no other animal comes near.

        Walking through hundreds of gaps one by one to confirm that nothing
        else was nearby is the slow part of closing them; this does that part.
        The judgment is made on the straight line across the gap, which is
        also what "Interpolate all flagged" will draw.
        """
        threshold = self.vicinity_threshold.value()
        n_flagged = n_gaps = 0
        for row in range(self.table.rowCount()):
            gap = self.miss_id_gap(row)
            if gap is None:
                continue
            n_gaps += 1
            identity, start, length = gap
            if gap_is_unambiguous(
                self.trajectories, identity - 1, start, length, threshold
            ):
                self.table.item(row, FLAG_COLUMN).setCheckState(Qt.CheckState.Checked)
                n_flagged += 1
        self.flag_summary.setText(
            f"Flagged {n_flagged} of {n_gaps} Miss id gaps"
            f" (no other animal within {threshold} px)"
        )
        logging.info(f"Flagged {n_flagged} unambiguous Miss id gaps ({threshold} px)")

    def flagged_gaps(self) -> list[tuple[int, int, int]]:
        """`(identity, start, length)` of every Miss id gap whose Flag is checked."""
        flagged = []
        for row in range(self.table.rowCount()):
            gap = self.miss_id_gap(row)
            if (
                gap is not None
                and self.table.item(row, FLAG_COLUMN).checkState()
                == Qt.CheckState.Checked
            ):
                flagged.append(gap)
        return flagged

    def interpolate_flagged(self) -> None:
        flagged = self.flagged_gaps()
        if not flagged:
            self.flag_summary.setText("No gap is flagged")
            return
        # A gap at the very start or end of the video has a position on one side
        # only, so there is no line to draw. That can only be flagged by hand.
        fillable = [
            gap
            for gap in flagged
            if find_gap_anchors(self.trajectories, gap[0] - 1, gap[1], gap[2])
        ]
        if not fillable:
            self.flag_summary.setText(
                "None of the flagged gaps has a position on both sides"
            )
            return
        self.interpolate_all_flagged.emit(fillable)
        # The signal rebuilds the list, which clears the summary, so set it last.
        summary = f"Interpolated {len(fillable)} flagged gaps (Ctrl+Z undoes them)"
        if len(fillable) < len(flagged):
            summary += (
                f"; skipped {len(flagged) - len(fillable)} with no position"
                " on both sides"
            )
        self.flag_summary.setText(summary)

    def get_impossible_jumps(self):
        th = self.jumps_th.value()
        if th == 0 or not self.jumps_th_label.isChecked():
            return []
        speed = np.sqrt((np.diff(self.trajectories, axis=0) ** 2).sum(-1))
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                mean, std = np.nanmean(speed), np.nanstd(speed)
            except RuntimeWarning:
                return []

        too_fast = (speed > (mean + th * float(std))) & self.non_accepted_jumps
        out = get_list_of_Trues_for_id(too_fast)
        for _id, start, _length in out:
            start += 1
        return out


def get_list_of_Trues(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the start and the length of every True cluster in an array

    Input: [0,1,0,1,1,0]
    Output [1,3], [1,2] (one cluster at 1 of length 1 and another at 3 of length 2)
    """
    where = arr.nonzero()[0]
    is_edge = np.diff(where, prepend=-np.inf, append=np.inf) > 1
    starts = where[is_edge[:-1]]
    ends = where[is_edge[1:]]
    return starts, ends - starts + 1


def get_list_of_Trues_for_id(
    data: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    return [
        (fish_id + 1,) + get_list_of_Trues(data[:, fish_id])
        for fish_id in range(data.shape[1])
    ]
