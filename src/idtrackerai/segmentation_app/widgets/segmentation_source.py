"""Chooses where the animals' outlines come from: idtracker.ai's own
thresholding, or contours from an external segmentation model -- loaded from a
file, prepared with the Detectron2 pipeline, or produced here by SAM 3 or by
a YOLO segmentation model.
"""

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from idtrackerai.base.animals_detection.external_contours import ExternalContours
from idtrackerai.GUI_tools import WrappedLabel

THRESHOLDING = "Thresholding"
EXTERNAL = "External contours"
DETECTRON2 = "Detectron2 pipeline"
SAM3 = "SAM 3"
YOLO = "YOLO"
# the modes that make contours rather than track from them
MODEL_MODES = (DETECTRON2, SAM3, YOLO)


class SegmentationSourceWidget(QWidget):
    """Switches between thresholding and an external contour file.

    Emits the chosen file's path, or None for thresholding. The file is checked
    against the loaded video as soon as either changes, so a mismatched sidecar
    is caught here rather than after a long tracking run.
    """

    valueChanged = Signal(object)  # Path | None
    modeChanged = Signal(str)      # one of the mode constants above

    def __init__(self):
        super().__init__()
        self.paths: list[Path] = []
        self.n_frames: int | None = None
        self.video_size: tuple[int, int] | None = None
        self.video_paths: list[Path] = []

        self.label = QLabel("Segmentation")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source = QComboBox()
        self.source.addItems((THRESHOLDING, EXTERNAL, DETECTRON2, SAM3, YOLO))
        self.source.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.source.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )

        self.file_label = WrappedLabel(framed=True)
        self.file_label.setVisible(False)
        self.file_label.clicked.connect(self.choose_file)

        self.browse = QToolButton()
        self.browse.setText("Load contours")
        self.browse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.browse.setVisible(False)
        self.browse.clicked.connect(self.choose_file)

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label)
        layout.addWidget(self.source)
        layout.addWidget(self.file_label)
        layout.addWidget(self.browse)

        self.source.currentTextChanged.connect(self.source_changed)

    # --------------------------------------------------------------- the path
    @property
    def path(self) -> Path | None:
        """The single contour file, when there is exactly one.

        Kept because most sessions are one clip and the rest of the app reads
        it; :meth:`value` is what handles several.
        """
        return self.paths[0] if len(self.paths) == 1 else None

    # ------------------------------------------------------------- video info
    def set_video_info(
        self,
        n_frames: int,
        video_size: tuple[int, int],
        video_paths: list | None = None,
    ) -> None:
        """Called when a video is loaded, so the sidecars can be checked."""
        self.n_frames = n_frames
        self.video_size = video_size
        if video_paths is not None:
            new_paths = [Path(p) for p in video_paths]
            if self.paths and new_paths != self.video_paths:
                # the contours belong to the previous video. Dropping them here
                # means a .toml for another video is not first checked against
                # contours that were never meant for it; setValue restores its own.
                self.clear_to_thresholding()
            self.video_paths = new_paths
        if self.paths:
            problem = self.describe_or_reject(self.paths)
            if problem:
                QMessageBox.warning(self, "External contours do not fit", problem)
                self.clear_to_thresholding()

    # ------------------------------------------------------------- user input
    def source_changed(self, text: str) -> None:
        external = text == EXTERNAL
        self.file_label.setVisible(external and self.path is not None)
        self.browse.setVisible(external)
        self.modeChanged.emit(text)

        if text in MODEL_MODES:
            # Preparing or running a model, not tracking with one yet. There is
            # no contour file at this point, so this must not open a file
            # dialog the way EXTERNAL does; the panel below takes over. When
            # SAM 3 or YOLO finishes an export it hands the files back through
            # accept_paths, which switches the mode itself.
            self.valueChanged.emit(None)
            return

        if not external:
            self.valueChanged.emit(None)
            return

        if not self.paths and not self.choose_file():
            # nothing loaded and the dialog was dismissed, so stay on the
            # thresholding path rather than leaving the app in a state it
            # cannot segment in
            self.clear_to_thresholding()
            return

        self.valueChanged.emit(self.value())

    def choose_file(self) -> bool:
        """One file for one clip; a folder, matched by name, for several.

        A session made of several clips needs one contour file per clip, in
        the session's order. Asking for them one at a time invites a wrong
        order, which sums to the right total and would silently apply each
        clip's contours to another clip's frames. Matching by file name makes
        the order right by construction.
        """
        if len(self.video_paths) > 1:
            return self._choose_folder()

        start = str(self.paths[0].parent if self.paths else Path.cwd())
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open external contours", start, filter="Contours (*.h5 *.hdf5)"
        )
        if not file_name:
            return False
        return self._accept([Path(file_name)])

    def _choose_folder(self) -> bool:
        start = str(self.paths[0].parent if self.paths else Path.cwd())
        folder = QFileDialog.getExistingDirectory(
            self, "Folder holding one contour file per clip", start
        )
        if not folder:
            return False

        folder = Path(folder)
        matched, missing = self.match_by_name(self.video_paths, folder)
        if missing:
            QMessageBox.warning(
                self,
                "No contour file for every clip",
                f"{len(missing)} of the {len(self.video_paths)} clips in this "
                f"session have no matching file in {folder}:\n\n"
                + "\n".join(f"{p.name}  ->  {p.stem}.h5" for p in missing[:10])
                + ("\n..." if len(missing) > 10 else "")
                + "\n\nThe exporter writes one file per clip, named after it. "
                "Export the missing clips, or track them separately.",
            )
            return False
        return self._accept(matched)

    @staticmethod
    def match_by_name(video_paths, folder: Path) -> tuple[list[Path], list[Path]]:
        """Pairs each video with <its stem>.h5 in a folder. (matched, missing)"""
        matched: list[Path] = []
        missing: list[Path] = []
        for video in video_paths:
            video = Path(video)
            for suffix in (".h5", ".hdf5"):
                candidate = folder / f"{video.stem}{suffix}"
                if candidate.is_file():
                    matched.append(candidate)
                    break
            else:
                missing.append(video)
        return matched, missing

    def accept_paths(self, paths) -> bool:
        """Takes contour files produced in-app, as if they had been browsed to.

        The SAM 3 panel calls this when an export finishes: the files go
        through the same validation as a hand-picked one, and the mode switches
        to External contours, because from here on there is no difference
        between contours this app made and contours it was given.
        """
        return self._accept([Path(p) for p in paths])

    def _accept(self, paths: list[Path]) -> bool:
        problem = self.describe_or_reject(paths)
        if problem:
            QMessageBox.warning(self, "Could not use these contour files", problem)
            return False

        self.paths = paths
        self._show_paths()
        if self.source.currentText() != EXTERNAL:
            self.source.setCurrentText(EXTERNAL)  # re-enters source_changed
        else:
            self.valueChanged.emit(self.value())
        return True

    def _show_paths(self) -> None:
        if len(self.paths) == 1:
            self.file_label.setText(self.paths[0].name)
        else:
            self.file_label.setText(
                f"{len(self.paths)} files in {self.paths[0].parent.name}"
            )
        self.file_label.setVisible(True)

    def clear_to_thresholding(self) -> None:
        self.paths = []
        self.file_label.setVisible(False)
        self.browse.setVisible(False)
        # blockSignals so we emit once, below, instead of twice
        self.source.blockSignals(True)
        self.source.setCurrentText(THRESHOLDING)
        self.source.blockSignals(False)
        # the blocked signal is what tells the window to re-enable the
        # thresholds and hide the Detectron2 panel, so say it explicitly
        self.modeChanged.emit(THRESHOLDING)
        self.valueChanged.emit(None)

    # ------------------------------------------------------------ validation
    def describe_or_reject(self, path_or_paths) -> str | None:
        """Returns a human-readable problem, or None if the files are usable.

        Also updates the tooltip with what the files say about themselves, so
        the user can confirm they picked the right model's output.
        """
        paths = (
            [Path(path_or_paths)]
            if isinstance(path_or_paths, (str, Path))
            else [Path(p) for p in path_or_paths]
        )
        if not paths:
            return "No contour files were given."

        readers = []
        try:
            for path in paths:
                try:
                    readers.append(ExternalContours(path))
                except FileNotFoundError:
                    return f"{path} does not exist."
                except Exception as exc:  # noqa: BLE001
                    return f"{path.name} could not be read as a contour file.\n\n{exc}"

            total = sum(reader.n_frames for reader in readers)
            one = len(readers) == 1
            if self.n_frames is not None and total != self.n_frames:
                what = (
                    f"{paths[0].name} covers {total} frames"
                    if one
                    else f"These {len(paths)} files cover {total} frames between them"
                )
                listing = (
                    ""
                    if one
                    else "\n\n"
                    + "\n".join(f"{r.path.name}: {r.n_frames}" for r in readers[:10])
                )
                return (
                    f"{what} but the video has {self.n_frames}.\n\nThese contours"
                    " were probably computed from different clips." + listing
                )

            sizes = {(r.width, r.height) for r in readers}
            if len(sizes) > 1:
                detail = ", ".join(f"{r.path.name} {r.width}x{r.height}" for r in readers)
                return f"These files were computed at different sizes: {detail}."
            if self.video_size is not None and sizes and sizes != {self.video_size}:
                got_w, got_h = next(iter(sizes))
                subject = paths[0].name if one else "These files were"
                return (
                    f"{subject} computed at {got_w}x{got_h} but the video is "
                    f"{self.video_size[0]}x{self.video_size[1]}."
                )

            described = "\n".join(
                f"{key}: {value}"
                for key, value in readers[0].attrs.items()
                if key not in ("format_version",)
            )
            if not one:
                described = (
                    "\n".join(f"{r.path.name} ({r.n_frames} frames)" for r in readers)
                    + ("\n\n" + described if described else "")
                )
            self.file_label.setToolTip(described)
            logging.info(
                "External contours selected: %s",
                readers[0] if one else f"{len(readers)} files, {total} frames",
            )
            return None
        finally:
            for reader in readers:
                reader.close()

    # ------------------------------------------------------------- app access
    def setValue(self, path_or_paths) -> None:
        """Restores the widget from a session or .toml file."""
        if path_or_paths is None or (
            not isinstance(path_or_paths, (str, Path)) and not list(path_or_paths)
        ):
            self.clear_to_thresholding()
            return

        paths = (
            [Path(path_or_paths)]
            if isinstance(path_or_paths, (str, Path))
            else [Path(p) for p in path_or_paths]
        )
        problem = self.describe_or_reject(paths)
        if problem:
            QMessageBox.warning(self, "External contours unavailable", problem)
            self.clear_to_thresholding()
            return

        self.paths = paths
        self._show_paths()
        self.browse.setVisible(True)
        self.source.setCurrentText(EXTERNAL)
        self.valueChanged.emit(self.value())

    def value(self):
        """The contour file(s) tracking will use, or None.

        A single clip gives a plain path, so sessions saved before several
        were supported round-trip unchanged; several give a list, in the
        session's video order.

        The model modes deliberately return None: they are preparation
        modes, and nothing about them belongs in the saved parameters.
        """
        if self.source.currentText() != EXTERNAL or not self.paths:
            return None
        return self.paths[0] if len(self.paths) == 1 else list(self.paths)

    def mode(self) -> str:
        return self.source.currentText()

    def uses_external_contours(self) -> bool:
        return self.value() is not None
