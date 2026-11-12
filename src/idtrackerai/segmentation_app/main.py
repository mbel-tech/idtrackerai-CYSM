# Each Qt binding is different, so...
# pyright: reportIncompatibleMethodOverride=false
import importlib.util
import json
import logging
import numbers
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import h5py
import toml
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Session
from idtrackerai.GUI_tools import GUIBase, QHLine, VideoPlayer
from idtrackerai.utils import pprint_dict

from .widgets import (
    AreaThresholds,
    BkgWidget,
    BlobInfoWidget,
    Detectron2Panel,
    EnhancementPreview,
    EnhancementWidget,
    FrameAnalyzer,
    IntensityThresholds,
    OpenVideoWidget,
    ROIWidget,
    Sam3Panel,
    YoloPanel,
    SegmentationSourceWidget,
    TrackingIntervalsWidget,
)
from .widgets.segmentation_source import (
    DETECTRON2,
    EXTERNAL,
    SAM3,
    THRESHOLDING,
    YOLO,
)


class SegmentationGUI(GUIBase):
    """The main class implementing the Segmentation app.
    From here, all widgets are coordinated and connected.

    Parameters
    ----------
    GUIBase : QMainWindow
        Base configuration for all idtracker.ai GUIs
    """

    run_idtrackerai_after_closing: bool = False

    def __init__(self, session: Session | None = None):
        super().__init__()

        self.setWindowTitle("Segmentation App")
        self.session = session or Session()
        self.documentation_url = (
            "https://idtracker.ai/latest/user_guide/segmentation_app.html"
        )

        self.open_widget = OpenVideoWidget(self, self.session.frames_per_episode)
        self.videoPlayer = VideoPlayer(self)
        self.frame_analyzer = FrameAnalyzer()
        self.blobInfo = BlobInfoWidget()
        self.bkg_widget = BkgWidget()
        self.ROI_Widget = ROIWidget(self)
        self.tracking_interval = TrackingIntervalsWidget(self)
        self.widgets_to_close.append(self.videoPlayer)

        self.segmentation_source = SegmentationSourceWidget()
        self.enhancement = EnhancementWidget()
        self.enhancement_preview = EnhancementPreview()
        self.detectron2_panel = Detectron2Panel(self.enhancement)
        self.sam3_panel = Sam3Panel(self.enhancement)
        self.yolo_panel = YoloPanel(self.enhancement)
        # the preview is live in every mode, because the enhancement is applied
        # to segmentation in every mode
        self.enhancement_preview.set_enabled(True)
        self.detectron2_panel.setVisible(False)
        self.sam3_panel.setVisible(False)
        self.yolo_panel.setVisible(False)
        # so GUIBase.closeEvent gives it a chance to stop threads and save
        self.widgets_to_close.append(self.detectron2_panel)
        self.widgets_to_close.append(self.sam3_panel)
        self.widgets_to_close.append(self.yolo_panel)
        # an export hands its contour files to the source widget, which
        # validates them against the video before the session adopts them
        self.sam3_panel.contoursReady.connect(self.sam3_contours_ready)
        self.yolo_panel.contoursReady.connect(self.yolo_contours_ready)
        self.intensity_thresholds = IntensityThresholds(self, min=0, max=255)
        self.area_thresholds = AreaThresholds()

        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save_parameters.setShortcut("Ctrl+S")

        self.check_segm = QCheckBox("Stop tracking if #blobs > #animals")
        self.track_wo_id = QCheckBox("Track without identities")

        n_animals_row = QHBoxLayout()
        n_animals_label = QLabel("Number of animals")
        n_animals_row.addWidget(n_animals_label)
        self.n_animals = QSpinBox()
        self.n_animals.setMaximum(1000)
        self.n_animals.setMinimum(0)
        self.n_animals.setSpecialValueText("unknown")
        n_animals_row.addWidget(self.n_animals)
        n_animals_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        session_row = QHBoxLayout()
        session_label = QLabel("Session name")
        session_row.addWidget(session_label)
        self.session_name = SessionName()
        session_row.addWidget(self.session_name)
        session_row.addWidget(self.save_parameters)

        self.close_and_track_btn = QPushButton("Close window and track video")
        self.close_and_track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Connecting widgets
        self.open_widget.path_clicked.connect(self.videoPlayer.setCurrentFrame)
        self.open_widget.new_video_paths.connect(self.new_video_paths)
        self.open_widget.new_episodes.connect(self.bkg_widget.set_new_video_paths)
        self.open_widget.new_parameters.connect(self.new_parameters)
        self.open_widget.video_paths_reordered.connect(
            self.videoPlayer.reorder_video_paths
        )
        self.open_widget.video_paths_reordered.connect(
            lambda _paths: self.enhancement_preview.bump_generation()
        )
        self.open_widget.video_paths_reordered.connect(
            lambda paths: self.session_name.setPlaceholderText(
                "&".join(Path(path).stem for path in paths)
            )
        )
        self.n_animals.editingFinished.connect(self.n_animals.clearFocus)
        self.n_animals.valueChanged.connect(self.detectron2_panel.set_number_of_animals)
        self.n_animals.valueChanged.connect(self.sam3_panel.set_number_of_animals)
        self.sam3_panel.tracking_intervals = lambda: self.tracking_interval.value()
        # each model panel can write the session as a Colab notebook
        self.detectron2_panel.session_parameters = self.notebook_parameters
        self.sam3_panel.session_parameters = self.notebook_parameters
        self.yolo_panel.session_parameters = self.notebook_parameters
        self.n_animals.valueChanged.connect(self.yolo_panel.set_number_of_animals)
        self.yolo_panel.tracking_intervals = lambda: self.tracking_interval.value()
        self.n_animals.valueChanged.connect(self.blobInfo.setNAnimals)
        self.tracking_interval.range_slider.single_value_changed.connect(
            self.videoPlayer.setCurrentFrame
        )
        self.tracking_interval.newValue.connect(self.open_widget.set_tracking_interval)
        self.tracking_interval.newValue.connect(self.blobInfo.setTrackingIntervals)
        self.intensity_thresholds.newValue.connect(
            self.frame_analyzer.set_intensity_ths
        )
        self.session_name.editingFinished.connect(self.session_name.clearFocus)
        self.save_parameters.clicked.connect(self.save_parameters_func)
        self.area_thresholds.valueChanged.connect(self.frame_analyzer.set_area_ths)
        self.close_and_track_btn.clicked.connect(self.close_and_track_video)
        self.ROI_Widget.valueChanged.connect(self.frame_analyzer.set_ROI_mask)
        self.ROI_Widget.needToDraw.connect(self.videoPlayer.update)
        self.segmentation_source.valueChanged.connect(
            self.segmentation_source_changed
        )
        self.bkg_widget.new_bkg_data.connect(self.frame_analyzer.set_bkg)
        self.bkg_widget.new_bkg_data.connect(self.intensity_thresholds.bkg_changed)
        self.frame_analyzer.new_areas.connect(self.blobInfo.setAreas)
        self.frame_analyzer.new_parameters.connect(self.videoPlayer.update)
        self.segmentation_source.modeChanged.connect(self.segmentation_mode_changed)
        self.enhancement.settingsChanged.connect(self.enhancement_preview.set_settings)
        self.enhancement.settingsChanged.connect(self.frame_analyzer.set_enhancement)
        self.enhancement.settingsChanged.connect(lambda _s: self.update_enhancement_hint())
        self.enhancement.settingsCommitted.connect(self.enhancement_committed)
        self.enhancement.splitChanged.connect(self.enhancement_preview.set_split)
        self.enhancement.splitChanged.connect(lambda _v: self.videoPlayer.update())
        self.enhancement_preview.failed.connect(self.enhancement.show_error)
        # connected before the two below, so the enhanced frame is painted
        # under the blob polygons and the ROI outlines rather than over them
        self.videoPlayer.painting_time.connect(
            self.enhancement_preview.paint_on_canvas
        )
        self.videoPlayer.painting_time.connect(self.frame_analyzer.paint_on_canvas)
        self.videoPlayer.painting_time.connect(self.ROI_Widget.paint_on_canvas)
        self.videoPlayer.canvas.click_event.connect(self.ROI_Widget.click_event)

        # Tooltips texts
        tooltips = toml.load(Path(__file__).parent / "tooltips.toml")

        self.open_widget.button_open.setToolTip(tooltips["open_btn"])
        self.open_widget.single_file_label.setToolTip(tooltips["open_path_label"])
        self.open_widget.list_of_files.setToolTip(tooltips["open_path_list"])
        self.tracking_interval.setToolTip(tooltips["tracking_interval"])
        self.ROI_Widget.setToolTip(tooltips["region_of_interest"])
        self.ROI_Widget.exclusive_rois.setToolTip(tooltips["exclusive_rois"])
        self.segmentation_source.setToolTip(tooltips["segmentation_source"])
        self.segmentation_source.browse.setToolTip(tooltips["external_contours"])
        self.enhancement.setToolTips(tooltips)
        self.detectron2_panel.setToolTips(tooltips)
        self.sam3_panel.setToolTips(tooltips)
        self.yolo_panel.setToolTips(tooltips)
        self.bkg_widget.setToolTip(tooltips["background_subtraction"])
        self.bkg_widget.bkg_stat.setToolTip(tooltips["background_stat"])
        self.bkg_widget.view_bkg.setToolTip(tooltips["background_view"])
        self.n_animals.setToolTip(tooltips["number_of_animals"])
        n_animals_label.setToolTip(tooltips["number_of_animals"])
        self.check_segm.setToolTip(tooltips["check_segm"])
        self.area_thresholds.setToolTip(tooltips["area_thresholds"])
        self.track_wo_id.setToolTip(tooltips["track_wo_id"])
        self.save_parameters.setToolTip(tooltips["save_params"])
        self.close_and_track_btn.setToolTip(tooltips["close_and_track"])
        self.blobInfo.setToolTip(tooltips["blobs_info"])
        self.session_name.setToolTip(tooltips["session_name"])
        session_label.setToolTip(tooltips["session_name"])
        self.intensity_thresholds.setToolTips(
            tooltips["intensity_thresholds_nobkg"],
            tooltips["intensity_thresholds_yesbkg"],
        )

        left_layout = QVBoxLayout()
        self.open_widget.layout().setContentsMargins(0, 8, 0, 0)
        left_layout.addWidget(self.open_widget)
        for widget in (
            QHLine(),
            self.tracking_interval,
            self.ROI_Widget,
            QHLine(),
            n_animals_row,
            self.enhancement,
            self.segmentation_source,
            self.detectron2_panel,
            self.sam3_panel,
            self.yolo_panel,
            self.bkg_widget,
            self.intensity_thresholds,
            self.area_thresholds,
            self.blobInfo,
            QHLine(),
        ):
            if isinstance(widget, (QVBoxLayout, QHBoxLayout)):
                widget.setContentsMargins(0, 0, 0, 0)
                superwidget = QWidget()
                superwidget.setLayout(widget)
                left_layout.addWidget(
                    superwidget, alignment=Qt.AlignmentFlag.AlignVCenter
                )
            else:
                lay = widget.layout()
                if lay is not None:
                    lay.setContentsMargins(0, 0, 0, 0)
                left_layout.addWidget(
                    widget,
                    alignment=Qt.AlignmentFlag.AlignVCenter,
                    stretch=1 if isinstance(widget, QHLine) else 3,
                )
        left_layout.addWidget(self.check_segm)
        left_layout.addWidget(self.track_wo_id)
        left_layout.addLayout(session_row)
        left_layout.addWidget(self.close_and_track_btn)

        self.videoPlayer.layout().setContentsMargins(8, 0, 0, 0)

        left = QWidget()
        left.setLayout(left_layout)

        # The column asks for more height than a laptop screen has, and a
        # QVBoxLayout answers that by compressing every child below its
        # natural size: the Detectron2 panel was getting 248px for content
        # needing 440, which pushed a step's own button behind a scrollbar
        # inside the step. Scrolling the column instead lets each widget be
        # the size it asked for, in every mode.
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(self.videoPlayer)
        main_splitter.setSizes([400, 600])
        self.centralWidget().layout().addWidget(main_splitter)
        self.list_of_widgets = self.get_list_of_widgets(left_layout)
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.videoPlayer.setEnabled(False)
        self.enabled = False
        self.open_widget.setEnabled(True)

        self.setTabOrder(self.n_animals, self.videoPlayer.canvas)
        self.setTabOrder(self.videoPlayer.canvas, self.n_animals)
        for widget in self.findChildren(QCheckBox):
            assert isinstance(widget, QWidget)
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        QTimer.singleShot(0, self.load_parameters)

    def update_enhancement_hint(self) -> None:
        """Warn that enhancement moves the ground the thresholds stand on.

        Enhancement re-centres the image on mid-grey with a different spread,
        so an intensity threshold chosen for the raw frames no longer means
        what it did and will usually detect far too much. The blob count makes
        that visible, but only if you know to look.
        """
        enhancing = self.enhancement.settings().get("enhance", False)
        thresholding = self.segmentation_source.mode() == THRESHOLDING
        if enhancing and thresholding:
            self.enhancement.set_hint(
                "Enhancement changes the brightness range, so the blob "
                "intensity thresholds below need re-tuning. Watch the blob "
                "count while you adjust them."
            )
        else:
            self.enhancement.set_hint("")

    def enhancement_committed(self, settings: dict) -> None:
        """Carry the settings to everything that derives from the pixels.

        A background model computed from differently enhanced frames is stale,
        so it is discarded rather than compared against the new ones.
        """
        self.bkg_widget.enhancement_changed(settings)
        self.detectron2_panel.enhancement_committed(settings)
        self.sam3_panel.enhancement_committed(settings)
        self.yolo_panel.enhancement_committed(settings)

    def yolo_contours_ready(self, paths) -> None:
        """Adopts contours the YOLO panel exported or imported; as for SAM 3,
        success is announced only once the source widget has accepted them."""
        if self.segmentation_source.accept_paths(paths):
            QMessageBox.information(
                self,
                "Tracking from the YOLO contours",
                f"This session now segments from the {len(paths)} contour "
                "file(s) written from YOLO. The intensity thresholds and "
                "background model take no further part.",
            )

    def sam3_contours_ready(self, paths) -> None:
        """Adopts contours SAM 3 just exported, and says whether it worked.

        accept_paths validates them against the loaded video and refuses a set
        that does not fit -- which is what a cancelled batch leaves behind, and
        what a clip exported at the wrong resolution would produce. It warns on
        its own when it refuses, so the only thing left to say here is the good
        news, and only when it is true. Announcing success from the panel
        instead would claim the session had switched over in exactly the cases
        where it had not.
        """
        if self.segmentation_source.accept_paths(paths):
            QMessageBox.information(
                self,
                "Tracking from the exported contours",
                f"This session now segments from the {len(paths)} contour "
                "file(s) SAM 3 wrote. The intensity thresholds and background "
                "model take no further part.",
            )

    def segmentation_source_changed(self, path: Path | None) -> None:
        """Keeps the preview in step with the chosen contour file."""
        self.frame_analyzer.set_external_contours(path)
        self.videoPlayer.update()

    def segmentation_mode_changed(self, mode: str) -> None:
        """Shows only what the chosen mode actually uses.

        With external contours neither the intensity thresholds nor the
        background model take part in segmentation, so they are greyed out:
        leaving them live would suggest they still do something.

        The Detectron2 and SAM 3 modes go further and hide them, because their
        panels need the vertical space and because neither segments by
        threshold at all. Both are modes for producing contours rather than
        tracking from them, so the live preview is suspended too: it has
        nothing to draw until a model has run.
        """
        external = mode == EXTERNAL
        detectron2 = mode == DETECTRON2
        sam3 = mode == SAM3
        yolo = mode == YOLO
        with_a_model = detectron2 or sam3 or yolo

        self.detectron2_panel.setVisible(detectron2)
        self.sam3_panel.setVisible(sam3)
        self.yolo_panel.setVisible(yolo)
        self.frame_analyzer.set_suspended(with_a_model)

        for widget in (self.intensity_thresholds, self.bkg_widget):
            widget.setVisible(not with_a_model)
            widget.setEnabled(not external and not with_a_model)
        for widget in (self.area_thresholds, self.blobInfo):
            widget.setVisible(not with_a_model)

        self.update_enhancement_visibility()
        self.update_enhancement_hint()
        self.videoPlayer.update()

    def update_enhancement_visibility(self) -> None:
        """Offers the frame enhancement only where it does something.

        Thresholding finds animals by contrast, which is what enhancement
        changes, and a Detectron2 model is trained on enhanced frames (step 1
        of its panel). SAM 3 and YOLO were trained on frames as recorded, and
        contour files are already made, so there the panel is left out. It is
        still shown when enhancement is switched on, because then it is being
        applied and must stay in view.
        """
        mode = self.segmentation_source.mode()
        enhancing = bool(self.enhancement.settings().get("enhance", False))
        self.enhancement.setVisible(mode in (THRESHOLDING, DETECTRON2) or enhancing)

    def manageDropedPaths(self, paths: Sequence[str]) -> None:
        self.open_widget.process_paths(paths)

    def load_parameters(self):
        """Sets all widgets to the values indicated by self.session"""
        self.open_widget.open_video_paths(self.session.video_paths)
        self.tracking_interval.setValue(self.session.tracking_intervals)
        self.ROI_Widget.setValue(self.session.roi_list, self.session.exclusive_rois)
        self.enhancement.setSettings(self.session.enhancement)
        self.detectron2_panel.set_model_spec(self.session.segmentation_model)
        self.sam3_panel.set_model_spec(self.session.segmentation_model)
        self.yolo_panel.set_model_spec(self.session.segmentation_model)
        contours = self.contours_to_show()
        self.segmentation_source.setValue(contours)
        backend = (self.session.segmentation_model or {}).get("backend", "detectron2")
        if contours is None and self.session.segmentation_model:
            # a .toml naming a model whose contours are not exported yet opens
            # on the panel that exports them, not on thresholding
            self.segmentation_source.source.setCurrentText(
                {"sam3": SAM3, "yolo": YOLO}.get(backend, DETECTRON2)
            )
        self.update_enhancement_visibility()
        self.intensity_thresholds.setValue(self.session.intensity_ths)
        self.area_thresholds.setValue(self.session.area_ths)
        self.n_animals.setValue(self.session.number_of_animals)
        self.track_wo_id.setChecked(self.session.track_wo_identities)
        self.check_segm.setChecked(self.session.check_segmentation)
        self.session_name.setText(self.session.name)
        self.bkg_widget.set_bkg_stat(self.session.background_subtraction_stat)
        self.bkg_widget.bkg_thread.n_frames_for_background = (
            self.session.number_of_frames_for_background
        )
        self.bkg_widget.checkBox.setChecked(self.session.use_bkg)
        self.videoPlayer.update()

    def contours_to_show(self):
        """The session's contour files, including those a [segmentation_model]
        table places in its contours_dir, once they exist. Shown so that a
        .toml written for the model route opens on that route."""
        if self.session.external_contours is not None:
            return self.session.external_contours
        if not self.session.segmentation_model:
            return None
        from idtrackerai.extra_tools.detectron2_pipeline import (
            segmentation_model as sm,
        )

        try:
            spec = sm.normalize_model(self.session.segmentation_model)
            paths = sm.default_contour_paths(spec, self.session.video_paths)  # type: ignore[arg-type]
        except sm.SegmentationModelError as exc:
            logging.warning(str(exc))
            return None
        if not all(path.is_file() for path in paths):
            logging.info(
                "The [segmentation_model] contours are not exported yet; showing"
                " thresholding until they are"
            )
            return None
        return [str(path) for path in paths]

    def new_parameters(self, params: dict):
        "Receives new toml parameters from OpenVideoWidget"
        unrecognized_params = self.session.set_parameters(**params, reset=True)
        if unrecognized_params:
            QMessageBox.warning(
                self,
                "Error loading parameters",
                f"Unrecognized parameters in toml file:\n\n{unrecognized_params}",
            )
        self.load_parameters()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.ROI_Widget.add.setChecked(False)

    def close_and_track_video(self):
        """Action when clicked "close and track video".
        It gathers widgets parameters, writes them in self.user_params and exits"""
        if importlib.util.find_spec("torch") is None:
            # tracking starts after this window closes, and it needs PyTorch.
            # Without it the app would just vanish, with the reason only in the log
            QMessageBox.critical(
                self,
                "PyTorch is not installed",
                "Tracking needs PyTorch, which is missing from this environment."
                " Install it, then try again:\n\n"
                f'  "{sys.executable}" -m pip install torch torchvision\n\n'
                "(add  --index-url https://download.pytorch.org/whl/cpu  for a"
                " CPU-only build). Your settings are still here: use 'Save"
                " parameters' to keep them.",
            )
            return

        parameters = self.out_parameters()
        if self.unacceptable_parameters(parameters):
            return

        logging.info(pprint_dict(parameters, "GUI params"), extra={"markup": True})
        self.session.set_parameters(**parameters)
        # set_parameters only writes the keys it is given, and out_parameters
        # leaves this one out when thresholding, so clear it explicitly.
        # Otherwise switching back to thresholding would silently keep tracking
        # from a contour file loaded earlier.
        self.session.external_contours = self.segmentation_source.value()
        # out_parameters leaves it out unless a model route was chosen, so a
        # table loaded from the .toml cannot bring contours back after the
        # user switched to thresholding here
        self.session.segmentation_model = parameters.get("segmentation_model")
        # same reason: out_parameters leaves enhancement out when it is off, so
        # a value loaded from a .toml would otherwise outlive choosing "None"
        self.session.enhancement = parameters.get("enhancement")
        bkg = self.bkg_widget.getBkg()
        if bkg is not None:
            tmp_bkg_path = Path(self.session.video_paths[0]).with_suffix(
                ".tmp_background.png"
            )
            cv2.imencode(".png", bkg)[1].tofile(str(tmp_bkg_path))
            self.session.background_from_segmentation_gui = tmp_bkg_path
        self.run_idtrackerai_after_closing = True
        self.close()

    def getSessionName(self) -> str:
        return (
            self.session_name.text() or self.session_name.placeholderText() or "no_name"
        ).strip()

    def out_parameters(self) -> dict:
        """Generates dict of all widgets content

        Returns
        -------
        dict
            Parameter dict containing all widgets content
        """
        out: dict[str, Any] = {
            "video_paths": self.open_widget.getVideoPaths(),
            "intensity_ths": self.intensity_thresholds.value(),
            "area_ths": self.area_thresholds.value(),
            "tracking_intervals": self.tracking_interval.value(),
            "number_of_animals": self.n_animals.value(),
            "use_bkg": self.bkg_widget.checkBox.isChecked(),
            "check_segmentation": self.check_segm.isChecked(),
            "track_wo_identities": self.track_wo_id.isChecked(),
            "roi_list": self.ROI_Widget.getValue(),
        }

        settings = self.enhancement.settings()
        if settings.get("enhance"):
            out["enhancement"] = settings

        external_contours = self.segmentation_source.value()
        if external_contours is not None:
            # a list for a multi-clip session, one path per clip in order
            out["external_contours"] = (
                [str(path) for path in external_contours]
                if isinstance(external_contours, list)
                else str(external_contours)
            )

        # The model the contours come from, whether they exist yet or not:
        # in the Detectron2 mode the weights chosen in its export step, and
        # with contour files loaded only weights chosen explicitly, since
        # those files may come from SAM 3 or another model entirely.
        panel = self.model_panel_for(self.segmentation_source.mode(), external_contours)
        if panel is not None:
            spec = panel.model_spec()
            if spec is not None:
                out["segmentation_model"] = spec

        if self.session_name.text():  # put the name at the first position
            out = {"name": self.session_name.text()} | out

        if self.bkg_widget.checkBox.isChecked():
            out["background_subtraction_stat"] = self.bkg_widget.get_bkg_stat()

        if (
            self.ROI_Widget.exclusive_rois.isVisible()
            and self.ROI_Widget.exclusive_rois.isEnabled()
        ):
            out["exclusive_rois"] = self.ROI_Widget.exclusive_rois.isChecked()
        return out

    def notebook_parameters(self) -> dict | None:
        """The session as it would be saved, always with a name, and with
        the folder a notebook is best saved in. None before a video is open."""
        if not hasattr(self.open_widget, "single_file"):  # set by the first video
            return None
        return (
            {"name": self.getSessionName()}
            | self.out_parameters()
            | {"output_dir": str(self.session.output_dir or "")}
        )

    def model_panel_for(self, mode: str, external_contours):
        """The panel whose model the session's contours come from, or None.

        In the Detectron2 and SAM 3 modes, that mode's panel. With contour
        files loaded, the panel of the backend the files record making them,
        and only if a model was chosen there explicitly: files from elsewhere
        must not be labelled with whatever a panel happens to hold.
        """
        if mode == DETECTRON2:
            return self.detectron2_panel
        if mode == SAM3:
            return self.sam3_panel
        if mode == YOLO:
            return self.yolo_panel
        if mode != EXTERNAL or external_contours is None:
            return None
        first = (
            external_contours[0]
            if isinstance(external_contours, list)
            else external_contours
        )
        try:
            with h5py.File(first, "r") as file:
                backend = file.attrs.get("backend", "detectron2")
        except OSError:
            return None
        backend = backend.decode() if isinstance(backend, bytes) else str(backend)
        panel = {"sam3": self.sam3_panel, "yolo": self.yolo_panel}.get(
            backend, self.detectron2_panel
        )
        return panel if panel.has_chosen_weights() else None

    def unacceptable_parameters(self, parameters: dict) -> bool:
        if (
            parameters["number_of_animals"] == 0
            and not parameters["track_wo_identities"]
        ):
            QMessageBox.warning(
                self,
                "Missing parameters",
                'Please, define the number of animals in the video or check "Track'
                ' without identities".\n\nEven if tracking without identities,'
                " adding the number of animals is recommended to improve the"
                " individual/crossing blob detection.",
            )
            return True
        if parameters["roi_list"] is not None and len(parameters["roi_list"]) == 0:
            QMessageBox.warning(
                self,
                "Missing parameters",
                "Please, add a region of interest or uncheck the Regions of"
                " interest parameter.",
            )
            return True
        return False

    def save_parameters_func(self):
        parameters = self.out_parameters()

        if self.unacceptable_parameters(parameters):
            return

        file_dialog = QFileDialog()
        settings_key = "save_parameters_filedialog_state"
        state = self.settings.value(settings_key, b"")
        if state:
            file_dialog.restoreState(state)
        fileName, _ = file_dialog.getSaveFileName(
            self,
            "Save parameter file",
            str(Path.cwd() / (self.getSessionName() + ".toml")),
            filter="TOML (*.toml)",
        )
        self.settings.setValue(settings_key, file_dialog.saveState())

        if not fileName:
            return

        # format everything before opening the file, so a failure cannot leave
        # a truncated .toml behind
        try:
            text = "".join(
                f"{key} = {toml_format(value)}\n" for key, value in parameters.items()
            )
            Path(fileName).write_text(text, encoding="utf_8")
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not save the parameters", str(exc))

    def new_video_paths(
        self,
        video_paths: list[str],
        video_size: tuple[int, int],
        n_frames: int,
        fps: float,
        episodes: list,
    ):
        self.session_name.setPlaceholderText(
            "&".join(Path(path).stem for path in video_paths)
        )
        # another video can have the same size, and the preview cache keys on
        # frame number and shape only
        self.enhancement_preview.bump_generation()
        self.ROI_Widget.set_video_size(video_size)
        self.segmentation_source.set_video_info(n_frames, video_size, video_paths)
        self.detectron2_panel.set_video_context(video_paths, self.session.output_dir)
        self.sam3_panel.set_video_context(video_paths, self.session.output_dir)
        self.yolo_panel.set_video_context(video_paths, self.session.output_dir)
        self.videoPlayer.setEnabled(False)
        self.tracking_interval.reset(n_frames)
        self.frame_analyzer.drawn_frame = -1
        self.bkg_widget.set_new_video_paths(video_paths, episodes)
        self.ROI_Widget.list.ListChanged.emit()
        self.videoPlayer.update_video_paths(video_paths, n_frames, video_size, fps)

        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True
            self.videoPlayer.setEnabled(True)
            # enabling every widget above would also re-enable the threshold
            # controls, which must stay off while external contours are in use
            self.segmentation_mode_changed(self.segmentation_source.mode())

        self.setWindowTitle(
            f"Segmentation App | {Path(video_paths[0]).name}"
            + (" [...]" if len(video_paths) > 1 else "")
        )
        self.videoPlayer.setEnabled(True)
        self.videoPlayer.update()


def toml_format(value: Any, width: int = 50) -> str:
    """Custom .toml formatter.

    Parameters
    ----------
    value : Any
        The value to format
    width : int, optional
        Maximum line width before wrapping, by default 50

    Returns
    -------
    str
        Formatted value
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return repr(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))  # inf and nan are valid TOML floats
    if isinstance(value, (str, Path)):
        # a JSON string is a valid TOML basic string. repr() would write a
        # literal one, which keeps the doubled backslashes of a Windows path
        # and cannot hold a string with both kinds of quote
        return json.dumps(str(value), ensure_ascii=False)
    if value is None:
        return '""'
    if isinstance(value, dict):
        # list(dict) would keep only the keys and silently lose the values
        items = ", ".join(f"{k} = {toml_format(v)}" for k, v in value.items())
        return "{ " + items + " }" if items else "{}"
    if not value:
        return "[]"

    items = [toml_format(item) for item in value]
    one_line = "[" + ", ".join(items) + "]"
    if len(one_line) < width:
        return one_line

    return "[\n" + "".join(f"    {item},\n" for item in items) + "]"


class SessionName(QLineEdit):
    "Double click to set the placeholder text as the text"

    def mouseDoubleClickEvent(self, event):
        if not self.text() and self.placeholderText():
            self.setText(self.placeholderText())
        else:
            super().mouseDoubleClickEvent(event)
