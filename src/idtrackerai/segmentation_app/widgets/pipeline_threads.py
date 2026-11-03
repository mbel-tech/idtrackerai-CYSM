"""Background workers for the Detectron2 preparation steps.

All three follow the shape :class:`BkgComputationThread` established in
``bkg_widget.py``: a plain ``QThread`` subclass, parameters pushed in before
``start()``, progress reported through signals handed straight to the worker
function, and ``quit()`` overridden to set a flag rather than stop the thread,
so cancelling is cooperative.

Following the deliberate convention documented at ``bkg_widget.py:60``, work
that can raise is kept out of ``run()`` where possible; what cannot be is
caught here and reported through a ``failed`` signal, because an exception
escaping a Qt slot aborts the process.
"""

import logging
from pathlib import Path

from qtpy.QtCore import QThread, Signal  # type: ignore[reportPrivateImportUsage]

from idtrackerai.extra_tools.detectron2_pipeline import dataset as dataset_mod
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline.errors import PipelineError


class SamplingThread(QThread):
    """Extracts frames for annotation without blocking the UI."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.result = None
        self.abort = False

    def set_parameters(
        self,
        videos: list[Path],
        output: Path,
        n_frames: int,
        settings: dict,
        seed: int = 0,
        image_format: str = "png",
        counts: list[int] | None = None,
    ) -> None:
        self.videos = videos
        self.output = output
        self.n_frames = n_frames
        self.settings = settings
        self.seed = seed
        self.image_format = image_format
        # frame counts the panel already gathered, so the run does not reopen
        # every clip a second time just to plan
        self.counts = counts
        self.result = None
        self.abort = False

    def quit(self):
        # Overridden deliberately, as BkgComputationThread does: this sets a
        # flag the worker checks, so a cancel finishes the current frame
        # rather than tearing the thread down mid-write.
        self.abort = True

    def run(self):
        self.set_progress_max.emit(self.n_frames)
        try:
            self.result = sampling_mod.sample_frames(
                self.videos,
                self.output,
                self.n_frames,
                self.settings,
                seed=self.seed,
                image_format=self.image_format,
                progress=self.set_progress_value.emit,
                abort=lambda: self.abort,
                counts=self.counts,
            )
        except PipelineError as exc:
            self.result = None
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.result = None
            logging.exception("Frame sampling failed")
            self.failed.emit(str(exc))


class FrameCountThread(QThread):
    """Counts the frames in each clip, so the preview can be instant.

    Counting means opening every file, which on a folder of fifty clips from a
    external drive takes seconds -- far too slow to do while a spinbox is being
    dragged. It is also the part that does not change when the spinbox does, so
    it is done once, here, and the panel keeps the answers.

    A clip that will not open counts zero rather than raising: the sampling
    step reports unreadable clips itself, and a preview is no place to fail.
    """

    counted = Signal(object)  # {Path: int}
    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.videos: list[Path] = []
        self.counts: dict[Path, int] = {}
        self.abort = False

    def set_parameters(self, videos: list[Path]) -> None:
        self.videos = list(videos)
        self.counts = {}
        self.abort = False

    def quit(self):
        self.abort = True

    def run(self):
        counts: dict[Path, int] = {}
        self.set_progress_max.emit(max(len(self.videos), 1))
        for i, video in enumerate(self.videos, 1):
            if self.abort:
                break
            counts[video] = sampling_mod.frame_count_or_zero(video)
            self.set_progress_value.emit(i)
        self.counts = counts
        self.counted.emit(counts)


class GpuCheckThread(QThread):
    """Asks whether this machine can run the GPU stages itself.

    Off the GUI thread because the check starts a subprocess that imports
    torch, which takes seconds when torch is installed.
    """

    reported = Signal(object)  # GpuReport

    def __init__(self):
        super().__init__()
        self.report = None
        self.abort = False

    def quit(self):
        self.abort = True

    def run(self):
        from idtrackerai.extra_tools.detectron2_pipeline import gpu

        self.report = gpu.describe_gpu()
        self.reported.emit(self.report)


class DatasetThread(QThread):
    """Validates annotations and builds the COCO dataset."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.result = None
        self.abort = False

    def set_parameters(self, request) -> None:
        self.request = request
        self.result = None
        self.abort = False

    def quit(self):
        self.abort = True

    def run(self):
        try:
            n = len(list(self.request.input_dir.glob("*.json")))
            self.set_progress_max.emit(max(n, 1))
            self.result = dataset_mod.build_coco_dataset(
                self.request,
                progress=self.set_progress_value.emit,
                abort=lambda: self.abort,
            )
        except PipelineError as exc:
            self.result = None
            if not self.abort:
                self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.result = None
            logging.exception("Dataset build failed")
            self.failed.emit(str(exc))


class Sam3PrelabelThread(QThread):
    """Drafts annotations with SAM 3, so the UI stays alive while it runs."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.summary = None
        self.abort = False
        self._reported_total = None

    def set_parameters(
        self,
        frames_dir: Path,
        weights: Path,
        prompt: str,
        label: str,
        device: str,
        score_threshold: float,
        max_instances: int,
        overwrite: bool = False,
    ) -> None:
        self.frames_dir = frames_dir
        self.weights = weights
        self.prompt = prompt
        self.label = label
        self.device = device
        self.score_threshold = score_threshold
        self.max_instances = max_instances
        self.overwrite = overwrite
        self.summary = None
        self.abort = False

    def quit(self):
        self.abort = True

    def _progress(self, done: int, total: int) -> None:
        # The dialog is created with a maximum of 100. Without this the bar
        # fills at frame 100 of however many there are and stays full, which
        # on a long run is indistinguishable from a hang.
        if total != self._reported_total:
            self._reported_total = total
            self.set_progress_max.emit(max(total, 1))
        self.set_progress_value.emit(done)

    def run(self):
        # Imported here, not at module scope: torch and sam3 are optional and
        # heavy, and the Segmentation App has to open on a machine with
        # neither installed.
        try:
            from idtrackerai.extra_tools.detectron2_pipeline import prelabel as prelabel_mod
            from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
                Sam3Predictor,
            )
        except ImportError as exc:
            self.failed.emit(
                f"SAM 3 is not installed in this environment ({exc}).\n\n"
                "Install it with:  pip install sam3\n"
                "It also needs torch with CUDA support."
            )
            return

        try:
            predictor = Sam3Predictor(
                checkpoint=self.weights,
                prompt=self.prompt,
                device=self.device,
                score_threshold=self.score_threshold,
                max_instances=self.max_instances,
            )
        except FileNotFoundError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logging.exception("Could not load SAM 3")
            self.failed.emit(f"Could not load SAM 3: {exc}")
            return

        try:
            self.summary = prelabel_mod.prelabel(
                input_dir=self.frames_dir,
                predictor=predictor,
                label=self.label,
                overwrite=self.overwrite,
                progress=self._progress,
                abort=lambda: self.abort,
            )
        except (PipelineError, FileNotFoundError) as exc:
            self.summary = None
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.summary = None
            logging.exception("Pre-labelling failed")
            self.failed.emit(str(exc))


class Sam3ExportThread(QThread):
    """Exports contour sidecars with SAM 3, one file per clip."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    set_progress_label = Signal(str)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.written: list[Path] = []
        self.skipped: list[Path] = []
        self.abort = False
        self._reported_total = None

    def set_parameters(
        self,
        videos: list[Path],
        output_dir: Path,
        weights: Path,
        prompt: str,
        device: str,
        score_threshold: float,
        max_instances: int,
        enhancement: dict,
        on_overlap: str = "merge",
        overwrite: bool = False,
        tracking_intervals=None,
    ) -> None:
        self.videos = videos
        self.tracking_intervals = tracking_intervals
        self.output_dir = output_dir
        self.weights = weights
        self.prompt = prompt
        self.device = device
        self.score_threshold = score_threshold
        self.max_instances = max_instances
        self.enhancement = enhancement
        self.on_overlap = on_overlap
        self.overwrite = overwrite
        self.written = []
        self.skipped = []
        self.abort = False
        self._reported_total = None

    def quit(self):
        self.abort = True

    def _progress(self, done: int, total: int) -> None:
        """Per clip, because the total frame count of a batch is not known
        without opening every file. The label says which clip, so a bar that
        starts again is readable rather than alarming."""
        if total != self._reported_total:
            self._reported_total = total
            self.set_progress_max.emit(max(total, 1))
        self.set_progress_value.emit(done)

    def run(self):
        from argparse import Namespace

        try:
            from idtrackerai.extra_tools.detectron2_pipeline import (
                inference as inference_mod,
            )
            from idtrackerai.extra_tools.detectron2_pipeline import (
                preprocessing as preprocessing_mod,
            )
            from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import (
                Sam3Predictor,
            )
        except ImportError as exc:
            self.failed.emit(
                f"SAM 3 is not installed in this environment ({exc}).\n\n"
                "Install it with:  pip install sam3"
            )
            return

        try:
            predictor = Sam3Predictor(
                checkpoint=self.weights,
                prompt=self.prompt,
                device=self.device,
                score_threshold=self.score_threshold,
                max_instances=self.max_instances,
            )
        except FileNotFoundError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logging.exception("Could not load SAM 3")
            self.failed.emit(f"Could not load SAM 3: {exc}")
            return

        # The same defaults the exporter's own command line uses, so a run
        # started here and one started from a terminal agree.
        args = Namespace(
            backend="sam3",
            prompt=self.prompt,
            weights=self.weights,
            score_threshold=self.score_threshold,
            max_instances=self.max_instances,
            min_component=80,
            dilate=1,
            dedup_iou=0.7,
            min_area=1.0,
            on_overlap=self.on_overlap,
            merge_overlap=0.15,
            limit=0,
            local_cache=None,
            keep_cache=False,
            progress_every=1_000_000,  # the progress bar reports instead
            classes=None,
            intervals=None,
        )
        enhance = preprocessing_mod.make_enhancer_from(self.enhancement)

        from idtrackerai.extra_tools.detectron2_pipeline import (
            segmentation_model as sm,
        )

        # Only the frames the session tracks, as the --session command line
        # does: each clip's share of the tracking intervals.
        n_frames = [inference_mod.video_frame_count(video) for video in self.videos]
        per_video = (
            sm.local_intervals(self.tracking_intervals, n_frames)
            if self.tracking_intervals
            else [None] * len(self.videos)
        )
        self.set_progress_label.emit("Hashing the SAM 3 checkpoint...")
        args.weights_sha256 = inference_mod.weights_sha256(self.weights)

        try:
            write_contours = inference_mod.load_writer()
            for index, video in enumerate(self.videos, start=1):
                if self.abort:
                    break
                output = self.output_dir / f"{video.stem}.h5"
                args.intervals = per_video[index - 1]

                # Resume, as the command line does. An export runs for hours,
                # and redoing a clip that already has its file would throw that
                # away every time someone cancels and comes back. The file is
                # still reported as written, so the session adopts the whole
                # set rather than only this run's share of it.
                # A file made with another prompt, threshold or frame range
                # is redone rather than adopted, as the command line does.
                if (
                    output.exists()
                    and not self.overwrite
                    and inference_mod.stale_reason(
                        output, video, args, self.enhancement
                    )
                    is None
                ):
                    self.skipped.append(output)
                    self.written.append(output)
                    continue

                self.set_progress_label.emit(
                    f"SAM 3 on {video.name}  ({index} of {len(self.videos)})"
                )
                stats = inference_mod.export_video(
                    video,
                    output,
                    args,
                    predictor,
                    predictor.description,
                    enhance,
                    self.enhancement,
                    write_contours,
                    progress=self._progress,
                    abort=lambda: self.abort,
                )
                if stats is None:  # cancelled part-way; nothing was written
                    break
                self.written.append(output)
        except PipelineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logging.exception("SAM 3 export failed")
            self.failed.emit(str(exc))
