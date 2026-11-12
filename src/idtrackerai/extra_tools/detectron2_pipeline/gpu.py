"""Whether this machine can run the GPU stages itself.

Colab exists because most people tracking animals do not have a CUDA card, not
because the training and export have to happen elsewhere. When the card is
there, sending the work to a hosted runtime means uploading tens of gigabytes
of video to answer a question the machine could answer locally.

The check runs in a subprocess. Importing torch costs seconds and a broken
CUDA install can abort the interpreter outright, neither of which should
happen inside the Segmentation App merely because someone opened a panel.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field

# Printed by the subprocess, read back as JSON. Kept as one string so there is
# one place to look when it needs changing.
PROBE = """
import json
report = {"torch": None, "detectron2": None, "sam3": None,
          "ultralytics": None, "cuda": False,
          "device": None, "notes": []}
try:
    import torch
    report["torch"] = torch.__version__
    try:
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["device"] = torch.cuda.get_device_name(0)
            report["notes"].append(
                "%.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9)
            )
    except Exception as exc:
        report["notes"].append("CUDA check failed: %s" % exc)
except Exception as exc:
    report["notes"].append("torch not importable: %s" % exc)
try:
    import detectron2
    report["detectron2"] = getattr(detectron2, "__version__", "unknown")
except Exception:
    pass
# Looked up rather than imported. Importing sam3 pulls in timm and a good
# deal else, and this probe answers for Detectron2 too: a slow import here
# would push the one subprocess past its timeout and report a machine that
# can train as one that cannot. Presence is enough, because loading it for
# real happens later and reports its own errors.
try:
    import importlib.metadata
    import importlib.util
    if importlib.util.find_spec("sam3") is not None:
        try:
            report["sam3"] = importlib.metadata.version("sam3")
        except Exception:
            report["sam3"] = "unknown"
    # ultralytics likewise: its import is slow and loads torch a second way
    if importlib.util.find_spec("ultralytics") is not None:
        try:
            report["ultralytics"] = importlib.metadata.version("ultralytics")
        except Exception:
            report["ultralytics"] = "unknown"
except Exception:
    pass
print(json.dumps(report))
"""


@dataclass
class GpuReport:
    """What the machine can and cannot do, in terms a person can act on."""

    torch_version: str | None = None
    detectron2_version: str | None = None
    sam3_version: str | None = None
    ultralytics_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    notes: list[str] = field(default_factory=list)
    checked: bool = False

    @property
    def usable(self) -> bool:
        """True when training and export can run here."""
        return bool(self.cuda_available and self.detectron2_version)

    @property
    def sam3_usable(self) -> bool:
        """True when SAM 3 can run here.

        Deliberately not ``usable``: SAM 3 needs a CUDA GPU and its own
        package, and does not need Detectron2 at all. Asking one question for
        both backends would refuse SAM 3 on a machine that can run it.
        """
        return bool(self.cuda_available and self.sam3_version)

    @property
    def sam3_missing(self) -> list[str]:
        """What stands between this machine and a SAM 3 run."""
        gaps = []
        if self.torch_version is None:
            gaps.append("PyTorch is not installed")
        elif not self.cuda_available:
            gaps.append("PyTorch cannot see a CUDA GPU")
        if self.sam3_version is None:
            gaps.append("the sam3 package is not installed")
        return gaps

    @property
    def yolo_usable(self) -> bool:
        """True when a YOLO model can export here: a CUDA GPU and ultralytics.
        Its own question for the same reason SAM 3 has one."""
        return bool(self.cuda_available and self.ultralytics_version)

    @property
    def yolo_missing(self) -> list[str]:
        """What stands between this machine and a YOLO run."""
        gaps = []
        if self.torch_version is None:
            gaps.append("PyTorch is not installed")
        elif not self.cuda_available:
            gaps.append("PyTorch cannot see a CUDA GPU")
        if self.ultralytics_version is None:
            gaps.append("the ultralytics package is not installed")
        return gaps

    @property
    def missing(self) -> list[str]:
        """What stands in the way, each as a short phrase."""
        gaps = []
        if self.torch_version is None:
            gaps.append("PyTorch is not installed")
        elif not self.cuda_available:
            gaps.append("PyTorch cannot see a CUDA GPU")
        if self.detectron2_version is None:
            gaps.append("Detectron2 is not installed")
        return gaps

    def summary(self) -> str:
        if not self.checked:
            return "Checking what this machine can do..."
        if self.usable:
            where = self.device_name or "a CUDA GPU"
            extra = f" ({', '.join(self.notes)})" if self.notes else ""
            return (
                f"This machine can train and export here: {where}{extra}, "
                f"Detectron2 {self.detectron2_version}."
            )
        return (
            "This machine cannot run the GPU stages: "
            + "; ".join(self.missing)
            + ". Use the Colab notebook, or install what is missing."
        )


def describe_gpu(python: str | None = None, timeout: float = 60.0) -> GpuReport:
    """Asks a subprocess what it can see. Never raises."""
    report = GpuReport(checked=True)
    try:
        finished = subprocess.run(
            [python or sys.executable, "-c", PROBE],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.notes.append(f"Could not run the check: {exc}")
        return report

    line = (finished.stdout or "").strip().splitlines()
    if not line:
        report.notes.append(
            "The check produced no answer"
            + (f": {finished.stderr.strip()[-200:]}" if finished.stderr else "")
        )
        return report

    try:
        data = json.loads(line[-1])
    except ValueError:
        report.notes.append("The check produced an answer that could not be read")
        return report

    report.torch_version = data.get("torch")
    report.detectron2_version = data.get("detectron2")
    report.sam3_version = data.get("sam3")
    report.ultralytics_version = data.get("ultralytics")
    report.cuda_available = bool(data.get("cuda"))
    report.device_name = data.get("device")
    report.notes = list(data.get("notes") or [])
    return report


INSTALL_HINT = (
    "To run these stages here you need a CUDA GPU, PyTorch built for it, and "
    "Detectron2.\n\n"
    "  1. Install PyTorch for your CUDA version from pytorch.org\n"
    "  2. pip install "
    "'git+https://github.com/facebookresearch/detectron2.git'\n\n"
    "Detectron2 has no PyPI release and is built from source, which needs a "
    "compiler. If that is more than you want to take on, the Colab notebook "
    "does the same work on a hosted GPU."
)


SAM3_INSTALL_HINT = (
    "To run SAM 3 here you need a CUDA GPU, PyTorch built for it, and Meta's "
    "sam3 package.\n\n"
    "  1. Install PyTorch for your CUDA version from pytorch.org\n"
    "  2. pip install sam3\n\n"
    "SAM 3 is a 3.4 GB model. Drafting a few hundred annotation frames on a "
    "CPU is slow but workable; exporting a whole video is not, and the Colab "
    "notebook does that on a hosted GPU instead."
)


YOLO_INSTALL_HINT = (
    "To run a YOLO model here you need a CUDA GPU, PyTorch built for it, and "
    "the ultralytics package.\n\n"
    "  1. Install PyTorch for your CUDA version from pytorch.org\n"
    "  2. pip install ultralytics\n\n"
    "ultralytics is AGPL-3.0 and is never installed by idtracker.ai itself. "
    "Without a GPU the export still runs, at a few frames per second: fine "
    "for a short clip, and the Colab notebook does whole videos on a hosted "
    "GPU instead."
)
