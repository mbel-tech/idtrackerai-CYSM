"""Writes a Colab notebook that runs a session's segmentation model and tracks.

The Segmentation App, or a session ``.toml``, already holds everything a run
needs: the videos, the region of interest, the number of animals and the
``[segmentation_model]`` table. This writes those into a notebook, so that on
Colab only one thing is left to say: which Drive folder holds the videos and
the weights. Set that folder in the first cell, pick a GPU runtime, Run all.

    idtrackerai_colab_notebook --session pen5.toml --output pen5_colab.ipynb

It works the same for every backend of the table (``detectron2``, ``sam3``,
``yolo``); the notebook installs only the package its model needs.

Paths of this computer do not travel. A video or a weights file is written
into the notebook by its file name alone and looked up by that name anywhere
under the Drive folder, so the folder can be laid out in any way. ``output_dir``
and ``external_contours`` are left out: the notebook keeps its contours, logs
and tracked sessions in the Drive folder itself.

The notebook follows docs/colab-idtrackerai-rules.md. Its helper and RESCUE
cells are the ones of the packaged notebooks, taken from there when a notebook
is written, so there is one copy to keep right.
"""

import argparse
import json
from pathlib import Path, PureWindowsPath

from . import segmentation_model as sm

INSTALL_FROM = "git+https://github.com/mbel-tech/idtrackerai-CYSM.git"
DRIVE_FOLDER = "/content/drive/MyDrive/idtrackerai_CYSM"
PACKAGED = Path(__file__).parent / "notebooks" / "colab_inference_roi.ipynb"

# Left behind: where things are on this computer. The notebook has its own.
NOT_PORTABLE = ("output_dir", "external_contours", "session", "protocol3_action")


class ColabNotebookError(ValueError):
    """These sessions cannot be written as one Colab notebook."""


def _file_name(path) -> str:
    """The last part of a path written on any system, Windows included."""
    return PureWindowsPath(str(path)).name


def _is_local_file(value) -> bool:
    text = str(value)
    return "\\" in text or text.startswith(("/", "~")) or bool(PureWindowsPath(text).drive)


def portable_session(params: dict) -> dict:
    """A session as the notebook carries it: files by name, no local folders."""
    params = {str(k).lower(): v for k, v in params.items() if v not in ("", None)}
    try:
        spec = sm.normalize_model(params.get("segmentation_model"))
    except sm.SegmentationModelError as exc:
        raise ColabNotebookError(str(exc)) from exc
    videos = params.get("video_paths") or []
    if isinstance(videos, (str, Path)):
        videos = [videos]
    name = params.get("name") or (Path(_file_name(videos[0])).stem if videos else "")
    if spec is None:
        raise ColabNotebookError(
            f"Session {name or '(unnamed)'} has no segmentation model, so there"
            " is nothing for Colab to run. Choose Detectron2, SAM 3 or YOLO as"
            " the segmentation source first"
        )
    if not videos:
        raise ColabNotebookError(f"Session {name or '(unnamed)'} has no videos")

    originals = {}
    names = []
    for video in videos:
        names.append(_file_name(video))
        originals[names[-1]] = str(video)
    if len(set(names)) != len(names):
        raise ColabNotebookError(
            f"Two videos of session {name} have the same file name. The notebook"
            " finds files on Drive by name, so rename one of them"
        )

    model = dict(params["segmentation_model"])
    model = {str(k).lower(): v for k, v in model.items() if v not in ("", None)}
    model.pop("contours_dir", None)
    weights = model.get("weights")
    if weights and str(weights) not in (sm.MODEL_ZOO, sm.HUGGINGFACE):
        model["weights"] = _file_name(weights)
        originals[model["weights"]] = str(weights)
    # a Detectron2 config is a model zoo name unless it is a file of its own
    if model.get("config") and _is_local_file(model["config"]):
        originals[_file_name(model["config"])] = str(model["config"])
        model["config"] = _file_name(model["config"])
        model["config_is_file"] = True

    settings = {
        k: v
        for k, v in params.items()
        if k not in NOT_PORTABLE + ("name", "video_paths", "segmentation_model")
    }
    return {
        "name": str(name),
        "videos": names,
        "model": model,
        "settings": settings,
        "_originals": originals,
    }


def _portable(sessions) -> list[dict]:
    if not sessions:
        raise ColabNotebookError("There is no session to write a notebook for")
    portable = [portable_session(s) for s in sessions]
    seen: dict[str, str] = {}
    names = set()
    for session in portable:
        if session["name"] in names:
            raise ColabNotebookError(
                f"Two sessions have the same name ({session['name']}). Their"
                " files on Drive would overwrite each other; rename one"
            )
        names.add(session["name"])
        for file_name, original in session.pop("_originals").items():
            if seen.setdefault(file_name, original) != original:
                raise ColabNotebookError(
                    f"Two different files are both called {file_name}"
                    f" ({seen[file_name]} and {original}). The notebook finds"
                    " files on Drive by name, so rename one of them"
                )
    return portable


def _session_files(session: dict) -> list[str]:
    files = list(session["videos"])
    model = session["model"]
    if model.get("weights") not in (None, sm.MODEL_ZOO, sm.HUGGINGFACE):
        files.append(model["weights"])
    if model.get("config_is_file"):
        files.append(model["config"])
    return files


def files_to_upload(sessions) -> list[str]:
    """The file names the Drive folder must hold, each once, weights first."""
    portable = _portable(sessions)
    videos, others = [], []
    for session in portable:
        for name in _session_files(session):
            target = videos if name in session["videos"] else others
            if name not in videos + others:
                target.append(name)
    return others + videos


def _backends(portable) -> list[str]:
    return sorted({s["model"]["backend"] for s in portable})


def _from_huggingface(portable) -> bool:
    return any(s["model"].get("weights") == sm.HUGGINGFACE for s in portable)


# ------------------------------------------------------------------ the cells


def _packaged_cell(starts_with: str) -> str:
    notebook = json.loads(PACKAGED.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.lstrip().startswith(starts_with):
            return source
    raise ColabNotebookError(f"{PACKAGED.name} has no cell starting {starts_with!r}")


def _intro(portable) -> str:
    backends = _backends(portable)
    label = {"detectron2": "Detectron2", "sam3": "SAM 3", "yolo": "YOLO"}
    models = " and ".join(label[b] for b in backends)
    count = len(portable)
    lines = [
        f"# idtracker.ai on Colab: {models}",
        "",
        (f"Written from {count} session{'' if count == 1 else 's'}:"
        f" {', '.join(s['name'] for s in portable[:6])}"
        f"{' ...' if count > 6 else ''}. The model, its settings, the region of"
        " interest and the tracking settings are already in the notebook."),
        "",
        ("1. Put these files in one folder on Google Drive. Subfolders are fine:"
        " each file is found by its name."),
        "",
        "```",
    ]
    listed: list[str] = []
    for session in portable:
        for name in _session_files(session):
            if name not in listed:
                listed.append(name)
    weights_first = [n for n in listed if not any(n in s["videos"] for s in portable)]
    videos = [n for n in listed if n not in weights_first]
    lines += weights_first + videos[:40]
    if len(videos) > 40:
        lines.append(f"... and {len(videos) - 40} more videos, all named in the settings cell")
    lines += ["```", ""]
    if "detectron2" in backends:
        lines += [
            ("Keep `training_metadata.json` beside the Detectron2 weights if you"
            " have it: it tells the model how it was trained."),
            "",
        ]
    if _from_huggingface(portable):
        lines += [
            ("The SAM 3 weights are fetched from Hugging Face, so nothing has to"
            " be uploaded for them. They are gated: request access at"
            " https://huggingface.co/facebook/sam3, then add your token as a"
            " Colab secret named `HF_TOKEN` (key icon in the left sidebar,"
            " Notebook access on)."),
            "",
        ]
    lines += [
        "2. Set that folder in the first code cell. It is the only thing to edit.",
        "3. Runtime -> Change runtime type -> a GPU.",
        "4. Runtime -> Run all.",
        "",
        ("Contours, logs, tracked sessions and `tracking_timetable.csv` are"
        " written into the same folder. After a disconnect, Run all again:"
        " what is finished is skipped."),
    ]
    return "\n".join(lines)


FOLDER_CELL = f"""from pathlib import Path

# ---- the only thing to edit: the Drive folder with the videos and weights ----
FOLDER = Path('{DRIVE_FOLDER}')
"""

MOUNT_CELL = """import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
      or 'No GPU: switch the runtime to GPU and rerun.')
from google.colab import drive
drive.mount('/content/drive')
assert FOLDER.is_dir(), f'{FOLDER} is not a folder on Drive: fix it in the first cell'
"""


def _settings_cell(portable) -> str:
    text = json.dumps(portable, indent=1, ensure_ascii=True)
    if "'''" in text:
        literal = repr(text)
    else:
        literal = "r'''" + text + "'''"
    return (
        "# Everything the sessions set. Nothing here needs editing: change the\n"
        "# session and write the notebook again instead.\n"
        "import json\n"
        "\n"
        "PROJECT = FOLDER                 # logs/ and tracked/ are kept here\n"
        "CONTOURS = FOLDER / 'contours'\n"
        "SESSION_FILES = FOLDER / 'sessions'\n"
        "CACHE = Path('/content/cache')   # videos are read from a local copy\n"
        "\n"
        f"SESSIONS = json.loads({literal})\n"
        "print(len(SESSIONS), 'session(s):', ', '.join(s['name'] for s in SESSIONS))\n"
    )


def _pip(*args: str) -> str:
    quoted = ", ".join(repr(a) for a in args)
    return f"pip({quoted})\n"


def _install_cell(portable, install_from: str) -> str:
    backends = _backends(portable)
    source = (
        "import os, subprocess, sys\n"
        "\n"
        "\n"
        "def pip(*args):\n"
        "    print('$ pip', *args, flush=True)\n"
        "    if subprocess.run([sys.executable, '-m', 'pip', *args]).returncode:\n"
        "        raise SystemExit('pip failed; see the output above')\n"
        "\n"
        "\n"
    )
    if "detectron2" in backends:
        source += (
            "# Detectron2 is built once (about 10 minutes) and the wheel kept on\n"
            "# Drive, so later runtimes install it in seconds.\n"
            "import torch\n"
            "try:\n"
            "    import detectron2\n"
            "except ImportError:\n"
            "    wheels_dir = (FOLDER / 'wheels' /\n"
            "                  f'py{sys.version_info.major}{sys.version_info.minor}'\n"
            "                  f'-torch{torch.__version__}')\n"
            "    wheels_dir.mkdir(parents=True, exist_ok=True)\n"
            "    if not list(wheels_dir.glob('detectron2*.whl')):\n"
            "        pip('wheel', '--no-deps', '--no-build-isolation', '-q',\n"
            "            '--wheel-dir', str(wheels_dir),\n"
            "            'git+https://github.com/facebookresearch/detectron2.git')\n"
            "    pip('install', '-q', str(next(wheels_dir.glob('detectron2*.whl'))))\n"
        )
    if "sam3" in backends:
        source += _pip("install", "-q", "sam3")
    if "yolo" in backends:
        source += _pip("install", "-q", "ultralytics")
    source += _pip("install", "-q", install_from)
    if _from_huggingface(portable):
        source += (
            "\n"
            "# SAM 3's weights come from Hugging Face. The token is read from the\n"
            "# Colab secret HF_TOKEN; never paste it into a cell. The download is\n"
            "# kept on the runtime's disk, not on Drive.\n"
            "os.environ['HF_HOME'] = '/content/hf_cache'\n"
            "try:\n"
            "    from google.colab import userdata\n"
            "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
            "    print('HF_TOKEN read from Colab secrets')\n"
            "except Exception as exc:\n"
            "    if not os.environ.get('HF_TOKEN'):\n"
            "        raise SystemExit(\n"
            "            f'No HF_TOKEN secret ({type(exc).__name__}). Add it under the'\n"
            "            ' key icon in the left sidebar, with Notebook access on.')\n"
        )
    source += (
        "\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type'\n"
        "GPU_NAME = torch.cuda.get_device_name(0)\n"
        "print('GPU:', GPU_NAME)\n"
    )
    return source


SESSIONS_CELL = """# Finds every video and weights file by its name under FOLDER, then writes one
# session file per recording into FOLDER/sessions. Nothing is written if a
# file is missing: the list of what to upload is printed instead.
import math


def toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float) and math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(toml_value(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k} = {toml_value(x)}' for k, x in v.items()) + '}'
    return json.dumps(v)   # a string with its backslashes escaped, or a number


skip = {CONTOURS, SESSION_FILES, FOLDER / 'tracked', FOLDER / 'logs', FOLDER / 'wheels'}
on_drive = {}
for path in sorted(FOLDER.rglob('*')):
    if path.is_file() and not any(s in path.parents for s in skip):
        on_drive.setdefault(path.name, []).append(path)


def find(name, problems):
    found = on_drive.get(name, [])
    if len(found) == 1:
        return str(found[0])
    problems.append(f'{name}: not found' if not found else
                    f'{name}: found {len(found)} times ({found[0]}, {found[1]})')


problems = []
written = {}
for session in SESSIONS:
    model = dict(session['model'])
    if model.get('weights') not in (None, 'huggingface', 'model_zoo'):
        model['weights'] = find(model['weights'], problems)
    if model.pop('config_is_file', False):
        model['config'] = find(model['config'], problems)
    model['contours_dir'] = str(CONTOURS / session['name'])
    params = dict(session['settings'])
    params['name'] = session['name']
    params['video_paths'] = [find(v, problems) for v in session['videos']]
    params['segmentation_model'] = model
    written[session['name']] = params

if problems:
    raise SystemExit(f'Fix these under {FOLDER}, then rerun this cell:\\n   '
                     + '\\n   '.join(sorted(set(problems))))

SESSION_FILES.mkdir(parents=True, exist_ok=True)
SESSION_TOMLS = []
for name, params in written.items():
    path = SESSION_FILES / f'{name}.toml'
    path.write_text(''.join(f'{k} = {toml_value(v)}\\n' for k, v in params.items()),
                    encoding='utf-8')
    SESSION_TOMLS.append(path)
print(len(SESSION_TOMLS), 'session file(s) written to', SESSION_FILES)
"""

EXPORT_CELL = """# Runs the model over each video on the GPU and writes its contours to
# FOLDER/contours/<session>/. Resumable: after a disconnect, finished videos
# are skipped. The exporter exits 0 on success; a failure stops the notebook.
run_step(['idtrackerai_d2_export', '--session', *SESSION_TOMLS,
          '--local-cache', CACHE, '--progress-every', 1000],
         LOGS / 'export.txt')
"""

TRACK_CELL = """# Tracks each recording on the Colab disk (LOCAL_SESSIONS) and copies its
# session folder to TRACKED on Drive only once it has finished, so a
# disconnect never leaves half a session on Drive. A failure in one recording
# does not stop the others; a rerun skips recordings already in TRACKED.
# Success is judged by tracking_succeeded(), NOT by the exit code:
# idtracker.ai exits 1 when it succeeds.
os.environ['QT_QPA_PLATFORM'] = 'offscreen'   # no display on a runtime
LOCAL_SESSIONS.mkdir(parents=True, exist_ok=True)
TRACKED.mkdir(parents=True, exist_ok=True)

results = []
for path in SESSION_TOMLS:
    name = path.stem
    local = LOCAL_SESSIONS / f'session_{name}'
    drive = TRACKED / f'session_{name}'
    if tracking_succeeded(drive):
        print(f'=== {name}: already tracked, in {drive}')
        results.append((name, 'already tracked'))
        continue

    shutil.rmtree(local, ignore_errors=True)   # start from a clean session folder
    print(f'=== {name} ' + '=' * 40, flush=True)
    log = LOGS / f'track_{name}.txt'
    rc, tail = run_logged(['idtrackerai', '--load', path, '--track',
                           '--output_dir', LOCAL_SESSIONS], log)
    if tracking_succeeded(local, tail):
        copy_to_drive(local, drive)
        results.append((name, 'ok'))
        print(f'=== {name}: ok, copied to {drive}\\n', flush=True)
    else:
        results.append((name, f'FAILED (exit {rc}, log {log})'))
        print(f'=== {name}: FAILED, see {log}\\n', flush=True)

print('Summary')
for name, status in results:
    print(f'   {name:<24} {status}')
failed = [n for n, s in results if s.startswith('FAILED')]
if failed:
    print(f'\\n{len(failed)} recording(s) failed: {failed}')
    print('If a run finished but was judged failed, the RESCUE cell below copies it.')
"""

TIMETABLE_CELL = """# tracking_timetable.csv in FOLDER: one row per tracked session with the GPU
# and the hours its tracking took, read from the session's own idtrackerai.log
# (the start is its "Date:" line, the end is when the log was last written).
# Rerunning updates the rows of these sessions and keeps the others.
import csv, re
from datetime import datetime

TIMETABLE = FOLDER / 'tracking_timetable.csv'
FIELDS = ['session', 'backend', 'gpu', 'start', 'end', 'hours', 'status']
gpu = globals().get('GPU_NAME') or 'unknown'

rows = {}
if TIMETABLE.is_file():
    with open(TIMETABLE, newline='', encoding='utf-8') as f:
        rows = {r['session']: r for r in csv.DictReader(f)}

for session in SESSIONS:
    name = session['name']
    log = TRACKED / f'session_{name}' / 'idtrackerai.log'
    if not log.is_file():
        continue
    started = re.search(r'Date: (\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d)',
                        log.read_text(errors='ignore'))
    end = datetime.fromtimestamp(log.stat().st_mtime).replace(microsecond=0)
    start = datetime.fromisoformat(started.group(1)) if started else None
    rows[name] = {
        'session': name,
        'backend': session['model']['backend'],
        'gpu': gpu,
        'start': start.isoformat(sep=' ') if start else '',
        'end': end.isoformat(sep=' '),
        'hours': f'{(end - start).total_seconds() / 3600:.3f}' if start else '',
        'status': 'ok' if tracking_succeeded(log.parent) else 'check',
    }

with open(TIMETABLE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
total = sum(float(r['hours'] or 0) for r in rows.values())
print(f'{TIMETABLE}: {len(rows)} session(s), {total:.2f} h of tracking')
"""


def _cell(cell_id: str, source: str, kind: str = "code") -> dict:
    cell = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": source.rstrip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    return cell


def build_notebook(sessions, install_from: str = INSTALL_FROM) -> dict:
    """The notebook of these sessions (dicts of session parameters)."""
    portable = _portable(sessions)
    md = "markdown"
    cells = [
        _cell("intro", _intro(portable), md),
        _cell("folder", FOLDER_CELL),
        _cell("mount", MOUNT_CELL),
        _cell("settings", _settings_cell(portable)),
        _cell("helpers", _packaged_cell("# Running idtracker.ai and the exporter")),
        _cell("install-title", "## Install", md),
        _cell("install", _install_cell(portable, install_from)),
        _cell("sessions-title", "## Find the files and write the session files", md),
        _cell("sessions", SESSIONS_CELL),
        _cell("export-title", "## Segment: the model writes each video's contours", md),
        _cell("export", EXPORT_CELL),
        _cell("track-title", "## Track", md),
        _cell("track", TRACK_CELL),
        _cell("rescue", _packaged_cell("# RESCUE")),
        _cell("timetable-title", "## Time used", md),
        _cell("timetable", TIMETABLE_CELL),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path, sessions, install_from: str = INSTALL_FROM) -> list[str]:
    """Writes the notebook and returns the file names to put on Drive."""
    notebook = build_notebook(sessions, install_from)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return files_to_upload(sessions)


def upload_message(files: list[str], sessions) -> str:
    """What to do with the notebook, for whoever just saved it."""
    lines = ["Put these in one folder on Google Drive (subfolders are fine):", ""]
    lines += [f"   {name}" for name in files[:25]]
    if len(files) > 25:
        lines.append(f"   ... and {len(files) - 25} more, listed in the notebook")
    if _from_huggingface(_portable(sessions)):
        lines += ["", ("The SAM 3 weights are fetched from Hugging Face with your"
                  " HF_TOKEN Colab secret.")]  # fmt: skip
    lines += [
        "",
        ("Then open the notebook in Colab, set that folder in the first cell,"
        " choose a GPU runtime and Run all."),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- command line


def load_session(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import tomllib

        params = tomllib.loads(text)
    except ImportError:  # Python < 3.11
        import toml

        params = toml.loads(text)
    params = {str(k).lower(): v for k, v in params.items()}
    params.setdefault("name", Path(path).stem)
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Write a Colab notebook that segments and tracks sessions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session",
        type=Path,
        nargs="+",
        required=True,
        help="session .toml file(s) with a [segmentation_model] table, or a glob",
    )
    parser.add_argument("--output", type=Path, required=True, help="the .ipynb to write")
    parser.add_argument(
        "--install-from",
        default=INSTALL_FROM,
        help="what the notebook pip-installs idtracker.ai from, e.g. a branch:"
        f" {INSTALL_FROM}@my-branch",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.session:
        pattern = Path(pattern).expanduser()
        if any(ch in pattern.name for ch in "*?"):
            paths.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            paths.append(pattern)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing or not paths:
        raise SystemExit("ERROR: no such session file: " + (", ".join(missing) or "none"))
    try:
        sessions = [load_session(p) for p in paths]
        files = write_notebook(args.output, sessions, args.install_from)
    except ColabNotebookError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Wrote {args.output} for {len(sessions)} session(s).")
    print(upload_message(files, sessions))


if __name__ == "__main__":
    main()
