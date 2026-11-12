"""Writes a Colab notebook that runs a session's segmentation model and tracks.

The Segmentation App, or a session ``.toml``, already holds everything a run
needs: the videos, the region of interest, the number of animals and the
``[segmentation_model]`` table. This writes those into a notebook, so that on
Colab only one thing is left to say: which Drive folder holds the videos and
the weights. Set that folder in the first cell, pick a GPU runtime, Run all.

    idtrackerai_colab_notebook --session pen5.toml --output pen5_colab.ipynb

It works the same for every backend of the table (``detectron2``, ``sam3``,
``yolo``); the notebook installs only the package its model needs.

On Drive, a video's own settings come first. A ``.toml`` in the folder named
after a video (``pen5.mp4``, ``pen5.toml``), or listing it in ``video_paths``,
is that video's companion and is what it is run with. A video the notebook was
written for falls back to the session saved in the notebook. Any other video
waits: the notebook has a settings cell for them, and runs them only once its
``USE_FOR_VIDEOS_WITHOUT_A_TOML`` box is ticked, so one pen's region of
interest is never applied to another pen by accident. With no ``--session``
at all, the notebook runs purely from the folder.

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
    models = " and ".join(label[b] for b in backends) or "any segmentation model"
    count = len(portable)
    lines = [f"# idtracker.ai on Colab: {models}", ""]
    if portable:
        lines += [
            (f"Written from {count} session{'' if count == 1 else 's'}:"
             f" {', '.join(s['name'] for s in portable[:6])}"
             f"{' ...' if count > 6 else ''}. The model, its settings, the region of"
             " interest and the tracking settings are already in the notebook."),
            "",
        ]  # fmt: skip
    lines += [
        ("1. Put the videos and the weights in one folder on Google Drive."
         " Subfolders are fine: each file is found by its name."),
        "",
    ]  # fmt: skip
    listed: list[str] = []
    for session in portable:
        for name in _session_files(session):
            if name not in listed:
                listed.append(name)
    if listed:
        weights_first = [
            n for n in listed if not any(n in s["videos"] for s in portable)
        ]
        videos = [n for n in listed if n not in weights_first]
        lines += ["```", *weights_first, *videos[:40]]
        if len(videos) > 40:
            lines.append(
                f"... and {len(videos) - 40} more videos, all named in the settings cell"
            )
        lines += ["```", ""]
    if "detectron2" in backends:
        lines += [
            ("Keep `training_metadata.json` beside the Detectron2 weights if you"
             " have it: it tells the model how it was trained."),
            "",
        ]  # fmt: skip
    if _from_huggingface(portable):
        lines += [
            ("The SAM 3 weights are fetched from Hugging Face, so nothing has to"
             " be uploaded for them. They are gated: request access at"
             " https://huggingface.co/facebook/sam3, then add your token as a"
             " Colab secret named `HF_TOKEN` (key icon in the left sidebar,"
             " Notebook access on)."),
            "",
        ]  # fmt: skip
    lines += [
        "2. Set that folder in the first code cell.",
        "3. Runtime -> Change runtime type -> a GPU.",
        "4. Runtime -> Run all.",
        "",
        "**How each video gets its settings**",
        "",
        ("- A video with a companion `.toml` in the folder is run with that file:"
         " a `.toml` with the video's own name (`pen5.mp4` and `pen5.toml`), or"
         " any `.toml` that lists the video in `video_paths`. It may sit in"
         " another subfolder. Paths inside it are ignored; only file names count."),
        ("- Otherwise, a video this notebook was written for is run with the"
         " settings saved in the notebook."),
        ("- Any other video is not run until you have looked at the settings cell"
         " (the third code cell) and ticked `USE_FOR_VIDEOS_WITHOUT_A_TOML`"
         " there. If no video has settings at all, the notebook stops and says so."),
        "",
        ("Contours, logs, tracked sessions and `tracking_timetable.csv` are"
         " written into the same folder. After a disconnect, Run all again:"
         " what is finished is skipped."),
    ]  # fmt: skip
    return "\n".join(lines)


FOLDER_CELL = f"""from pathlib import Path

# ---- the only thing to edit: the Drive folder with the videos and weights ----
FOLDER = Path('{DRIVE_FOLDER}')
"""

MOUNT_CELL = """import torch
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type, then Run all'
GPU_NAME = torch.cuda.get_device_name(0)
print('GPU:', GPU_NAME)
from google.colab import drive
drive.mount('/content/drive')
assert FOLDER.is_dir(), f'{FOLDER} is not a folder on Drive: fix it in the first cell'
"""


def _json_literal(value) -> str:
    text = json.dumps(value, indent=1, ensure_ascii=True)
    if "\'\'\'" in text or text.endswith("\'"):
        return f"json.loads({text!r})"
    return "json.loads(r\'\'\'" + text + "\'\'\')"


def _settings_cell(portable) -> str:
    """The form for videos that have no companion .toml, filled in from the
    first session the notebook was written from, and the sessions themselves."""
    first = portable[0] if portable else {"model": {"backend": "yolo"}, "settings": {}}
    model = dict(first["model"])
    settings = dict(first["settings"])
    backend = model["backend"]
    weights = model.pop("weights", "") or ""
    score = model.pop("score_threshold", None)
    if score is None:  # the backend's own default, as tracking would take it
        score = sm.normalize_model(first["model"])["score_threshold"] if portable else 0.5
    prompt = model.pop("prompt", "") or ""
    imgsz = model.pop("imgsz", 0) or 0
    animals = settings.pop("number_of_animals", 0) or 0
    without = bool(settings.pop("track_wo_identities", False))
    roi = settings.pop("roi_list", None)
    return (
        "#@title Settings for videos that have no companion .toml\n"
        "# A video with a .toml of its own in FOLDER is run with that file, and a\n"
        "# video this notebook was written for with the settings saved below.\n"
        "# These are for every other video, and only once the last box but one is\n"
        "# ticked: check them first, the region of interest above all.\n"
        "import json\n"
        "\n"
        f"BACKEND = {backend!r}  #@param [\"yolo\", \"sam3\", \"detectron2\"]\n"
        "# the weights' file name in FOLDER ('huggingface' fetches SAM 3's)\n"
        f"WEIGHTS = {weights!r}  #@param {{type:\"string\"}}\n"
        f"NUMBER_OF_ANIMALS = {int(animals)}  #@param {{type:\"integer\"}}\n"
        f"TRACK_WITHOUT_IDENTITIES = {without}  #@param {{type:\"boolean\"}}\n"
        f"SCORE_THRESHOLD = {float(score)}  #@param {{type:\"number\"}}\n"
        "# SAM 3 only: what to look for, e.g. 'bird'\n"
        f"PROMPT = {prompt!r}  #@param {{type:\"string\"}}\n"
        "# YOLO only: 0 = the size the model was trained at\n"
        f"IMAGE_SIZE = {int(imgsz)}  #@param {{type:\"integer\"}}\n"
        "# idtracker.ai's roi_list, e.g. ['+ Polygon [[x, y], ...]', '- Polygon [...]'];\n"
        "# None tracks the whole frame\n"
        f"ROI_LIST = {roi!r}\n"
        "# run the videos that have no .toml with the settings above\n"
        "USE_FOR_VIDEOS_WITHOUT_A_TOML = False  #@param {type:\"boolean\"}\n"
        "# run every video with the model above, whatever its .toml says\n"
        "USE_THIS_MODEL_FOR_EVERY_VIDEO = False  #@param {type:\"boolean\"}\n"
        "\n"
        "# The rest of what the notebook was written with. To change it, change\n"
        "# the session and write the notebook again.\n"
        f"OTHER_SETTINGS = {_json_literal(settings)}\n"
        f"MODEL_EXTRA = {_json_literal(model)}\n"
        f"SAVED_SESSIONS = {_json_literal(portable)}\n"
        "\n"
        "PROJECT = FOLDER                 # logs/ and tracked/ are kept here\n"
        "CONTOURS = FOLDER / 'contours'\n"
        "SESSION_FILES = FOLDER / 'sessions'\n"
        "CACHE = Path('/content/cache')   # videos are read from a local copy\n"
    )


def _install_cell(install_from: str) -> str:
    return (
        "# Installs this fork, and only the package of each model the sessions\n"
        "# found above need (BACKENDS).\n"
        "import os, subprocess, sys\n"
        "\n"
        "\n"
        "def pip(*args):\n"
        "    print('$ pip', *args, flush=True)\n"
        "    if subprocess.run([sys.executable, '-m', 'pip', *args]).returncode:\n"
        "        raise SystemExit('pip failed; see the output above')\n"
        "\n"
        "\n"
        "if 'detectron2' in BACKENDS:\n"
        "    # Built once (about 10 minutes); the wheel is kept on Drive, so\n"
        "    # later runtimes install it in seconds.\n"
        "    try:\n"
        "        import detectron2\n"
        "    except ImportError:\n"
        "        from importlib import metadata\n"
        "        try:\n"
        "            torch_version = metadata.version('torch')\n"
        "        except metadata.PackageNotFoundError:\n"
        "            torch_version = 'unknown'\n"
        "        wheels_dir = (FOLDER / 'wheels' /\n"
        "                      f'py{sys.version_info.major}{sys.version_info.minor}'\n"
        "                      f'-torch{torch_version}')\n"
        "        wheels_dir.mkdir(parents=True, exist_ok=True)\n"
        "        if not list(wheels_dir.glob('detectron2*.whl')):\n"
        "            pip('wheel', '--no-deps', '--no-build-isolation', '-q',\n"
        "                '--wheel-dir', str(wheels_dir),\n"
        "                'git+https://github.com/facebookresearch/detectron2.git')\n"
        "        built = list(wheels_dir.glob('detectron2*.whl'))\n"
        "        pip('install', '-q', str(built[0]) if built else 'detectron2')\n"
        "if 'sam3' in BACKENDS:\n"
        "    pip('install', '-q', 'sam3')\n"
        "if 'yolo' in BACKENDS:\n"
        "    pip('install', '-q', 'ultralytics')\n"
        f"pip('install', '-q', {install_from!r})\n"
        "\n"
        "if NEEDS_HUGGINGFACE:\n"
        "    # SAM 3's weights come from Hugging Face. The token is read from the\n"
        "    # Colab secret HF_TOKEN; never paste it into a cell. The download is\n"
        "    # kept on the runtime's disk, not on Drive.\n"
        "    os.environ['HF_HOME'] = '/content/hf_cache'\n"
        "    try:\n"
        "        from google.colab import userdata\n"
        "        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
        "        print('HF_TOKEN read from Colab secrets')\n"
        "    except Exception as exc:\n"
        "        if not os.environ.get('HF_TOKEN'):\n"
        "            raise SystemExit(\n"
        "                f'No HF_TOKEN secret ({type(exc).__name__}). Add it under the'\n"
        "                ' key icon in the left sidebar, with Notebook access on.')\n"
    )


SESSIONS_CELL = """# Works out how each video in FOLDER is to be run, then writes one session
# file per recording into FOLDER/sessions:
#   1. a companion .toml in FOLDER: named after the video, or listing it in
#      video_paths (only file names count, so a file written on Windows works);
#   2. otherwise the session saved in this notebook for that video;
#   3. otherwise the settings cell, once USE_FOR_VIDEOS_WITHOUT_A_TOML is ticked.
# Nothing is written if something is missing: what to fix is printed instead.
import math, tomllib
from pathlib import PureWindowsPath

VIDEO_TYPES = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv'}
KEYWORDS = ('huggingface', 'model_zoo')
LEFT_OUT = ('output_dir', 'external_contours', 'session', 'protocol3_action',
            'video_paths', 'segmentation_model', 'name')


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


def file_name(path):
    return PureWindowsPath(str(path)).name   # of a Windows or a Linux path


def is_a_path(text):
    text = str(text)
    return ('\\\\' in text or text.startswith(('/', '~'))
            or bool(PureWindowsPath(text).drive))


skip = {CONTOURS, SESSION_FILES, FOLDER / 'tracked', FOLDER / 'logs', FOLDER / 'wheels'}
on_drive = {}
for path in sorted(FOLDER.rglob('*')):
    if (path.is_file() and not path.name.startswith('._')
            and not any(s in path.parents for s in skip)):
        on_drive.setdefault(path.name, []).append(path)
videos_here = sorted(n for n in on_drive if Path(n).suffix.lower() in VIDEO_TYPES)


def find(name, problems):
    found = on_drive.get(name, [])
    if len(found) == 1:
        return str(found[0])
    problems.append(f'{name}: not found' if not found else
                    f'{name}: found {len(found)} times ({found[0]}, {found[1]})')


problems, notes, sessions, claimed = [], [], [], {}


def claim(videos, who):
    for video in videos:
        if video in claimed:
            problems.append(f'{video}: both {claimed[video]} and {who} are for'
                            ' this video; keep one of them')
        claimed[video] = who


# 1. companion .toml files
for name in sorted(on_drive):
    if not name.lower().endswith('.toml'):
        continue
    for path in on_drive[name]:
        try:
            raw = tomllib.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            notes.append(f'{path.name}: not read ({exc})')
            continue
        params = {str(k).lower(): v for k, v in raw.items() if v != ''}
        listed = params.get('video_paths') or []
        listed = [listed] if isinstance(listed, str) else list(listed)
        videos = ([file_name(v) for v in listed]
                  or [v for v in videos_here if Path(v).stem == path.stem])
        if not videos:
            continue                      # some other .toml, not a session's
        absent = [v for v in videos if v not in on_drive]
        if absent:
            notes.append(f'{path.name}: left out, {absent[0]} is not in the folder')
            continue
        claim(videos, path.name)
        sessions.append({'name': str(params.get('name') or path.stem),
                         'source': path.name, 'videos': videos,
                         'model': params.get('segmentation_model'),
                         'settings': {k: v for k, v in params.items()
                                      if k not in LEFT_OUT}})

# 2. the sessions this notebook was written from
for saved in SAVED_SESSIONS:
    if any(v in claimed for v in saved['videos']):
        continue                          # its .toml in the folder comes first
    absent = [v for v in saved['videos'] if v not in on_drive]
    if absent:
        notes.append(f"{saved['name']}: left out, {absent[0]} is not in the folder")
        continue
    claim(saved['videos'], 'this notebook')
    sessions.append({'name': saved['name'], 'source': 'this notebook',
                     'videos': saved['videos'], 'model': dict(saved['model']),
                     'settings': dict(saved['settings'])})

# 3. the settings cell, for what is left
others = [v for v in videos_here if v not in claimed]
ask = ''
if others and USE_FOR_VIDEOS_WITHOUT_A_TOML:
    if not WEIGHTS:
        problems.append('settings cell: WEIGHTS is empty')
    if not NUMBER_OF_ANIMALS and not TRACK_WITHOUT_IDENTITIES:
        problems.append('settings cell: set NUMBER_OF_ANIMALS, or tick'
                        ' TRACK_WITHOUT_IDENTITIES')
    for video in others:
        settings = dict(OTHER_SETTINGS)
        settings.update(number_of_animals=NUMBER_OF_ANIMALS,
                        track_wo_identities=TRACK_WITHOUT_IDENTITIES)
        if ROI_LIST:
            settings['roi_list'] = list(ROI_LIST)
        sessions.append({'name': Path(video).stem, 'source': 'the settings cell',
                         'videos': [video], 'model': None, 'settings': settings})
elif others:
    shown = ', '.join(others[:8]) + (' ...' if len(others) > 8 else '')
    ask = (f'{len(others)} video(s) have no companion .toml and are not run: {shown}.\\n'
           'To run them, check the settings cell (the third code cell: model,'
           ' number of animals, region of interest), tick'
           ' USE_FOR_VIDEOS_WITHOUT_A_TOML there and Run all again. Or put a'
           ' .toml named after each video in the folder.')


def settings_model():
    model = dict(MODEL_EXTRA) if MODEL_EXTRA.get('backend') == BACKEND else {}
    model.update(backend=BACKEND, weights=WEIGHTS, score_threshold=SCORE_THRESHOLD)
    if BACKEND == 'sam3' and PROMPT:
        model['prompt'] = PROMPT
    if BACKEND == 'yolo' and IMAGE_SIZE:
        model['imgsz'] = IMAGE_SIZE
    return model


written, RUN = {}, []
for session in sessions:
    name, model = session['name'], session['model']
    if name in written:
        problems.append(f'two sessions are called {name}; rename one')
    if USE_THIS_MODEL_FOR_EVERY_VIDEO or not model:
        if not WEIGHTS:
            problems.append(f"{name}: {session['source']} names no model; set"
                            ' BACKEND and WEIGHTS in the settings cell')
            continue
        model = settings_model()
    model = {str(k).lower(): v for k, v in model.items() if v != ''}
    model.pop('contours_dir', None)
    weights = model.get('weights')
    if weights and weights not in KEYWORDS:
        model['weights'] = find(file_name(weights), problems)
    if model.pop('config_is_file', False) or is_a_path(model.get('config', '')):
        model['config'] = find(file_name(model['config']), problems)
    model['contours_dir'] = str(CONTOURS / name)
    params = dict(session['settings'])
    params['name'] = name
    params['video_paths'] = [find(v, problems) for v in session['videos']]
    params['segmentation_model'] = model
    written[name] = params
    RUN.append({'name': name, 'backend': model.get('backend'),
                'source': session['source'],
                'huggingface': weights == 'huggingface'})

for note in notes:
    print('note:', note)
if problems:
    raise SystemExit(f'Fix these under {FOLDER}, then rerun this cell:\\n   '
                     + '\\n   '.join(sorted(set(problems))))
if not RUN:
    raise SystemExit(ask or f'No video with settings was found under {FOLDER}.'
                     ' Check the folder in the first cell.')

BACKENDS = sorted({r['backend'] for r in RUN})
NEEDS_HUGGINGFACE = any(r['huggingface'] for r in RUN)
SESSION_FILES.mkdir(parents=True, exist_ok=True)
SESSION_TOMLS = []
for name, params in written.items():
    path = SESSION_FILES / f'{name}.toml'
    path.write_text(''.join(f'{k} = {toml_value(v)}\\n' for k, v in params.items()),
                    encoding='utf-8')
    SESSION_TOMLS.append(path)

print(f'{len(RUN)} session(s) to run; session files written to {SESSION_FILES}')
for r in RUN:
    print(f"   {r['name']:<28} {r['backend']:<11} settings from {r['source']}")
if ask:
    print('\\n' + ask)
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

for session in RUN:
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
        'backend': session['backend'],
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
        _cell("sessions-title", "## Find the videos, their settings and the weights", md),
        _cell("sessions", SESSIONS_CELL),
        _cell("install-title", "## Install", md),
        _cell("install", _install_cell(install_from)),
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
    if files:
        lines = ["Put these in one folder on Google Drive (subfolders are fine):", ""]
        lines += [f"   {name}" for name in files[:25]]
        if len(files) > 25:
            lines.append(f"   ... and {len(files) - 25} more, listed in the notebook")
    else:
        lines = ["Put the videos and the weights in one folder on Google Drive."]
    if _from_huggingface(_portable(sessions)):
        lines += [
            "",
            "The SAM 3 weights are fetched from Hugging Face with your"
            " HF_TOKEN Colab secret.",
        ]
    lines += [
        "",
        "Then open the notebook in Colab, set that folder in the first cell,"
        " choose a GPU runtime and Run all.",
        "",
        "A video with a .toml of its own in that folder (same name, or one"
        " that lists it) is run with that file. Other videos wait until you"
        " have checked the notebook's settings cell.",
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
        default=[],
        help="session .toml file(s) with a [segmentation_model] table, or a glob."
        " Without any, the notebook runs from the .toml files it finds beside"
        " the videos on Drive, and from its settings cell",
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
    if missing or (args.session and not paths):
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
