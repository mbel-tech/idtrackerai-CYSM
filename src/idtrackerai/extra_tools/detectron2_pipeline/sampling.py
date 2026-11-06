"""Extracting frames from videos for annotation.

The logic lives here rather than in the command-line script so that the
Segmentation App and the CLI run the same code. :func:`sample_frames` takes
``progress`` and ``abort`` callables, matching the convention
:func:`idtrackerai.base.animals_detection.segmentation.generate_frame_stack`
already uses, which is what lets one function drive a terminal progress bar and
a ``QProgressDialog`` without knowing which it is talking to.

Sampling is stratified by default: each video is cut into as many equal blocks
as there are frames to take, and one frame is drawn from each. Uniform random
sampling over a 30000-frame clip clumps, leaving stretches unrepresented;
stratifying guarantees coverage of the whole recording, including whatever the
lighting does halfway through.
"""

import glob
import json
import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import NamedTuple

import cv2

try:
    from .errors import SamplingError
    from .preprocessing import (
        DEFAULT_SETTINGS,
        SETTING_KEYS,
        load_profile,
        make_enhancer_from,
        save_profile,
    )
except ImportError:  # loaded by path, without the package around it
    from errors import SamplingError  # type: ignore[no-redef]
    from preprocessing import (  # type: ignore[no-redef]
        DEFAULT_SETTINGS,
        SETTING_KEYS,
        load_profile,
        make_enhancer_from,
        save_profile,
    )


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
"""What counts as a video when a whole folder is given instead of a file list."""


@dataclass
class VideoPlan:
    """How many frames to take from one video, and which."""

    video: Path
    available: int
    requested: int
    indices: list[int] = field(default_factory=list)
    readable: bool = True
    """False when the file could not be opened or reports no frames.

    Unreadable clips stay in the plan rather than aborting it, so one bad file
    in a folder of fifty does not cost the whole run, and so the caller can
    name exactly which ones were skipped.
    """
    name: str = ""
    """What the clip's frames are called: the file name, or the one
    :func:`assign_names` gave it when file names are shared. Defaults to the
    file name's stem."""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.video.stem


@dataclass
class SamplingResult:
    frames: list[dict]
    requested: int
    output: Path
    manifest_path: Path | None = None
    profile_path: Path | None = None
    per_video: dict[str, int] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)
    complete: bool = True

    @property
    def written(self) -> int:
        return len(self.frames)


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SamplingError(f"Could not open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def frame_count_or_zero(path: Path) -> int:
    """Frame count, or 0 for a file that cannot be read as a video."""
    try:
        return max(0, video_frame_count(path))
    except (SamplingError, cv2.error):
        return 0


def _location(path) -> str:
    """A path in a form two spellings of the same file agree on."""
    return os.path.normcase(os.path.realpath(path))


def recorded_clips(output: Path) -> list[tuple[str, str]]:
    """``(source path, name)`` of each clip with a frame still in ``output``.

    The manifest records which clip every frame came from and the name the
    frames were given. Only frames still on disk count, as in
    :func:`_merge_manifest`: a batch the user deleted leaves nothing to
    overwrite, so it no longer holds its name. A clip counts once, however many
    of its frames remain, and costs about one ``stat`` to confirm.

    A record with no source path (written before paths were kept) cannot say
    which clip it came from. A manifest that is missing, unreadable or the wrong
    shape says nothing at all. Neither objects: they are skipped. A
    ``source_video`` that carries a video extension is read as the stem it
    stands for, since that is the spelling older manifests used.
    """
    try:
        records = json.loads(
            (output / "sampling_manifest.json").read_text(encoding="utf-8")
        ).get("frames", [])
    except (OSError, ValueError, AttributeError):
        return []
    if not isinstance(records, list):
        return []

    held: dict[tuple[str, str], None] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        file_name = record.get("file_name")
        name = record.get("source_video")
        path = record.get("source_path")
        if not (
            isinstance(file_name, str)
            and isinstance(name, str)
            and isinstance(path, str)
            and file_name
            and name
            and path
        ):
            continue
        for suffix in VIDEO_SUFFIXES:
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break
        if (path, name) in held:
            continue
        try:
            present = (output / file_name).is_file()
        except OSError:  # a file name the file system will not even look up
            present = False
        if present:
            held[(path, name)] = None
    return list(held)


_ILLEGAL_IN_A_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitise(part: str) -> str:
    """A folder name made safe to put inside a file name on any platform.

    Windows refuses ``<>:"/\\|?*`` and control characters in file names, while
    Linux and macOS allow most of them in a folder name. Without this, a
    folder called ``a:b`` would produce frame names that cannot be written on
    one of the machines the frames are shared with.
    """
    return _ILLEGAL_IN_A_NAME.sub("_", part)


def _folder_chain(path: PurePath) -> list[str]:
    """The names of the folders above ``path``, outermost first.

    The drive or share (``D:\\``, ``\\\\server\\share\\``) and the root are
    left out: the same experiment on two drives should not need the drive
    letter to tell its folders apart, and a letter says nothing about the
    recording.
    """
    parts = path.parent.parts
    return list(parts[1:] if path.anchor else parts)


class _Clip(NamedTuple):
    """One video as naming sees it."""

    path: str  # absolute, as the caller spelled it
    location: str  # where the file really is: two spellings of it agree
    folder: str  # where its folder really is, likewise
    stem: str


class _Survey(NamedTuple):
    """The clips a naming decision looks at, found once."""

    clip_of: dict[Path, _Clip]
    """Each video as given, mapped to its clip (one clip per location)."""
    new: dict[str, _Clip]
    """Location to clip, in location order, for clips with no recorded name."""
    held: dict[str, tuple[_Clip, str]]
    """Location to clip and recorded name, in path order."""
    taken: set[str]
    """Every recorded name, case-folded."""
    stems: dict[str, list[_Clip]]
    """Case-folded stem to the new and recorded clips that have it, by path."""


def _survey(videos: Sequence[Path], recorded: Sequence[tuple[str, str]]) -> _Survey:
    folders: dict[str, str] = {}

    def clip(path) -> _Clip:
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent not in folders:  # a long list repeats its folders
            folders[parent] = _location(parent)
        return _Clip(path, _location(path), folders[parent], Path(path).stem)

    # One file spelled two ways is one clip. It takes the spelling that sorts
    # first, so the answer does not depend on the order the videos came in.
    given = {video: clip(video) for video in videos}
    spelled: dict[str, _Clip] = {}
    for found in given.values():
        if found.location not in spelled or found.path < spelled[found.location].path:
            spelled[found.location] = found

    entries = sorted((os.path.abspath(path), name) for path, name in recorded)
    held: dict[str, tuple[_Clip, str]] = {}
    for path, name in entries:
        found = clip(path)
        held.setdefault(found.location, (found, name))

    new = {
        location: found
        for location, found in sorted(spelled.items())
        if location not in held
    }
    everyone = sorted(
        [*new.values(), *(found for found, _ in held.values())],
        key=lambda found: found.path,
    )
    stems: dict[str, list[_Clip]] = {}
    for found in everyone:
        stems.setdefault(found.stem.casefold(), []).append(found)
    return _Survey(
        clip_of={video: spelled[found.location] for video, found in given.items()},
        new=new,
        held=held,
        taken={name.casefold() for _, name in entries},
        stems=stems,
    )


def _first_elsewhere(clip: _Clip, stems: Mapping[str, list[_Clip]]) -> _Clip | None:
    """The first clip, by path, with this file name in a different folder."""
    group = stems.get(clip.stem.casefold(), ())
    return next((other for other in group if other.folder != clip.folder), None)


def _qualified(prefix: str, stem: str) -> str:
    return f"{prefix}-{stem}" if prefix else stem


def _choose_prefixes(
    folders: Sequence[str],
    members: Mapping[str, list[_Clip]],
    recorded_prefixes: Sequence[str],
    taken: set[str],
) -> dict[str, str]:
    """The last ``d`` folder names of each folder, for the smallest ``d`` that works.

    ``d`` works when the prefixes are different from each other and from every
    prefix already on record, and no name they make is already taken. One ``d``
    serves all the folders so the rule is easy to state, which means a deep
    collision elsewhere in the tree can lengthen an unrelated prefix. If no
    ``d`` works the deepest is used and numbering finishes the job.
    """
    # a folder reached by two spellings is named as its first clip spelled it
    chains = {
        folder: _folder_chain(Path(members[folder][0].path)) for folder in folders
    }
    deepest = max(map(len, chains.values()), default=0)
    on_record = {prefix.casefold() for prefix in recorded_prefixes}

    def at(depth: int) -> dict[str, str]:
        return {
            folder: "-".join(
                _sanitise(part) for part in chain[max(len(chain) - depth, 0) :]
            )
            for folder, chain in chains.items()
        }

    for depth in range(1, deepest + 1):
        prefixes = at(depth)
        folded = [prefix.casefold() for prefix in prefixes.values()]
        if len(set(folded)) < len(folded) or not on_record.isdisjoint(folded):
            continue
        if any(
            _qualified(prefixes[folder], clip.stem).casefold() in taken
            for folder in folders
            for clip in members[folder]
        ):
            continue
        return prefixes
    return at(deepest)


def assign_names(
    videos: Sequence[Path], recorded: Sequence[tuple[str, str]] = ()
) -> dict[Path, str]:
    """One unique name per video, built from its folders where file names clash.

    Frames are named after the clip and the train/validation split groups on
    that name, so two clips with one file name (``day1/cam1.mp4`` and
    ``day2/cam1.mp4``) would overwrite each other's frames and count as one
    recording. A clip keeps its file name unless its *folder* is involved in a
    clash; then every clip from that folder takes the folder's name as a prefix,
    so the pieces of one recording stay together in the split.

    ``recorded`` is what :func:`recorded_clips` reads from the output folder.
    Frames that exist are never renamed: a recorded clip keeps its name, found
    by location so a relative and an absolute path to one file agree, and a
    folder with qualified clips on record keeps the prefix they were given.

    1. A recorded clip keeps its recorded name.
    2. Every other clip is new. It is *contested* if its file name, case-folded,
       is the file name of a clip in another folder, new or recorded, or is a
       name already on record.
    3. The folders of contested clips are qualified, and so are folders with a
       prefix on record. Their new clips take the folder's prefix; others keep
       their file name.
    4. A new prefix is the last ``d`` folder names joined by ``-`` (see
       :func:`_choose_prefixes`). Folder names have characters Windows refuses
       replaced by ``_``; a file name is never altered.
    5. A name still taken gets ``-2``, ``-3``, ... in location order, the first
       keeping the bare name.

    Everything is compared case-folded, because ``Cam1`` and ``cam1`` collide on
    Windows and macOS, and clips are handled in location order, so the same clips
    always get the same names. Only path strings and the manifest are read: an
    unplugged drive changes nothing. The result is keyed by the paths as given.
    """
    survey = _survey(videos, recorded)
    new, held, taken = survey.new, survey.held, survey.taken

    on_record: dict[str, str] = {}  # folder -> the prefix its clips were given
    for clip, name in held.values():
        tail = f"-{clip.stem}"
        if len(name) > len(tail) and name.endswith(tail):
            on_record.setdefault(clip.folder, name[: -len(tail)])

    members: dict[str, list[_Clip]] = {}
    for clip in new.values():
        members.setdefault(clip.folder, []).append(clip)

    contested = {
        clip.folder
        for clip in new.values()
        if clip.stem.casefold() in taken or _first_elsewhere(clip, survey.stems)
    }
    prefixes = _choose_prefixes(
        sorted(folder for folder in contested if folder not in on_record),
        members,
        list(on_record.values()),
        taken,
    )
    prefixes.update({f: on_record[f] for f in members if f in on_record})

    used = set(taken)
    named: dict[str, str] = {}
    for location, clip in new.items():
        base = _qualified(prefixes.get(clip.folder, ""), clip.stem)
        name, number = base, 2
        while name.casefold() in used:
            name, number = f"{base}-{number}", number + 1
        used.add(name.casefold())
        named[location] = name

    wanted = {clip.location: clip for clip in survey.clip_of.values()}
    chosen = {
        location: held[location][1] if location in held else named[location]
        for location in wanted
    }
    users: dict[str, list[tuple[str, str]]] = {}  # folded name -> (name, path)
    for location, name in chosen.items():
        users.setdefault(name.casefold(), []).append((name, wanted[location].path))
    repeated = sorted(sorted(group) for group in users.values() if len(group) > 1)
    if repeated:  # only a manifest that already repeats a name can get here
        detail = "\n  ".join(
            f"{group[0][0]}: " + ", ".join(path for _, path in group)
            for group in repeated
        )
        raise SamplingError(
            "Could not give every video a different name, because this frames"
            " folder already records different videos under one name:\n  "
            + detail
            + "\nSample into a new frames folder, or delete the frames of one of"
            " these videos from this folder first."
        )
    return {video: chosen[clip.location] for video, clip in survey.clip_of.items()}


def describe_renames(
    videos: Sequence[Path],
    names: Mapping[Path, str],
    recorded: Sequence[tuple[str, str]] = (),
) -> list[str]:
    """One line for each clip whose name is not its file name, saying why.

    A clip is renamed because another clip shares its file name, so the line
    names that clip, which lets the user tell a move from a different video. A
    clip with no such partner is renamed because its folder was: the line names a
    clip in that folder that does have one. Nothing renamed gives ``[]``. A clip
    listed twice under different spellings is described once.
    """
    survey = _survey(videos, recorded)

    mates: dict[str, _Clip] = {}  # folder -> its first clip that shares a name
    for group in survey.stems.values():
        for clip in group:
            if _first_elsewhere(clip, survey.stems) and (
                clip.folder not in mates or clip.path < mates[clip.folder].path
            ):
                mates[clip.folder] = clip

    lines: list[str] = []
    seen: set[str] = set()
    for video in videos:
        clip = survey.clip_of[video]
        name = names[video]
        if clip.location in seen or name == video.stem:
            continue
        seen.add(clip.location)
        # a partner in another folder is why a folder is renamed; a twin in the
        # same folder (cam1.mp4 beside cam1.avi) only explains a number
        group = survey.stems.get(clip.stem.casefold(), [])
        other = _first_elsewhere(clip, survey.stems) or next(
            (found for found in group if found.location != clip.location), None
        )
        if other is not None:
            lines.append(f"{video} -> {name}  (shares a file name with {other.path})")
        elif clip.folder in mates:
            mate = mates[clip.folder]
            lines.append(
                f"{video} -> {name}  (same folder as {mate.path},"
                " which shares a file name)"
            )
        else:  # named after clips already on record, which are not shown here
            lines.append(f"{video} -> {name}")
    return lines


def repeated_stems(videos: Sequence[Path]) -> dict[str, list[Path]]:
    """Each file name shared by two or more videos, with the videos sharing it.

    Contour files are named ``<stem>.h5`` and paired with their video by that
    name, so these are the clips an export cannot keep apart. Names are compared
    the way the platform compares paths, as the exporter's own ``Path`` equality
    does: ignoring case on Windows, exactly elsewhere. (macOS also ignores case
    on disk but compares ``Path`` objects exactly, so ``Cam1`` and ``cam1`` go
    unreported there, here and in the exporter alike.) Each result is keyed by
    the spelling met first, with its videos in the order given.
    """
    by_stem: dict[str, list[Path]] = {}
    spelling: dict[str, str] = {}  # compared form -> first spelling met
    for video in videos:
        key = os.path.normcase(video.stem)
        by_stem.setdefault(spelling.setdefault(key, video.stem), []).append(video)
    return {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}


def stratified_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    """One frame from each of n_take equal blocks, at a random point within it."""
    if n_take >= n_frames:
        return list(range(n_frames))
    edges = [round(i * n_frames / n_take) for i in range(n_take + 1)]
    return [
        rng.randrange(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(n_take)
    ]


def uniform_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    if n_take >= n_frames:
        return list(range(n_frames))
    return sorted(rng.sample(range(n_frames), n_take))


def _fit_to_capacity(
    allocation: list[int], counts: Sequence[int], total: int
) -> list[int]:
    """No video is asked for more frames than it has, and nothing is lost.

    The shortfall goes round the remaining videos one frame at a time rather
    than onto whichever happens to sort first, so a short clip in the set does
    not quietly hand its share to a single neighbour.
    """
    alloc = [min(a, c) for a, c in zip(allocation, counts)]
    short = total - sum(alloc)
    while short > 0:
        room = [i for i, (a, c) in enumerate(zip(alloc, counts)) if a < c]
        if not room:
            break
        for i in room:
            if short == 0:
                break
            alloc[i] += 1
            short -= 1
    return alloc


def allocate(counts: Sequence[int], total: int, even: bool) -> list[int]:
    """Decides how many frames to take from each video.

    Proportional shares rarely come out whole, and how the leftover frames are
    handed out is the whole ball game when the budget is small: giving them all
    to the longest video means asking for 20 frames from 49 clips samples one
    clip twenty times and 48 clips not at all. So the leftovers go to the
    largest fractional parts, one each -- the standard largest-remainder
    apportionment -- which keeps equal-length clips within one frame of each
    other at any budget.
    """
    n_videos = len(counts)
    if n_videos == 0:
        raise SamplingError("No videos to sample from")

    if even:
        base, extra = divmod(total, n_videos)
        allocation = [base + (1 if i < extra else 0) for i in range(n_videos)]
        return _fit_to_capacity(allocation, counts, total)

    grand_total = sum(counts)
    if grand_total == 0:
        raise SamplingError("The videos contain no frames")

    exact = [total * c / grand_total for c in counts]
    allocation = [int(e) for e in exact]
    leftover = total - sum(allocation)
    if leftover > 0:
        # largest fractional part first, ties to the longer video
        order = sorted(
            range(n_videos),
            key=lambda i: (exact[i] - allocation[i], counts[i]),
            reverse=True,
        )
        for i in order[:leftover]:
            allocation[i] += 1

    return _fit_to_capacity(allocation, counts, total)


def _is_sidecar(path) -> bool:
    """Filesystem metadata that merely looks like a media file.

    macOS writes a "._name.mp4" companion beside every file it copies onto
    a non-Apple filesystem. It carries the same extension, so any *.mp4
    glob picks it up and OpenCV then reports it as an unreadable video.
    """
    name = Path(path).name
    return name.startswith("._") or name in (".DS_Store", "Thumbs.db", "desktop.ini")


def videos_in_folder(folder: Path, recursive: bool = False) -> list[Path]:
    """The videos in a folder, and in everything beneath it if ``recursive``.

    Sorted by path, so the same folder always yields the same list in the same
    order: the selection number only reproduces a selection if the videos it is
    applied to come in a fixed order. Folders whose name starts with a dot
    (``.git``, ``.Trash``, ...) are not entered, and macOS companion files are
    skipped, as for any other list of videos. Extensions match whatever their
    case, which a ``*.mp4`` glob does not on a case-sensitive filesystem.
    """
    found: list[Path] = []
    if recursive:
        for here, subfolders, names in os.walk(folder):
            subfolders[:] = [d for d in subfolders if not d.startswith(".")]
            found.extend(Path(here) / name for name in names)
    else:
        try:
            found = list(Path(folder).iterdir())
        except OSError:
            return []
    return sorted(
        p
        for p in found
        if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file() and not _is_sidecar(p)
    )


def _first_spellings(videos: Sequence[Path]) -> list[int]:
    """Positions of the videos to keep when one file is listed under two spellings.

    ``day1/cam1.avi`` and ``/data/day1/cam1.avi`` are one clip, so only the
    first mention stays. Without this the clip would be planned twice and take
    two shares of the frames.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for position, video in enumerate(videos):
        where = _location(video)
        if where not in seen:
            seen.add(where)
            keep.append(position)
    return keep


def resolve_videos(patterns: Sequence[Path], recursive: bool = False) -> list[Path]:
    """Expands globs and folders, and drops anything that is not a file.

    A folder contributes the videos directly inside it, or every video beneath
    it with ``recursive``; ``recursive`` changes nothing for a file or a glob,
    which already say exactly what they mean. Each video appears once, at its
    first mention, however many ways the file was spelled.
    """
    videos: list[Path] = []
    for pattern in patterns:
        pattern = Path(pattern)
        if pattern.is_dir():
            videos.extend(videos_in_folder(pattern, recursive))
        elif any(ch in str(pattern) for ch in "*?"):
            # glob.glob on the whole pattern, so wildcards in folder names work
            videos.extend(Path(m) for m in sorted(glob.glob(str(pattern))))
        else:
            videos.append(pattern)
    found = [v for v in videos if v.is_file() and not _is_sidecar(v)]
    return [found[position] for position in _first_spellings(found)]


def plan_sampling(
    videos: Sequence[Path],
    n_frames: int,
    seed: int = 0,
    even: bool = False,
    uniform_random: bool = False,
    counts: Sequence[int] | None = None,
    names: Mapping[Path, str] | None = None,
) -> list[VideoPlan]:
    """Works out what would be sampled, without touching a single frame.

    Cheap enough for the GUI to call on every spinbox change, so the user can
    see "600 frames: clip_01 -> 312, clip_02 -> 288" before committing --
    provided it passes ``counts``. Counting frames means opening every file,
    which is the expensive part and the part that does not change while a
    spinbox does, so the caller may pass counts it has already gathered.

    ``names`` is what :func:`assign_names` gave each video, so a caller that
    can see the frames folder's manifest shows the names a run would use. Left
    out, the names are assigned as for a folder that holds nothing yet.

    A clip that cannot be read gets a plan with ``readable=False`` and no
    frames, rather than bringing the whole run down.
    """
    if not videos:
        raise SamplingError("No videos matched")
    if names is None:
        names = assign_names(videos)

    if counts is None:
        counts = [frame_count_or_zero(v) for v in videos]
    elif len(counts) != len(videos):
        raise SamplingError("Frame counts do not line up with the videos")
    counts = [max(0, int(c)) for c in counts]

    if not any(counts):
        raise SamplingError(
            "None of the videos could be read: "
            + ", ".join(v.name for v in list(videos)[:5])
        )

    allocation = allocate(counts, n_frames, even)
    rng = random.Random(seed)
    pick = uniform_indices if uniform_random else stratified_indices

    plans = []
    for video, available, requested in zip(videos, counts, allocation):
        indices = pick(available, requested, rng) if requested else []
        plans.append(
            VideoPlan(
                video,
                available,
                requested,
                indices,
                readable=available > 0,
                name=names[video],
            )
        )
    return plans


def recorded_enhancement(output: Path) -> dict | None:
    """The enhancement settings a frames folder was already sampled with, if any."""
    profile = output / "preprocess_profile.json"
    try:
        if profile.is_file():
            return load_profile(profile)
        manifest = output / "sampling_manifest.json"
        if manifest.is_file():
            found = json.loads(manifest.read_text(encoding="utf-8")).get("enhancement")
            return found if isinstance(found, dict) and found else None
    except Exception:  # unreadable, or a PreprocessingError from a bad profile
        return None
    return None


def check_enhancement_unchanged(output: Path, settings: dict) -> None:
    """Refuses to mix frames enhanced differently into one folder.

    The profile beside the frames is overwritten by every run, so frames sampled
    under different settings would end up under a profile that describes only
    the last batch, and training would then record settings that are wrong for
    the rest.
    """
    recorded = recorded_enhancement(output)
    if not recorded:
        return
    differences = [
        f"{key}: folder has {recorded.get(key, DEFAULT_SETTINGS[key])!r}, now {settings[key]!r}"
        for key in SETTING_KEYS
        if key in settings and recorded.get(key, DEFAULT_SETTINGS[key]) != settings[key]
    ]
    if differences:
        raise SamplingError(
            f"{output} already holds frames enhanced differently from these"
            " settings:\n  "
            + "\n  ".join(differences)
            + "\nMixing them would make the folder's profile wrong for some"
            " frames. Sample into a new folder, run without enhancement flags to"
            " reuse the folder's own profile, or pass --force-new-enhancement"
            " to record the new settings anyway."
        )


def _merge_manifest(path: Path, manifest: dict, output: Path) -> dict:
    """Folds a new batch into whatever the folder already recorded.

    Sampling a second time into the same folder is an invited flow -- it is how
    an annotation corpus grows -- and overwriting the manifest orphaned every
    earlier frame. An orphaned frame falls back to having its source guessed
    from its name, which used to disagree with the recorded spelling, so one
    clip could end up on both sides of the train/validation split.

    Records whose image has since been deleted are dropped, so the manifest
    does not accumulate ghosts.
    """
    if not path.is_file():
        return manifest
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return manifest  # unreadable or not JSON: the new batch stands alone

    by_name: dict[str, dict] = {}
    for record in old.get("frames", []):
        name = record.get("file_name")
        if name and (output / name).is_file():
            by_name[name] = record
    for record in manifest["frames"]:
        by_name[record["file_name"]] = record

    merged = dict(manifest)
    merged["frames"] = sorted(by_name.values(), key=lambda r: r["file_name"])

    old_batches = old.get("batches")
    if not old_batches:
        # a manifest written before batches were recorded: keep what it said
        # about itself so the earlier frames' provenance is not simply lost
        old_batches = [
            {
                key: old[key]
                for key in (
                    "created",
                    "seed",
                    "sampling",
                    "allocation",
                    "complete",
                    "enhancement",
                    "videos",
                )
                if key in old
            }
        ]
    merged["batches"] = old_batches + manifest["batches"]

    videos = list(dict.fromkeys(old.get("videos", []) + manifest["videos"]))
    merged["videos"] = videos
    return merged


def sample_frames(
    videos: Sequence[Path],
    output: Path,
    n_frames: int,
    settings: dict,
    seed: int = 0,
    even: bool = False,
    uniform_random: bool = False,
    image_format: str = "png",
    progress: Callable[[int], None] | None = None,
    abort: Callable[[], bool] | None = None,
    counts: Sequence[int] | None = None,
    force_new_enhancement: bool = False,
) -> SamplingResult:
    """Writes enhanced frames for annotation, plus a manifest and a profile.

    The frames are written already enhanced, with the same settings inference
    will use, so that what gets annotated is what the model will see.

    An aborted run keeps what it has written and reports ``complete=False``.
    That is a deliberate departure from the background-computation thread this
    otherwise mirrors, which discards partial results: two hundred annotated
    frames are genuinely useful, an incomplete background model is not.

    Frames are named after the clip, from its folders where file names are shared
    (see :func:`assign_names`), against the clips the output folder already
    records. Frames that exist keep their names, and one file listed twice is
    sampled once. Two clips already recorded there under one name raise
    :class:`SamplingError` before anything is written.

    Sampling into a folder that already holds frames enhanced with different
    settings raises :class:`SamplingError` unless ``force_new_enhancement``.
    """
    videos = list(videos)
    if image_format not in ("png", "jpg"):
        raise SamplingError(f"Unsupported image format {image_format!r}")
    keep = _first_spellings(videos)
    if len(keep) < len(videos):  # one file listed twice is one clip
        if counts is not None and len(counts) == len(videos):
            counts = [counts[position] for position in keep]
        videos = [videos[position] for position in keep]
    names = assign_names(videos, recorded_clips(output))
    plans = plan_sampling(videos, n_frames, seed, even, uniform_random, counts, names)
    if not force_new_enhancement:
        check_enhancement_unchanged(output, settings)
    output.mkdir(parents=True, exist_ok=True)
    enhance = make_enhancer_from(settings)

    records: list[dict] = []
    # Every video gets an entry up front. An aborted run then reports zero for
    # the clips it never reached, instead of omitting them and reading as
    # though the collection were smaller than it is.
    per_video: dict[str, int] = {plan.name: 0 for plan in plans}
    unreadable: list[str] = [p.name for p in plans if not p.readable]
    complete = True

    result = SamplingResult(
        frames=records,
        requested=n_frames,
        output=output,
        per_video=per_video,
        unreadable=unreadable,
    )

    # The manifest is written in the finally, so frames already on disk stay
    # recorded when a write fails or the run is interrupted part-way.
    finished = False
    try:
        for plan in plans:
            if plan.requested == 0:
                continue

            cap = cv2.VideoCapture(str(plan.video))
            written = 0
            try:
                for index in plan.indices:
                    if abort is not None and abort():
                        complete = False
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        unreadable.append(f"{plan.name}:{index}")
                        continue

                    image = enhance(frame)
                    name = f"{plan.name}_f{index:06d}.{image_format}"
                    # imencode+tofile rather than imwrite, so non-ASCII paths work
                    encoded, buffer = cv2.imencode(f".{image_format}", image)
                    if not encoded:
                        raise SamplingError(
                            f"Could not encode frame {index} of {plan.video.name}"
                            f" as {image_format}"
                        )
                    buffer.tofile(str(output / name))
                    records.append(
                        {
                            "file_name": name,
                            # the name the frames carry, matching how the dataset
                            # step derives a source from a file name when no
                            # manifest covers it; storing the file name with its
                            # extension made one clip look like two different
                            # recordings
                            "source_video": plan.name,
                            # absolute, so a later run from another directory
                            # finds the clip again in recorded_clips
                            "source_path": os.path.abspath(plan.video),
                            "frame_number": index,
                            "width": image.shape[1],
                            "height": image.shape[0],
                        }
                    )
                    written += 1
                    if progress is not None:
                        progress(len(records))
            finally:
                cap.release()
                per_video[plan.name] = written
            if not complete:
                break
        finished = True
    finally:
        result.complete = complete and finished
        if records:
            _write_manifest(
                result, videos, settings, seed, even, uniform_random, image_format
            )
    return result


def _write_manifest(
    result: SamplingResult,
    videos: Sequence[Path],
    settings: dict,
    seed: int,
    even: bool,
    uniform_random: bool,
    image_format: str,
) -> None:
    output, records, complete = result.output, result.frames, result.complete
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    batch = {
        "created": created,
        "seed": seed,
        "sampling": "uniform-random" if uniform_random else "stratified",
        "allocation": "even" if even else "proportional",
        "complete": complete,
        "enhancement": settings,
        "image_format": image_format,
        "videos": [str(v) for v in videos],
    }
    for record in records:
        record["batch"] = created

    manifest = {
        "created": created,
        "seed": seed,
        "sampling": batch["sampling"],
        "allocation": batch["allocation"],
        "complete": complete,
        # kept at the top level because read_enhancement() looks for it here
        "enhancement": settings,
        "image_format": image_format,
        "videos": batch["videos"],
        "batches": [batch],
        "frames": records,
    }
    result.manifest_path = output / "sampling_manifest.json"
    manifest = _merge_manifest(result.manifest_path, manifest, output)
    result.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The settings live beside the frames as a reusable profile, so the same
    # recording setup can be prepared identically next time, and so the
    # conversion step can carry them on towards training. The image format is
    # recorded in the manifest only: profiles reject keys they do not know.
    result.profile_path = save_profile(
        output / "preprocess_profile.json",
        settings,
        name=output.name,
        notes="Written by the sampling step; reuse with --preprocess-profile",
    )
