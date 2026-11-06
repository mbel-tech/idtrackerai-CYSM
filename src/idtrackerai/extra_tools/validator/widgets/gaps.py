"""Straight-line fills for gaps in one identity's trajectory.

Pure NumPy, no Qt: the error list uses it to decide which gaps are safe to fill
automatically, and the interpolator uses it to fill them, so both have to agree
on what "the line across the gap" is.

A gap is `length` consecutive frames starting at `start` in which identity
`animal_idx` (0-based) has no position, the same shape the "Miss id" errors have.
`trajectories` is `(n_frames, n_animals, 2)` with NaN where there is no position.
"""

import numpy as np


def find_gap_anchors(
    trajectories: np.ndarray, animal_idx: int, start: int, length: int
) -> tuple[int, int] | None:
    """Frames of the last position before the gap and the first one after it.

    These are the two points the straight line runs between. They are searched
    for beyond the gap, so a gap that touches a tracking-interval boundary
    still finds them. None when the animal has no position on one of the sides,
    which leaves nothing to draw the line from.
    """
    present = ~np.isnan(trajectories[:, animal_idx, 0])
    before = np.flatnonzero(present[:start])
    end = start + length
    after = np.flatnonzero(present[end:])
    if len(before) == 0 or len(after) == 0:
        return None
    return int(before[-1]), int(end + after[0])


def interpolate_gap(
    trajectories: np.ndarray, animal_idx: int, start: int, length: int
) -> np.ndarray | None:
    """Positions of the straight line across the gap, `(length, 2)`.

    None when the gap has no anchors on both sides.
    """
    anchors = find_gap_anchors(trajectories, animal_idx, start, length)
    if anchors is None:
        return None
    frame_before, frame_after = anchors
    position_before = trajectories[frame_before, animal_idx]
    position_after = trajectories[frame_after, animal_idx]
    t = (np.arange(start, start + length) - frame_before) / (frame_after - frame_before)
    return position_before + t[:, None] * (position_after - position_before)


def gap_is_unambiguous(
    trajectories: np.ndarray, animal_idx: int, start: int, length: int, threshold: float
) -> bool:
    """Whether no other animal comes within `threshold` of the line across the gap.

    A straight line is a sound guess for where an animal went only if nobody
    else could have been mistaken for it along the way. Every frame of the gap
    is checked, against every other animal that has a position in that frame.
    A gap with no anchors is never unambiguous, because it cannot be filled.
    """
    path = interpolate_gap(trajectories, animal_idx, start, length)
    if path is None:
        return False

    others = np.delete(trajectories[start : start + length], animal_idx, axis=1)
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(others - path[:, None, :], axis=-1)
    # A missing position gives a NaN distance, which is never less than anything.
    return not bool(np.any(distance < threshold))
