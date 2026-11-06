"""Flagging the Miss id gaps nobody else is near, and filling them in one go."""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

# Importing any part of the Validator package imports PyTorch, and CI installs
# without it. gaps.py is plain NumPy, so it is loaded by path and its tests run
# everywhere; the tests that need the widgets are skipped where PyTorch is
# missing.
GAPS_PATH = (
    Path(__file__).parent.parent
    / "src/idtrackerai/extra_tools/validator/widgets/gaps.py"
)
_spec = importlib.util.spec_from_file_location("validator_gaps", GAPS_PATH)
gaps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaps)
find_gap_anchors = gaps.find_gap_anchors
gap_is_unambiguous = gaps.gap_is_unambiguous
interpolate_gap = gaps.interpolate_gap

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="the Validator package imports PyTorch"
)
if HAS_TORCH:
    from idtrackerai.extra_tools.validator.widgets.errors_explorer import (  # noqa: E402
        FLAG_COLUMN,
        ErrorsExplorer,
    )
    from idtrackerai.extra_tools.validator.widgets.interpolator import (  # noqa: E402
        Interpolator,
    )

app = QApplication.instance() or QApplication([])

N_FRAMES = 100


@pytest.fixture
def trajectories() -> np.ndarray:
    """Three animals over 100 frames, with three gaps.

    Identity 1 walks along y=0 and has a gap in frames 20-29 that nobody else
    is near, and another in frames 0-4 that has nothing before it to draw a line
    from. Identity 2 sits still at (65, 200) and has a gap in frames 60-69, in
    which identity 3 passes 5 px from it. Identity 3 is otherwise far away.
    """
    traj = np.empty((N_FRAMES, 3, 2))
    frames = np.arange(N_FRAMES)
    traj[:, 0] = np.stack([frames, np.zeros(N_FRAMES)], axis=1)
    traj[:, 1] = (65, 200)
    traj[:, 2] = (65, -300)
    traj[60:70, 2] = (65, 205)
    traj[0:5, 0] = np.nan
    traj[20:30, 0] = np.nan
    traj[60:70, 1] = np.nan
    return traj


def explorer_for(traj: np.ndarray) -> "ErrorsExplorer":
    explorer = ErrorsExplorer()
    explorer.set_references(
        traj,
        np.zeros(len(traj), bool),
        np.zeros(traj.shape[:2], bool),
        None,  # only read when a row is clicked
        None,
    )
    return explorer


def row_of(explorer: "ErrorsExplorer", gap: tuple[int, int, int]) -> int:
    for row in range(explorer.table.rowCount()):
        if explorer.miss_id_gap(row) == gap:
            return row
    raise AssertionError(f"{gap} is not in the list")


class TestGaps:
    def test_anchors_are_the_nearest_positions_on_each_side(self, trajectories):
        assert find_gap_anchors(trajectories, 0, 20, 10) == (19, 30)

    def test_anchors_reach_past_a_neighbouring_gap(self, trajectories):
        trajectories[30:33, 0] = np.nan
        assert find_gap_anchors(trajectories, 0, 20, 10) == (19, 33)

    def test_no_anchor_before_the_start_of_the_video(self, trajectories):
        assert find_gap_anchors(trajectories, 0, 0, 5) is None
        assert interpolate_gap(trajectories, 0, 0, 5) is None

    def test_no_anchor_after_the_end_of_the_video(self, trajectories):
        trajectories[90:, 1] = np.nan
        assert find_gap_anchors(trajectories, 1, 90, 10) is None

    def test_the_line_is_straight_and_hits_every_frame_of_the_gap(self, trajectories):
        path = interpolate_gap(trajectories, 0, 20, 10)
        assert path is not None and path.shape == (10, 2)
        np.testing.assert_allclose(path[:, 0], np.arange(20, 30))
        np.testing.assert_allclose(path[:, 1], 0)

    def test_the_line_is_not_sampled_at_unit_steps_when_anchors_are_far(self):
        traj = np.full((11, 1, 2), np.nan)
        traj[0, 0] = (0, 0)
        traj[10, 0] = (100, 50)
        path = interpolate_gap(traj, 0, 1, 9)
        np.testing.assert_allclose(path[:, 0], np.arange(1, 10) * 10)
        np.testing.assert_allclose(path[:, 1], np.arange(1, 10) * 5)

    def test_a_gap_nobody_is_near_is_unambiguous(self, trajectories):
        assert gap_is_unambiguous(trajectories, 0, 20, 10, 50)

    def test_a_gap_with_a_neighbour_in_range_is_not(self, trajectories):
        assert not gap_is_unambiguous(trajectories, 1, 60, 10, 50)

    def test_the_threshold_decides_it(self, trajectories):
        # identity 3 is 5 px from the line
        assert gap_is_unambiguous(trajectories, 1, 60, 10, 3)
        assert not gap_is_unambiguous(trajectories, 1, 60, 10, 6)

    def test_one_close_frame_is_enough(self, trajectories):
        trajectories[60:69, 2] = (65, -300)  # far in all but the last frame
        assert not gap_is_unambiguous(trajectories, 1, 60, 10, 50)

    def test_neighbours_with_no_position_are_ignored(self, trajectories):
        trajectories[60:70, 2] = np.nan
        assert gap_is_unambiguous(trajectories, 1, 60, 10, 50)

    def test_a_gap_that_cannot_be_filled_is_never_unambiguous(self, trajectories):
        assert not gap_is_unambiguous(trajectories, 0, 0, 5, 50)

    def test_a_lone_animal_has_nobody_to_be_confused_with(self):
        traj = np.full((10, 1, 2), np.nan)
        traj[0, 0] = traj[9, 0] = (0, 0)
        assert gap_is_unambiguous(traj, 0, 1, 8, 50)


@needs_torch
class TestErrorsExplorer:
    def test_only_miss_id_rows_have_something_to_check(self, trajectories):
        explorer = explorer_for(trajectories)
        kinds = set()
        for row in range(explorer.table.rowCount()):
            kind = explorer.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            kinds.add(kind)
            checkable = bool(
                explorer.table.item(row, FLAG_COLUMN).flags()
                & Qt.ItemFlag.ItemIsUserCheckable
            )
            assert checkable == (kind == "Miss id")
        assert {"Miss id", "Jump"} <= kinds

    def test_flagging_checks_exactly_the_unambiguous_gaps(self, trajectories):
        explorer = explorer_for(trajectories)
        explorer.flag_unambiguous_btn.click()
        # the gap at the start has no anchor, the one beside identity 3 is close
        assert explorer.flagged_gaps() == [(1, 20, 10)]
        assert "1 of 3" in explorer.flag_summary.text()

    def test_vicinity_is_what_the_button_uses(self, trajectories):
        explorer = explorer_for(trajectories)
        explorer.vicinity_threshold.setValue(3)
        explorer.flag_unambiguous_btn.click()
        assert sorted(explorer.flagged_gaps()) == [(1, 20, 10), (2, 60, 10)]

    @pytest.mark.parametrize(
        "order", [Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder]
    )
    def test_flagging_while_the_list_is_sorted_by_flag_misses_nothing(
        self, trajectories, order
    ):
        # The list can be sorted by the very column being ticked, and every gap
        # must still be visited exactly once.
        explorer = explorer_for(trajectories)
        explorer.vicinity_threshold.setValue(3)
        explorer.table.sortByColumn(FLAG_COLUMN, order)
        explorer.flag_unambiguous_btn.click()
        assert sorted(explorer.flagged_gaps()) == [(1, 20, 10), (2, 60, 10)]
        assert explorer.table.isSortingEnabled()

    def test_flagging_never_clears_a_flag_set_by_hand(self, trajectories):
        explorer = explorer_for(trajectories)
        by_hand = row_of(explorer, (2, 60, 10))
        explorer.table.item(by_hand, FLAG_COLUMN).setCheckState(Qt.CheckState.Checked)
        explorer.flag_unambiguous_btn.click()
        assert sorted(explorer.flagged_gaps()) == [(1, 20, 10), (2, 60, 10)]

    def test_rebuilding_the_list_drops_the_flags(self, trajectories):
        explorer = explorer_for(trajectories)
        explorer.flag_unambiguous_btn.click()
        explorer.update_list_of_errors()
        assert explorer.flagged_gaps() == []

    def test_interpolating_sends_the_flagged_gaps(self, trajectories):
        explorer = explorer_for(trajectories)
        sent = []
        explorer.interpolate_all_flagged.connect(sent.append)
        explorer.flag_unambiguous_btn.click()
        explorer.interpolate_flagged_btn.click()
        assert sent == [[(1, 20, 10)]]

    def test_interpolating_with_nothing_flagged_sends_nothing(self, trajectories):
        explorer = explorer_for(trajectories)
        sent = []
        explorer.interpolate_all_flagged.connect(sent.append)
        explorer.interpolate_flagged_btn.click()
        assert sent == []
        assert explorer.flag_summary.text() == "No gap is flagged"

    def test_a_gap_flagged_by_hand_with_no_anchor_is_skipped_and_said_so(
        self, trajectories
    ):
        explorer = explorer_for(trajectories)
        sent = []
        explorer.interpolate_all_flagged.connect(sent.append)
        explorer.flag_unambiguous_btn.click()
        at_start = row_of(explorer, (1, 0, 5))
        explorer.table.item(at_start, FLAG_COLUMN).setCheckState(Qt.CheckState.Checked)
        explorer.interpolate_flagged_btn.click()
        assert sent == [[(1, 20, 10)]]
        assert "skipped 1" in explorer.flag_summary.text()

    def test_nothing_is_sent_when_no_flagged_gap_can_be_filled(self, trajectories):
        explorer = explorer_for(trajectories)
        sent = []
        explorer.interpolate_all_flagged.connect(sent.append)
        at_start = row_of(explorer, (1, 0, 5))
        explorer.table.item(at_start, FLAG_COLUMN).setCheckState(Qt.CheckState.Checked)
        explorer.interpolate_flagged_btn.click()
        assert sent == []
        assert "both sides" in explorer.flag_summary.text()


class FakeBlobs:
    def __init__(self):
        self.added: list[tuple[int, int, np.ndarray]] = []

    def add_centroid(self, frame, identity, centroid):
        self.added.append((frame, identity, np.asarray(centroid)))


class FakeUndo:
    def __init__(self):
        self.pushed: list[tuple[str, list[int]]] = []

    def push_frames(self, description, frames):
        self.pushed.append((description, list(frames)))


@needs_torch
class TestBatchInterpolate:
    @pytest.fixture
    def interpolator(self, trajectories):
        interpolator = Interpolator()
        interpolator.blobs = FakeBlobs()
        interpolator.undo_manager = FakeUndo()
        interpolator.set_references(
            trajectories,
            np.zeros(N_FRAMES, bool),
            np.zeros(trajectories.shape[:2], bool),
            interpolator.blobs,
            None,
        )
        interpolator.updates = []
        interpolator.update_trajectories.connect(
            lambda *args: interpolator.updates.append(args)
        )
        interpolator.accepted = []
        interpolator.interpolation_accepted.connect(
            lambda: interpolator.accepted.append(True)
        )
        return interpolator

    def test_every_frame_of_the_gap_gets_a_point_on_the_line(self, interpolator):
        interpolator.batch_interpolate([(1, 20, 10)])
        added = interpolator.blobs.added
        assert [frame for frame, _, _ in added] == list(range(20, 30))
        assert {identity for _, identity, _ in added} == {1}
        for frame, _, centroid in added:
            np.testing.assert_allclose(centroid, (frame, 0))

    def test_several_gaps_are_one_undo_step_and_one_recompute(self, interpolator):
        interpolator.batch_interpolate([(2, 60, 10), (1, 20, 10)])
        assert len(interpolator.blobs.added) == 20
        assert interpolator.undo_manager.pushed == [
            ("Interpolate 2 flagged gaps", list(range(20, 30)) + list(range(60, 70)))
        ]
        assert interpolator.updates == [(20, 70, True)]

    def test_the_undo_snapshot_is_taken_before_anything_is_added(self, interpolator):
        order = []
        interpolator.undo_manager.push_frames = lambda *_: order.append("undo")
        interpolator.blobs.add_centroid = lambda *_: order.append("add")
        interpolator.batch_interpolate([(1, 20, 10)])
        assert order[0] == "undo" and set(order[1:]) == {"add"}

    def test_a_gap_without_anchors_is_left_alone(self, interpolator):
        interpolator.batch_interpolate([(1, 0, 5), (1, 20, 10)])
        assert [frame for frame, _, _ in interpolator.blobs.added] == list(
            range(20, 30)
        )

    def test_nothing_fillable_changes_nothing(self, interpolator):
        interpolator.batch_interpolate([(1, 0, 5)])
        assert interpolator.blobs.added == []
        assert interpolator.undo_manager.pushed == []
        assert interpolator.updates == []

    def test_a_gap_listed_twice_is_filled_once(self, interpolator):
        interpolator.batch_interpolate([(1, 20, 10), (1, 20, 10)])
        assert len(interpolator.blobs.added) == 10

    def test_it_does_not_tell_the_list_that_a_jump_was_corrected(self, interpolator):
        interpolator.batch_interpolate([(1, 20, 10)])
        assert interpolator.accepted == []

    def test_it_works_without_an_undo_manager(self, interpolator):
        interpolator.undo_manager = None
        interpolator.batch_interpolate([(1, 20, 10)])
        assert len(interpolator.blobs.added) == 10


@needs_torch
class TestInsideTheValidator:
    """The real Validator, with real blobs and the real Ctrl+Z, on a blank window."""

    @pytest.fixture
    def gui(self, trajectories, monkeypatch):
        from idtrackerai import ListOfBlobs
        from idtrackerai.extra_tools.validator.validation_GUI import ValidationGUI

        gui = ValidationGUI()
        gui.autosave_timer.stop()
        # The pop-up is modal and waits for a key press, which never comes here.
        monkeypatch.setattr(gui.light_popup, "info", lambda *args: None)
        blobs = ListOfBlobs([[] for _ in range(N_FRAMES)])
        for frame in range(N_FRAMES):
            for animal in range(trajectories.shape[1]):
                if not np.isnan(trajectories[frame, animal, 0]):
                    blobs.add_blob(
                        frame, tuple(trajectories[frame, animal]), animal + 1
                    )
        gui.blobs = blobs
        gui.n_frames = N_FRAMES
        gui.n_animals = trajectories.shape[1]
        gui.trajectories = np.full(trajectories.shape, np.nan)
        gui.duplicated = np.zeros(trajectories.shape[:2], bool)
        gui.unidentified = np.zeros(N_FRAMES, bool)
        for widget in (gui.interpolator, gui.errorsExplorer):
            args = (gui.trajectories, gui.unidentified, gui.duplicated, blobs, None)
            widget.set_references(*args)
        gui.update_trajectories_range(0, N_FRAMES)
        # a blank Validator keeps its panels disabled until a session is opened
        gui.centralWidget().setEnabled(True)
        yield gui
        gui.deleteLater()

    def gap_rows(self, gui):
        explorer = gui.errorsExplorer
        return sorted(
            gap
            for row in range(explorer.table.rowCount())
            if (gap := explorer.miss_id_gap(row)) is not None
        )

    def test_the_buttons_fill_the_gap_and_ctrl_z_brings_it_back(
        self, gui, trajectories
    ):
        assert np.isnan(gui.trajectories[20:30, 0]).all()
        assert self.gap_rows(gui) == [(1, 0, 5), (1, 20, 10), (2, 60, 10)]

        gui.errorsExplorer.flag_unambiguous_btn.click()
        gui.errorsExplorer.interpolate_flagged_btn.click()

        np.testing.assert_allclose(
            gui.trajectories[20:30, 0], np.stack([np.arange(20, 30), np.zeros(10)], 1)
        )
        assert self.gap_rows(gui) == [(1, 0, 5), (2, 60, 10)]
        # the gap that was judged ambiguous is untouched
        assert np.isnan(gui.trajectories[60:70, 1]).all()
        assert len(gui.undo_manager.stack) == 1

        gui.undo_manager.undo()

        assert np.isnan(gui.trajectories[20:30, 0]).all()
        assert self.gap_rows(gui) == [(1, 0, 5), (1, 20, 10), (2, 60, 10)]
        np.testing.assert_array_equal(gui.trajectories, trajectories)

    def test_the_summary_survives_the_list_being_rebuilt(self, gui):
        gui.errorsExplorer.flag_unambiguous_btn.click()
        gui.errorsExplorer.interpolate_flagged_btn.click()
        assert "Interpolated 1 flagged gaps" in gui.errorsExplorer.flag_summary.text()
