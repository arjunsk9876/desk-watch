"""
test_desk_state.py - Tests the 3-state machine with fake detections and fake times.

No camera or AI model is needed: we just tell the tracker "person yes/no,
item yes/no, time = X seconds" and check which state it lands in.
"""

from desk_state import DeskTracker, FREE, OCCUPIED, RESERVED_EMPTY


# ---- Helpers ----

def make_tracker(**options):
    """A tracker with smoothing turned off, so each frame counts on its own."""
    options.setdefault("smoothing_frames", 1)
    return DeskTracker(**options)


# ---- Normal transitions ----

def test_person_sits_at_free_desk():
    tracker = make_tracker()
    assert tracker.update("desk_1", False, False, now=0).state == FREE

    status = tracker.update("desk_1", person_present=True, item_present=False, now=1)

    assert status.state == OCCUPIED


def test_person_leaves_with_everything():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=True, now=0)

    status = tracker.update("desk_1", person_present=False, item_present=False, now=5)

    assert status.state == FREE


def test_person_leaves_item_behind():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=True, now=0)

    status = tracker.update("desk_1", person_present=False, item_present=True, now=5)

    assert status.state == RESERVED_EMPTY
    assert status.seconds_in_state == 0   # the timer starts when they leave
    assert status.long is False


def test_person_comes_back_resets_timer():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=True, now=0)
    tracker.update("desk_1", person_present=False, item_present=True, now=5)
    assert tracker.update("desk_1", False, True, now=65).seconds_in_state == 60

    status = tracker.update("desk_1", person_present=True, item_present=True, now=70)

    assert status.state == OCCUPIED
    assert status.seconds_in_state == 0


def test_item_removed_while_reserved():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=True, now=0)
    tracker.update("desk_1", person_present=False, item_present=True, now=5)

    status = tracker.update("desk_1", person_present=False, item_present=False, now=20)

    assert status.state == FREE


def test_item_alone_on_free_desk_stays_free():
    # Nobody "reserved" it - an item with no owner seen stays FREE.
    tracker = make_tracker()

    status = tracker.update("desk_1", person_present=False, item_present=True, now=0)

    assert status.state == FREE


def test_desks_are_tracked_separately():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=False, now=0)

    status = tracker.update("desk_2", person_present=False, item_present=False, now=0)

    assert status.state == FREE


# ---- Long reserved flag ----

def test_reserved_too_long_gets_flagged():
    tracker = make_tracker(long_reserved_minutes=2)
    tracker.update("desk_1", person_present=True, item_present=True, now=0)
    tracker.update("desk_1", person_present=False, item_present=True, now=10)

    almost = tracker.update("desk_1", person_present=False, item_present=True, now=10 + 119)
    too_long = tracker.update("desk_1", person_present=False, item_present=True, now=10 + 120)

    assert almost.long is False
    assert too_long.long is True
    assert too_long.state == RESERVED_EMPTY   # still reserved, just flagged


def test_coming_back_clears_long_flag():
    tracker = make_tracker(long_reserved_minutes=2)
    tracker.update("desk_1", person_present=True, item_present=True, now=0)
    tracker.update("desk_1", person_present=False, item_present=True, now=10)
    tracker.update("desk_1", person_present=False, item_present=True, now=500)

    status = tracker.update("desk_1", person_present=True, item_present=True, now=510)

    assert status.state == OCCUPIED
    assert status.long is False


# ---- Flicker smoothing ----

def test_person_in_3_of_10_frames_is_not_present():
    tracker = DeskTracker(smoothing_frames=10)
    frames = [True, False, False, True, False, False, False, True, False, False]

    for frame_number, person_seen in enumerate(frames):
        status = tracker.update("desk_1", person_seen, item_present=False, now=frame_number)
        assert status.state == FREE   # never flips to OCCUPIED, not even for one frame


def test_person_in_6_of_10_frames_is_present():
    tracker = DeskTracker(smoothing_frames=10)
    frames = [True, False, True, True, False, True, False, True, True, False]

    for frame_number, person_seen in enumerate(frames):
        status = tracker.update("desk_1", person_seen, item_present=False, now=frame_number)

    assert status.state == OCCUPIED


def test_two_missed_frames_do_not_end_occupied():
    tracker = DeskTracker(smoothing_frames=10)
    for frame_number in range(10):
        tracker.update("desk_1", person_present=True, item_present=True, now=frame_number)

    # The detector misses the person for two frames in a row.
    tracker.update("desk_1", person_present=False, item_present=True, now=10)
    status = tracker.update("desk_1", person_present=False, item_present=True, now=11)

    assert status.state == OCCUPIED
