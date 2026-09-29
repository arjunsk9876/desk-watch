"""Tests for the desk state machine. Uses fake detections and fake times, no camera."""

from desk_state import DeskTracker, FREE, OCCUPIED, RESERVED_EMPTY


def make_tracker(**options):
    # smoothing off so every frame counts
    options.setdefault("smoothing_frames", 1)
    return DeskTracker(**options)


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
    # nobody sat there first, so it was never reserved
    tracker = make_tracker()

    status = tracker.update("desk_1", person_present=False, item_present=True, now=0)

    assert status.state == FREE


def test_desks_are_tracked_separately():
    tracker = make_tracker()
    tracker.update("desk_1", person_present=True, item_present=False, now=0)

    status = tracker.update("desk_2", person_present=False, item_present=False, now=0)

    assert status.state == FREE


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


def test_person_in_3_of_10_frames_is_not_present():
    tracker = DeskTracker(smoothing_frames=10)
    frames = [True, False, False, True, False, False, False, True, False, False]

    for frame_number, person_seen in enumerate(frames):
        status = tracker.update("desk_1", person_seen, item_present=False, now=frame_number)
        assert status.state == FREE   # not even for one frame


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

    # YOLO misses them for 2 frames
    tracker.update("desk_1", person_present=False, item_present=True, now=10)
    status = tracker.update("desk_1", person_present=False, item_present=True, now=11)

    assert status.state == OCCUPIED
