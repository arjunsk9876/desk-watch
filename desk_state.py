"""Desk state machine: FREE / OCCUPIED / RESERVED_EMPTY.

No camera or YOLO in here, just the rules, so it's easy to test.
"""

from collections import deque
from dataclasses import dataclass

FREE = "FREE"
OCCUPIED = "OCCUPIED"
RESERVED_EMPTY = "RESERVED_EMPTY"   # stuff is there but the person isn't


@dataclass
class DeskStatus:
    state: str
    seconds_in_state: float
    long: bool   # reserved-empty for longer than the limit


def next_state(state, person_present, item_present):
    # state table is in the README
    if state == FREE:
        if person_present:
            return OCCUPIED
    elif state == OCCUPIED:
        if not person_present and not item_present:
            return FREE             # left and took everything
        if not person_present and item_present:
            return RESERVED_EMPTY   # left but their stuff is still here
    elif state == RESERVED_EMPTY:
        if person_present:
            return OCCUPIED         # came back
        if not item_present:
            return FREE             # stuff got picked up
    return state


def mostly_true(history):
    # divide by maxlen, not len, so a single frame at startup can't win the vote
    return sum(history) > history.maxlen / 2


class DeskTracker:
    def __init__(self, smoothing_frames=10, long_reserved_minutes=60):
        self.smoothing_frames = smoothing_frames
        self.long_reserved_seconds = long_reserved_minutes * 60
        self.desks = {}   # desk_id -> state, when it started, recent frames

    def update(self, desk_id, person_present, item_present, now):
        if desk_id not in self.desks:
            self.desks[desk_id] = {"state": FREE, "since": now,
                                   "people": deque(maxlen=self.smoothing_frames),
                                   "items": deque(maxlen=self.smoothing_frames)}
        desk = self.desks[desk_id]

        # YOLO misses people for a frame here and there, so vote over the
        # last few frames instead of trusting one frame (stops flickering)
        desk["people"].append(person_present)
        desk["items"].append(item_present)
        new_state = next_state(desk["state"], mostly_true(desk["people"]),
                               mostly_true(desk["items"]))
        if new_state != desk["state"]:
            desk["state"], desk["since"] = new_state, now   # restart the timer

        seconds_in_state = now - desk["since"]
        is_long = new_state == RESERVED_EMPTY and seconds_in_state >= self.long_reserved_seconds
        return DeskStatus(new_state, seconds_in_state, is_long)
