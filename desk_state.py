"""
desk_state.py - The brain of desk-watch: a 3-state machine for each desk.

Each frame, the detector answers two yes/no questions per desk (is a person
here? is an item here?). This file turns those answers into FREE, OCCUPIED,
or RESERVED_EMPTY. Pure Python - no camera, no AI - so it is easy to test.
"""

from collections import deque
from dataclasses import dataclass

# ---- 1. The three states ----

FREE = "FREE"                        # nothing on the desk
OCCUPIED = "OCCUPIED"                # a person is sitting there
RESERVED_EMPTY = "RESERVED_EMPTY"    # their stuff is there, they are not


# ---- 2. What we report for each desk ----

@dataclass
class DeskStatus:
    state: str                # FREE, OCCUPIED, or RESERVED_EMPTY
    seconds_in_state: float   # how long the desk has been in this state
    long: bool                # True once a desk has been reserved-but-empty too long


# ---- 3. The rules: which state comes next? ----

def next_state(state, person_present, item_present):
    """Apply one row of the state table (see README) and return the new state."""
    if state == FREE:
        if person_present:
            return OCCUPIED
    elif state == OCCUPIED:
        if not person_present and not item_present:
            return FREE             # they left and took everything with them
        if not person_present and item_present:
            return RESERVED_EMPTY   # they left, but their stuff is still here
    elif state == RESERVED_EMPTY:
        if person_present:
            return OCCUPIED         # they came back
        if not item_present:
            return FREE             # the item was taken and nobody sat down
    return state


def mostly_true(history):
    """True if more than half of the last `maxlen` frames said yes (missing frames count as no)."""
    return sum(history) > history.maxlen / 2


# ---- 4. Keep track of every desk over time ----

class DeskTracker:
    """Remembers each desk's state between frames and times how long it has lasted."""

    def __init__(self, smoothing_frames=10, long_reserved_minutes=30):
        self.smoothing_frames = smoothing_frames
        self.long_reserved_seconds = long_reserved_minutes * 60
        self.desks = {}   # desk_id -> its state, timer start, and recent detections

    def update(self, desk_id, person_present, item_present, now):
        """Feed in one frame's detections for one desk; get back its DeskStatus."""
        if desk_id not in self.desks:   # every desk starts FREE
            self.desks[desk_id] = {"state": FREE, "since": now,
                                   "people": deque(maxlen=self.smoothing_frames),
                                   "items": deque(maxlen=self.smoothing_frames)}
        desk = self.desks[desk_id]

        # The detector sometimes misses a person for a frame or two. Voting
        # over the last few frames stops the desk from flickering between states.
        desk["people"].append(person_present)
        desk["items"].append(item_present)
        new_state = next_state(desk["state"], mostly_true(desk["people"]),
                               mostly_true(desk["items"]))
        if new_state != desk["state"]:
            desk["state"], desk["since"] = new_state, now   # new state restarts the timer

        # After long_reserved_minutes, flag it so the dashboard can say "fair game".
        seconds_in_state = now - desk["since"]
        is_long = new_state == RESERVED_EMPTY and seconds_in_state >= self.long_reserved_seconds
        return DeskStatus(new_state, seconds_in_state, is_long)
