"""
desk_state.py - The brain of desk-watch: a 3-state machine for each desk.

Every frame, each desk gets two yes/no answers from the detector:
  person_present - is a person in this desk's zone?
  item_present   - is a bag/laptop/etc. in this desk's zone?
This file turns those answers into one of three states: FREE, OCCUPIED,
or RESERVED_EMPTY. It is pure Python - no camera, no AI model - so it can
be tested with made-up inputs and made-up clock times.
"""

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
            return FREE          # they left and took everything with them
        if not person_present and item_present:
            return RESERVED_EMPTY   # they left, but their stuff is still here

    elif state == RESERVED_EMPTY:
        if person_present:
            return OCCUPIED      # they came back

    return state


# ---- 4. Keep track of every desk over time ----

class DeskTracker:
    """Remembers each desk's state between frames and times how long it has lasted."""

    def __init__(self, smoothing_frames=10, long_reserved_minutes=30):
        self.smoothing_frames = smoothing_frames
        self.long_reserved_seconds = long_reserved_minutes * 60
        self.desks = {}   # desk_id -> {"state": ..., "since": ...}

    def update(self, desk_id, person_present, item_present, now):
        """Feed in one frame's detections for one desk; get back its DeskStatus."""
        # Every desk starts FREE the first time we see it.
        desk = self.desks.setdefault(desk_id, {"state": FREE, "since": now})

        new_state = next_state(desk["state"], person_present, item_present)
        if new_state != desk["state"]:
            desk["state"] = new_state
            desk["since"] = now   # entering a new state restarts the timer

        return DeskStatus(
            state=desk["state"],
            seconds_in_state=now - desk["since"],
            long=False,
        )
