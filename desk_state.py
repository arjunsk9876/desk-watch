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
