"""
logger.py - Writes a line to logs/events.csv every time a desk changes state.

Each line is only: when it happened, which desk, the old state, the new state.
No images, no names, no detection details - so the log can't identify anyone.
"""

import csv
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_PATH = "logs/events.csv"
COLUMNS = ["timestamp", "desk_id", "old_state", "new_state"]


class EventLogger:
    # ---- 1. Create the log file (with a header row) if needed ----

    def __init__(self, path=DEFAULT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as log_file:
                csv.writer(log_file).writerow(COLUMNS)

    # ---- 2. Append one state change ----

    def log(self, desk_id, old_state, new_state, when=None):
        when = when or datetime.now()
        with open(self.path, "a", newline="") as log_file:
            csv.writer(log_file).writerow(
                [when.isoformat(timespec="seconds"), desk_id, old_state, new_state]
            )
