"""Logs desk state changes to logs/events.csv.

Only time, desk, old state, new state - nothing that could identify someone.
"""

import csv
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_PATH = "logs/events.csv"
COLUMNS = ["timestamp", "desk_id", "old_state", "new_state"]


class EventLogger:
    def __init__(self, path=DEFAULT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as log_file:
                csv.writer(log_file).writerow(COLUMNS)

    def log(self, desk_id, old_state, new_state, when=None):
        when = when or datetime.now()
        with open(self.path, "a", newline="") as log_file:
            csv.writer(log_file).writerow(
                [when.isoformat(timespec="seconds"), desk_id, old_state, new_state]
            )
