"""
test_logger.py - Checks the event log holds only timestamps, desk ids, and states.
"""

import csv
from datetime import datetime

from logger import EventLogger, COLUMNS


# ---- Tests ----

def test_logs_one_row_per_state_change(tmp_path):
    log_path = tmp_path / "logs" / "events.csv"
    logger = EventLogger(log_path)

    logger.log("desk_1", "FREE", "OCCUPIED", when=datetime(2026, 9, 25, 20, 0, 0))
    logger.log("desk_1", "OCCUPIED", "RESERVED_EMPTY", when=datetime(2026, 9, 25, 20, 5, 0))

    rows = list(csv.reader(open(log_path)))
    assert rows[0] == COLUMNS
    assert rows[1] == ["2026-09-25T20:00:00", "desk_1", "FREE", "OCCUPIED"]
    assert rows[2] == ["2026-09-25T20:05:00", "desk_1", "OCCUPIED", "RESERVED_EMPTY"]
    assert all(len(row) == 4 for row in rows)   # nothing else sneaks in
