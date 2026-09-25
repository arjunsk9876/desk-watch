"""
config.py - Loads desk-watch settings from a JSON file.

The settings file (settings.json by default) says which camera to use,
where each desk is in the camera image, and how long a desk can sit
"reserved but empty" before it counts as fair game.
calibrate.py writes this file; main.py reads it.
"""

import json
from pathlib import Path

DEFAULT_SETTINGS_PATH = "settings.json"

# Values used when the settings file leaves a field out.
DEFAULTS = {
    "camera_index": 0,
    "confidence": 0.25,
    "smoothing_frames": 10,
    "long_reserved_minutes": 30,
    "item_classes": ["backpack", "handbag", "suitcase", "laptop", "book"],
    "desks": [],
}


class SettingsError(Exception):
    """Raised when the settings file is missing or unusable."""


# ---- 1. Load settings ----

def load_settings(path=DEFAULT_SETTINGS_PATH):
    """Read the settings file and fill in defaults for anything missing."""
    path = Path(path)

    # A missing file almost always means calibration hasn't been done yet,
    # so tell the user exactly what to run instead of showing a stack trace.
    if not path.exists():
        raise SettingsError(
            f"Could not find '{path}'. "
            "Run 'python calibrate.py' first to draw your desk zones."
        )

    with open(path) as settings_file:
        loaded = json.load(settings_file)

    # Start from the defaults, then let the file override them.
    return {**DEFAULTS, **loaded}


# ---- 2. Save settings ----

def save_settings(settings, path=DEFAULT_SETTINGS_PATH):
    """Write settings back to disk (used by calibrate.py)."""
    with open(path, "w") as settings_file:
        json.dump(settings, settings_file, indent=2)
