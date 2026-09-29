"""Load and save settings.json (camera, desk zones, timers)."""

import json
from pathlib import Path

DEFAULT_SETTINGS_PATH = "settings.json"

# used when settings.json doesn't have a value
DEFAULTS = {
    "camera_index": 0,
    "confidence": 0.25,
    "item_confidence": 0.15,
    "smoothing_frames": 10,
    "long_reserved_minutes": 60,
    "item_classes": ["backpack", "laptop", "cell phone", "book", "bottle", "cup"],
    "desks": [],
}


class SettingsError(Exception):
    pass


def load_settings(path=DEFAULT_SETTINGS_PATH):
    path = Path(path)

    # no file usually means calibrate.py hasn't been run yet
    if not path.exists():
        raise SettingsError(
            f"Could not find '{path}'. "
            "Run 'python calibrate.py' first to draw your desk zones."
        )

    with open(path) as settings_file:
        loaded = json.load(settings_file)

    return {**DEFAULTS, **loaded}


def save_settings(settings, path=DEFAULT_SETTINGS_PATH):
    with open(path, "w") as settings_file:
        json.dump(settings, settings_file, indent=2)
