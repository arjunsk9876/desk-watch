"""
test_config.py - Checks that config.py reads settings files correctly.
"""

import json

import pytest

from config import load_settings, SettingsError


# ---- Helpers ----

def write_settings(folder, data):
    """Save a settings dict as JSON in a temporary folder and return its path."""
    path = folder / "settings.json"
    path.write_text(json.dumps(data))
    return path


# ---- Tests ----

def test_loads_two_desks(tmp_path):
    path = write_settings(tmp_path, {
        "camera_index": 1,
        "desks": [
            {"id": "desk_1", "name": "Desk 1", "zone": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "desk_2", "name": "Desk 2", "zone": [[20, 0], [30, 0], [30, 10], [20, 10]]},
        ],
    })

    settings = load_settings(path)

    assert settings["camera_index"] == 1
    assert [desk["id"] for desk in settings["desks"]] == ["desk_1", "desk_2"]
    assert settings["desks"][1]["zone"][2] == [30, 10]


def test_missing_fields_use_defaults(tmp_path):
    path = write_settings(tmp_path, {
        "desks": [{"id": "desk_1", "name": "Desk 1", "zone": [[0, 0], [10, 0], [10, 10]]}],
    })

    settings = load_settings(path)

    assert settings["smoothing_frames"] == 10
    assert settings["long_reserved_minutes"] == 60
    assert settings["confidence"] == 0.25


def test_missing_file_tells_user_to_calibrate(tmp_path):
    with pytest.raises(SettingsError, match="calibrate.py"):
        load_settings(tmp_path / "does_not_exist.json")
