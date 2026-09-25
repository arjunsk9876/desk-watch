"""
test_privacy.py - Checks the privacy promises in PRIVACY.md against the actual code.

1. No product file saves images (no cv2.imwrite anywhere).
2. Only main.py can write video, and only inside start_recording (the --record flag).
3. No image files are sitting in the project folder.
"""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SKIP_FOLDERS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "tests"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


# ---- Helpers ----

def project_files(suffixes):
    """Every file in the project with one of these suffixes, skipping tool folders."""
    for path in PROJECT.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes \
                and not SKIP_FOLDERS & set(path.relative_to(PROJECT).parts):
            yield path


# ---- Tests ----

def test_no_code_saves_images():
    for path in project_files({".py"}):
        assert "imwrite" not in path.read_text(), f"{path.name} saves an image"


def test_only_record_flag_writes_video():
    writers = [path.name for path in project_files({".py"}) if "VideoWriter" in path.read_text()]
    assert writers == ["main.py"]

    # Inside main.py, the video writer is only created in start_recording().
    main_code = (PROJECT / "main.py").read_text()
    record_section = main_code.split("def start_recording")[1].split("\ndef ")[0]
    assert main_code.count("VideoWriter(") == record_section.count("VideoWriter(") == 1


def test_no_image_files_in_project():
    images = [str(path.relative_to(PROJECT)) for path in project_files(IMAGE_EXTENSIONS)]
    assert images == []
