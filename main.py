"""
main.py - Starts desk-watch: reads the camera and shows the live desk dashboard.

Run it:
  python main.py                   live dashboard
  python main.py --check-items     print how confident YOLO is about each class
"""

import argparse
import sys
import time

import cv2

from config import load_settings, SettingsError, DEFAULT_SETTINGS_PATH
from detector import DeskDetector, PERSON_CLASS, ITEM_CLASSES


# ---- 1. Command line options ----

def parse_args():
    parser = argparse.ArgumentParser(description="desk-watch: is this seat taken?")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH,
                        help="settings file to use (e.g. demo_settings.json)")
    parser.add_argument("--check-items", action="store_true",
                        help="print detection confidence per class, then exit with Ctrl+C")
    return parser.parse_args()


# ---- 2. Camera ----

def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {camera_index}. "
                 "Check camera_index in your settings file and camera permissions.")
    return camera


# ---- 3. --check-items: which classes can YOLO see reliably? ----

def check_items(settings):
    """Every 2 seconds, print the best confidence YOLO found for each class.

    Use this before filming: hold up your bag/laptop and pick the items that
    score well above the confidence setting.
    """
    detector = DeskDetector(settings["desks"], settings["confidence"])
    camera = open_camera(settings["camera_index"])
    watched = [PERSON_CLASS] + ITEM_CLASSES
    print(f"Watching for: {', '.join(watched)}  (confidence setting = {settings['confidence']})")
    print("Press Ctrl+C to stop.\n")

    best = {label: 0.0 for label in watched}
    last_print = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            # Use a very low threshold here so we can see weak detections too.
            for detection in detector.detect(frame, confidence=0.05):
                best[detection.label] = max(best[detection.label], detection.confidence)

            if time.time() - last_print >= 2:
                print("  ".join(f"{label}: {score:.2f}" if score else f"{label}: --"
                                for label, score in best.items()))
                best = {label: 0.0 for label in watched}
                last_print = time.time()
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        camera.release()


# ---- 4. Start here ----

def main():
    args = parse_args()
    try:
        settings = load_settings(args.settings)
    except SettingsError as error:
        sys.exit(str(error))

    if args.check_items:
        check_items(settings)
        return


if __name__ == "__main__":
    main()
