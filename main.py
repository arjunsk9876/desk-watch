"""
main.py - Starts desk-watch: reads the camera and shows the live desk dashboard.

Each frame goes: camera -> detector.py (people/items per desk)
-> desk_state.py (FREE / OCCUPIED / RESERVED_EMPTY) -> dashboard.py + logger.py.

Run it:
  python main.py                   live dashboard
  python main.py --debug           also show the camera view with boxes
  python main.py --record out.mp4  DEMO ONLY: save the debug view to a video file
  python main.py --check-items     print how confident YOLO is about each class
  python main.py --settings demo_settings.json   use short timers for filming
"""

import argparse
import sys
import time

import cv2

from config import load_settings, SettingsError, DEFAULT_SETTINGS_PATH
from dashboard import draw_dashboard
from desk_state import DeskTracker
from detector import DeskDetector, PERSON_CLASS, draw_debug, describe_desk
from logger import EventLogger

DASHBOARD_WINDOW = "desk-watch dashboard"
DEBUG_WINDOW = "desk-watch debug (camera - not for public display)"
MAX_FAILED_READS = 50   # about 1 second of the camera sending nothing


# ---- 1. Command line options ----

def parse_args():
    parser = argparse.ArgumentParser(description="desk-watch: is this seat taken?")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH,
                        help="settings file to use (e.g. demo_settings.json)")
    parser.add_argument("--debug", action="store_true",
                        help="also show the raw camera view with desk zones and boxes")
    parser.add_argument("--record", metavar="FILE",
                        help="DEMO ONLY: record the debug view to a video file (e.g. out.mp4)")
    parser.add_argument("--check-items", action="store_true",
                        help="print detection confidence per class, then exit with Ctrl+C")
    return parser.parse_args()


# ---- 2. Camera and detector setup ----

def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {camera_index}. "
                 "Check camera_index in your settings file and camera permissions.")
    return camera


def read_frame(camera):
    """Get the next frame, or stop with a clear message if the camera goes silent."""
    for _ in range(MAX_FAILED_READS):
        ok, frame = camera.read()
        if ok:
            return frame
        time.sleep(0.02)
    sys.exit("The camera stopped sending frames. Is another app using it?")


def make_detector(settings):
    try:
        return DeskDetector(settings["desks"], settings["confidence"],
                            settings["item_classes"], settings["item_confidence"])
    except ValueError as error:   # e.g. a misspelled item class in settings.json
        sys.exit(str(error))


# ---- 3. --check-items: which classes can YOLO see reliably? ----

def check_items(settings):
    """Every 2 seconds, print the best confidence YOLO found for each class.

    Use this before filming: hold up your bag/laptop and pick the items that
    score well above the confidence setting.
    """
    detector = make_detector(settings)
    camera = open_camera(settings["camera_index"])
    watched = [PERSON_CLASS] + detector.item_classes
    print(f"Watching for: {', '.join(watched)}  (person needs {settings['confidence']}, "
          f"items need {settings['item_confidence']})")
    print("Press Ctrl+C to stop.\n")

    best = {label: 0.0 for label in watched}
    last_print = time.time()
    try:
        while True:
            frame = read_frame(camera)
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


# ---- 4. --record: DEMO ONLY, never part of normal use ----

def start_recording(path, frame_size, fps=15):
    """Open a video file for the debug view. The product itself never records."""
    print("\n" + "!" * 70)
    print(f"  WARNING: --record is ON. Saving the camera view to {path}.")
    print("  This is only for making the demo video - normal use saves no video or images.")
    print("!" * 70 + "\n")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)


# ---- 5. The live loop: camera -> detector -> desk states -> dashboard ----

def update_desks(settings, detector, tracker, frame):
    """Run one frame through the pipeline. Returns (detections, {desk_id: DeskStatus})."""
    now = time.time()
    # Step 1: YOLO finds people and items in the frame.
    detections = detector.detect(frame)
    # Step 2: turn boxes into "person yes/no, item yes/no" for each desk.
    desk_answers = detector.check_desks(detections)
    # Step 3: update each desk's state machine.
    statuses = {}
    for desk in settings["desks"]:
        answer = desk_answers[desk["id"]]
        statuses[desk["id"]] = tracker.update(desk["id"], answer["person"], answer["item"], now)
    return detections, statuses


def run(settings, debug=False, record_path=None):
    detector = make_detector(settings)
    tracker = DeskTracker(settings["smoothing_frames"], settings["long_reserved_minutes"])
    camera = open_camera(settings["camera_index"])
    logger = EventLogger()
    recorder = None    # only created when --record is used
    last_debug_print = 0
    last_states = {}   # desk_id -> state last frame, so we can spot changes

    print("desk-watch running. Press q in a window (or Ctrl+C) to quit.")
    try:
        while True:
            frame = read_frame(camera)
            detections, statuses = update_desks(settings, detector, tracker, frame)

            # Step 4: log every state change (desk, states, time - nothing else).
            for desk_id, status in statuses.items():
                old_state = last_states.get(desk_id)
                if old_state is not None and old_state != status.state:
                    print(f"{desk_id}: {old_state} -> {status.state}")
                    logger.log(desk_id, old_state, status.state)
                last_states[desk_id] = status.state

            # Step 5: show the public dashboard - status cards only, never the camera image.
            dashboard = draw_dashboard(settings["desks"], statuses, settings["long_reserved_minutes"])
            cv2.imshow(DASHBOARD_WINDOW, dashboard)

            # Optional: the camera view with boxes, for building and testing only.
            if debug or record_path:
                debug_view = draw_debug(frame, settings["desks"], detections, statuses)
                if time.time() - last_debug_print >= 1:   # confidences once a second
                    for desk in settings["desks"]:
                        print("  " + describe_desk(desk, detections, statuses[desk["id"]]))
                    last_debug_print = time.time()
                cv2.imshow(DEBUG_WINDOW, debug_view)
                if record_path:
                    if recorder is None:
                        height, width = debug_view.shape[:2]
                        recorder = start_recording(record_path, (width, height))
                    recorder.write(debug_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if recorder is not None:
            recorder.release()
            print(f"Recording saved to {record_path}")
        cv2.destroyAllWindows()


# ---- 6. Start here ----

def main():
    args = parse_args()
    try:
        settings = load_settings(args.settings)
    except SettingsError as error:
        sys.exit(str(error))

    if args.check_items:
        check_items(settings)
        return
    if not settings["desks"]:
        sys.exit("No desks in the settings file. Run 'python calibrate.py' first.")

    run(settings, debug=args.debug, record_path=args.record)


if __name__ == "__main__":
    main()
