"""desk-watch - reads the camera and shows the desk dashboard.

camera -> detector -> desk_state -> dashboard + logger

  python main.py                   dashboard
  python main.py --debug           also show camera view with boxes
  python main.py --record out.mp4  record the debug view (demo video only)
  python main.py --check-items     print YOLO confidence for each class
  python main.py --settings demo_settings.json   short timers for filming
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
MAX_FAILED_READS = 50   # ~1 sec


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


def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {camera_index}. "
                 "Check camera_index in your settings file and camera permissions.")
    return camera


def read_frame(camera):
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
    except ValueError as error:   # bad item class name in settings
        sys.exit(str(error))


def check_items(settings):
    # prints the best score per class every 2 sec. hold stuff up to the
    # camera and keep the classes that score well above the threshold
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
            # really low threshold so weak detections show up too
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


def start_recording(path, frame_size, fps=15):
    # only used for the demo video. normal runs never save video
    print("\n" + "!" * 70)
    print(f"  WARNING: --record is ON. Saving the camera view to {path}.")
    print("  This is only for making the demo video - normal use saves no video or images.")
    print("!" * 70 + "\n")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)


def update_desks(settings, detector, tracker, frame):
    now = time.time()
    detections = detector.detect(frame)
    desk_answers = detector.check_desks(detections)
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
    recorder = None
    last_debug_print = 0
    last_states = {}

    print("desk-watch running. Press q in a window (or Ctrl+C) to quit.")
    try:
        while True:
            frame = read_frame(camera)
            detections, statuses = update_desks(settings, detector, tracker, frame)

            for desk_id, status in statuses.items():
                old_state = last_states.get(desk_id)
                if old_state is not None and old_state != status.state:
                    print(f"{desk_id}: {old_state} -> {status.state}")
                    logger.log(desk_id, old_state, status.state)
                last_states[desk_id] = status.state

            # dashboard only gets the statuses, never the frame
            dashboard = draw_dashboard(settings["desks"], statuses, settings["long_reserved_minutes"])
            cv2.imshow(DASHBOARD_WINDOW, dashboard)

            if debug or record_path:
                debug_view = draw_debug(frame, settings["desks"], detections, statuses)
                if time.time() - last_debug_print >= 1:
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
