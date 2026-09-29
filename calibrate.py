"""Click the 4 corners of each desk (up to 4 desks) and save them to settings.json.

Include the chair in the zone so a seated person lands inside it.
Keys: n = next desk, s = save, r = reset, q = quit
"""

import argparse
import sys
from pathlib import Path

import cv2

from config import DEFAULTS, DEFAULT_SETTINGS_PATH, load_settings, save_settings

MAX_DESKS = 4
POINTS_PER_DESK = 4
WINDOW = "desk-watch calibration"

DONE_COLOR = (80, 175, 76)       # green (opencv is BGR)
DRAWING_COLOR = (40, 200, 245)   # yellow


class Calibration:
    def __init__(self):
        self.zones = []     # finished desks
        self.current = []   # desk being drawn

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.current) < POINTS_PER_DESK and len(self.zones) < MAX_DESKS:
            self.current.append([x, y])

    def next_desk(self):
        if len(self.current) == POINTS_PER_DESK:
            self.zones.append(self.current)
            self.current = []

    def reset(self):
        self.zones = []
        self.current = []

    def finished_zones(self):
        # counts the current desk too if it has all 4 corners
        if len(self.current) == POINTS_PER_DESK:
            return self.zones + [self.current]
        return self.zones

    def instructions(self):
        desk_number = len(self.zones) + 1
        if len(self.zones) >= MAX_DESKS:
            return f"All {MAX_DESKS} desks drawn.  s = save   r = reset   q = quit"
        if len(self.current) < POINTS_PER_DESK:
            return f"Desk {desk_number}: click corner {len(self.current) + 1} of {POINTS_PER_DESK}"
        return f"Desk {desk_number} done.  n = next desk   s = save   r = reset   q = quit"


def draw(frame, calibration):
    for number, zone in enumerate(calibration.zones, start=1):
        draw_zone(frame, zone, DONE_COLOR, closed=True)
        cv2.putText(frame, f"Desk {number}", (zone[0][0] + 6, zone[0][1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DONE_COLOR, 2)

    if calibration.current:
        closed = len(calibration.current) == POINTS_PER_DESK
        draw_zone(frame, calibration.current, DRAWING_COLOR, closed)

    # dark bar at the top so the text is readable
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(frame, calibration.instructions(), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_zone(frame, points, color, closed):
    for x, y in points:
        cv2.circle(frame, (x, y), 5, color, -1)
    for start, end in zip(points, points[1:] + (points[:1] if closed else [])):
        cv2.line(frame, tuple(start), tuple(end), color, 2)


def save(calibration, path):
    zones = calibration.finished_zones()
    if not zones:
        print("Nothing to save yet - draw at least one desk (4 corners).")
        return

    # keep whatever else is already in the settings file
    settings = load_settings(path) if Path(path).exists() else dict(DEFAULTS)
    settings["desks"] = [
        {"id": f"desk_{number}", "name": f"Desk {number}", "zone": zone}
        for number, zone in enumerate(zones, start=1)
    ]
    save_settings(settings, path)
    print(f"Saved {len(zones)} desk(s) to {path}")


def main():
    parser = argparse.ArgumentParser(description="Draw desk zones for desk-watch")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH, help="file to save zones into")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (default: camera_index from the settings file, or 0)")
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        existing = load_settings(args.settings) if Path(args.settings).exists() else DEFAULTS
        camera_index = existing["camera_index"]

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {camera_index}. Check camera permissions.")

    calibration = Calibration()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, calibration.on_click)

    failed_reads = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            failed_reads += 1
            if failed_reads > 50:
                sys.exit("The camera stopped sending frames. Is another app using it?")
            continue
        failed_reads = 0
        draw(frame, calibration)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            calibration.next_desk()
        elif key == ord("s"):
            save(calibration, args.settings)
        elif key == ord("r"):
            calibration.reset()
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
