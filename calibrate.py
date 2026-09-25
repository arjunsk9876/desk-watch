"""
calibrate.py - Draw where each desk is in the camera view.

Run `python calibrate.py`, then click 4 corners around each desk (include
the chair, so a seated person's body lands inside the zone). Up to 4 desks.
Keys:  n = next desk   s = save   r = reset   q = quit
"""

import argparse
import sys
from pathlib import Path

import cv2

from config import DEFAULTS, DEFAULT_SETTINGS_PATH, load_settings, save_settings

MAX_DESKS = 4
POINTS_PER_DESK = 4
WINDOW = "desk-watch calibration"

DONE_COLOR = (80, 175, 76)       # green (OpenCV colors are Blue, Green, Red)
DRAWING_COLOR = (40, 200, 245)   # yellow


# ---- 1. Keep track of what has been drawn ----

class Calibration:
    def __init__(self):
        self.zones = []     # finished desk zones, each a list of 4 [x, y] points
        self.current = []   # corners of the desk being drawn right now

    def on_click(self, event, x, y, flags, param):
        """OpenCV calls this on every mouse event; we only care about left clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.current) < POINTS_PER_DESK and len(self.zones) < MAX_DESKS:
            self.current.append([x, y])

    def next_desk(self):
        """Finish the current zone (only once all 4 corners are placed)."""
        if len(self.current) == POINTS_PER_DESK:
            self.zones.append(self.current)
            self.current = []

    def reset(self):
        """Throw away every zone and start over."""
        self.zones = []
        self.current = []

    def finished_zones(self):
        """All complete zones, including the one being drawn if it has 4 corners."""
        if len(self.current) == POINTS_PER_DESK:
            return self.zones + [self.current]
        return self.zones

    def instructions(self):
        """The line of help text shown at the top of the window."""
        desk_number = len(self.zones) + 1
        if len(self.zones) >= MAX_DESKS:
            return f"All {MAX_DESKS} desks drawn.  s = save   r = reset   q = quit"
        if len(self.current) < POINTS_PER_DESK:
            return f"Desk {desk_number}: click corner {len(self.current) + 1} of {POINTS_PER_DESK}"
        return f"Desk {desk_number} done.  n = next desk   s = save   r = reset   q = quit"


# ---- 2. Draw the zones on top of the camera image ----

def draw(frame, calibration):
    for number, zone in enumerate(calibration.zones, start=1):
        draw_zone(frame, zone, DONE_COLOR, closed=True)
        cv2.putText(frame, f"Desk {number}", (zone[0][0] + 6, zone[0][1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DONE_COLOR, 2)

    if calibration.current:
        closed = len(calibration.current) == POINTS_PER_DESK
        draw_zone(frame, calibration.current, DRAWING_COLOR, closed)

    # Instruction bar across the top, dark background so it is always readable.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(frame, calibration.instructions(), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_zone(frame, points, color, closed):
    for x, y in points:
        cv2.circle(frame, (x, y), 5, color, -1)
    for start, end in zip(points, points[1:] + (points[:1] if closed else [])):
        cv2.line(frame, tuple(start), tuple(end), color, 2)


# ---- 3. Save zones into the settings file ----

def save(calibration, path):
    """Write the desk zones to settings, keeping any other settings already there."""
    zones = calibration.finished_zones()
    if not zones:
        print("Nothing to save yet - draw at least one desk (4 corners).")
        return

    settings = load_settings(path) if Path(path).exists() else dict(DEFAULTS)
    settings["desks"] = [
        {"id": f"desk_{number}", "name": f"Desk {number}", "zone": zone}
        for number, zone in enumerate(zones, start=1)
    ]
    save_settings(settings, path)
    print(f"Saved {len(zones)} desk(s) to {path}")


# ---- 4. Main loop: show camera, handle keys ----

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

    while True:
        ok, frame = camera.read()
        if not ok:
            continue
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
