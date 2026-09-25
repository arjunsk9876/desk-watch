"""
detector.py - Finds people and items in a camera frame using YOLO.

YOLO draws boxes around things it recognizes. We only ask it about two
kinds of things: people, and a short list of "items" (bags, laptops...).
We never ask WHO a person is - YOLO's "person" class has no identity.
The rest of desk-watch only ever sees simple True/False answers per desk.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# ---- 1. What we look for ----

MODEL_FILE = "yolov8n.pt"   # small, fast model; downloads automatically the first time

PERSON_CLASS = "person"
# Anything in this list counts as "an item is on the desk". We don't care
# which one it is - only that something was left behind.
ITEM_CLASSES = ["backpack", "handbag", "suitcase", "laptop", "book"]


@dataclass
class Detection:
    label: str          # "person" or one of ITEM_CLASSES
    confidence: float   # 0.0 - 1.0, how sure YOLO is
    box: tuple          # (x1, y1, x2, y2) corners in pixels

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_person(self):
        return self.label == PERSON_CLASS


# ---- 2. Load the model once ----

class DeskDetector:
    """Wraps YOLO so the rest of the app never touches raw model objects."""

    def __init__(self, desks, confidence=0.25):
        self.desks = desks
        self.confidence = confidence
        self.model = YOLO(MODEL_FILE)

        # YOLO knows 80 classes by number. Look up the numbers for the few we want,
        # so YOLO skips everything else (faster, and no cups/chairs cluttering things up).
        wanted = [PERSON_CLASS] + ITEM_CLASSES
        self.class_ids = [class_id for class_id, name in self.model.names.items()
                          if name in wanted]

    # ---- 3. Find people and items in one frame ----

    def detect(self, frame, confidence=None):
        """Run YOLO on one frame and return a plain list of Detections."""
        results = self.model.predict(
            frame,
            conf=confidence if confidence is not None else self.confidence,
            classes=self.class_ids,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            detections.append(Detection(
                label=label,
                confidence=float(box.conf),
                box=tuple(float(value) for value in box.xyxy[0]),
            ))
        return detections

    # ---- 4. Turn boxes into a simple yes/no per desk ----

    def check_desks(self, detections):
        """Return {desk_id: {"person": bool, "item": bool}} for every desk zone."""
        answers = {}
        for desk in self.desks:
            in_zone = [d for d in detections if point_in_zone(d.center, desk["zone"])]
            answers[desk["id"]] = {
                "person": any(d.is_person for d in in_zone),
                "item": any(not d.is_person for d in in_zone),
            }
        return answers


# ---- 5. Is a point inside a desk zone? ----

def point_in_zone(point, zone):
    """True if the (x, y) point is inside the polygon drawn during calibration.

    We use the CENTER of each box: a person's box center lands on their body,
    so it falls inside the desk zone they are sitting at, not the one next door.
    """
    contour = np.array(zone, dtype=np.float32)
    # pointPolygonTest returns +1 inside, 0 on the edge, -1 outside.
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
