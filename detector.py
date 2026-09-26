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
# which one it is - only that something was left behind. These are standard
# YOLO (COCO) class names; override them with "item_classes" in settings.json
# after testing with --check-items. (There is no "jacket" class in COCO.)
ITEM_CLASSES = ["backpack", "laptop", "cell phone", "book", "bottle", "cup"]


@dataclass
class Detection:
    label: str          # "person" or one of the item classes
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

    def __init__(self, desks, confidence=0.25, item_classes=ITEM_CLASSES, item_confidence=0.15):
        self.desks = desks
        self.confidence = confidence              # how sure YOLO must be about a person
        self.item_confidence = item_confidence    # items are small and harder to see, so allow lower
        self.item_classes = list(item_classes)
        self.model = YOLO(MODEL_FILE)

        # Catch typos early: a misspelled class would silently never be detected.
        unknown = set(self.item_classes) - set(self.model.names.values())
        if unknown:
            raise ValueError(f"Unknown item classes in settings: {sorted(unknown)}")

        # YOLO knows 80 classes by number. Look up the numbers for the few we want,
        # so YOLO skips everything else (faster, and no cups/chairs cluttering things up).
        wanted = [PERSON_CLASS] + self.item_classes
        self.class_ids = [class_id for class_id, name in self.model.names.items()
                          if name in wanted]

    # ---- 3. Find people and items in one frame ----

    def detect(self, frame, confidence=None):
        """Run YOLO on one frame and return a plain list of Detections.

        Passing `confidence` uses one threshold for everything (for --check-items).
        """
        lowest = confidence if confidence is not None else min(self.confidence, self.item_confidence)
        results = self.model.predict(frame, conf=lowest, classes=self.class_ids, verbose=False)[0]

        detections = []
        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            score = float(box.conf)
            # People must still pass the stricter person threshold.
            if confidence is None and label == PERSON_CLASS and score < self.confidence:
                continue
            detections.append(Detection(
                label=label,
                confidence=score,
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


def best_in_zone(detections, zone):
    """The most confident person and item inside a zone, or None (for the debug view)."""
    in_zone = [d for d in detections if point_in_zone(d.center, zone)]
    person = max((d for d in in_zone if d.is_person), key=lambda d: d.confidence, default=None)
    item = max((d for d in in_zone if not d.is_person), key=lambda d: d.confidence, default=None)
    return person, item


def describe_desk(desk, detections, status=None):
    """One line like 'Desk 1: OCCUPIED  person 0.82  item 0.41 (backpack)'."""
    person, item = best_in_zone(detections, desk["zone"])
    state = f": {status.state}" if status else ""
    person_text = f"person {person.confidence:.2f}" if person else "person --"
    item_text = f"item {item.confidence:.2f} ({item.label})" if item else "item --"
    return f"{desk['name']}{state}  {person_text}  {item_text}"


# ---- 6. Debug view (for building/testing only, never the public dashboard) ----

PERSON_BOX_COLOR = (219, 152, 52)   # blue
ITEM_BOX_COLOR = (15, 196, 241)     # yellow
ZONE_COLOR = (255, 255, 255)


def draw_debug(frame, desks, detections, statuses=None):
    """Draw desk zones, detection boxes, and confidences on a COPY of the camera frame."""
    view = frame.copy()
    for desk in desks:
        zone = np.array(desk["zone"], dtype=np.int32)
        cv2.polylines(view, [zone], isClosed=True, color=ZONE_COLOR, thickness=2)
        x, y = desk["zone"][0]
        cv2.putText(view, desk["name"], (int(x) + 6, int(y) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)

    for detection in detections:
        color = PERSON_BOX_COLOR if detection.is_person else ITEM_BOX_COLOR
        # The class name is shown here only to help tune item_classes.
        name = "person" if detection.is_person else f"item ({detection.label})"
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(view, f"{name} {detection.confidence:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # The dot is the point that decides which desk a box belongs to.
        center_x, center_y = (int(value) for value in detection.center)
        cv2.circle(view, (center_x, center_y), 5, color, -1)

    # Summary panel in the top-left corner: each desk's state and best confidences.
    statuses = statuses or {}
    for row, desk in enumerate(desks):
        line = describe_desk(desk, detections, statuses.get(desk["id"]))
        y = 30 + row * 32
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.rectangle(view, (0, y - 24), (text_width + 20, y + 8), (30, 30, 30), -1)
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2)
    return view
