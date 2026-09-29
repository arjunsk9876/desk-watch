"""Runs YOLO on a frame and works out if there's a person/item at each desk.

Only looks for "person" plus a few item classes. YOLO's person class
doesn't know who anyone is.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_FILE = "yolov8n.pt"   # smallest yolov8, downloads on first run

PERSON_CLASS = "person"
# COCO class names. can be changed with "item_classes" in settings.json
# (test with --check-items first). COCO doesn't have a jacket class.
ITEM_CLASSES = ["backpack", "laptop", "cell phone", "book", "bottle", "cup"]


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple   # x1, y1, x2, y2

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_person(self):
        return self.label == PERSON_CLASS


class DeskDetector:
    def __init__(self, desks, confidence=0.25, item_classes=ITEM_CLASSES, item_confidence=0.15):
        self.desks = desks
        self.confidence = confidence
        self.item_confidence = item_confidence   # lower, small stuff like phones scores low
        self.item_classes = list(item_classes)
        self.model = YOLO(MODEL_FILE)

        # a typo here would just never detect anything, so fail early
        unknown = set(self.item_classes) - set(self.model.names.values())
        if unknown:
            raise ValueError(f"Unknown item classes in settings: {sorted(unknown)}")

        # only ask YOLO for the classes we care about
        wanted = [PERSON_CLASS] + self.item_classes
        self.class_ids = [class_id for class_id, name in self.model.names.items()
                          if name in wanted]

    def detect(self, frame, confidence=None):
        # confidence overrides both thresholds (used by --check-items)
        lowest = confidence if confidence is not None else min(self.confidence, self.item_confidence)
        results = self.model.predict(frame, conf=lowest, classes=self.class_ids, verbose=False)[0]

        detections = []
        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            score = float(box.conf)
            # ran at the item threshold, so people still need to pass theirs
            if confidence is None and label == PERSON_CLASS and score < self.confidence:
                continue
            detections.append(Detection(
                label=label,
                confidence=score,
                box=tuple(float(value) for value in box.xyxy[0]),
            ))
        return detections

    def check_desks(self, detections):
        # {desk_id: {"person": bool, "item": bool}}
        answers = {}
        for desk in self.desks:
            in_zone = [d for d in detections if point_in_zone(d.center, desk["zone"])]
            answers[desk["id"]] = {
                "person": any(d.is_person for d in in_zone),
                "item": any(not d.is_person for d in in_zone),
            }
        return answers


def point_in_zone(point, zone):
    # using the box center so someone leaning over doesn't count for the next desk
    contour = np.array(zone, dtype=np.float32)
    # >= 0 means inside or on the edge
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def best_in_zone(detections, zone):
    in_zone = [d for d in detections if point_in_zone(d.center, zone)]
    person = max((d for d in in_zone if d.is_person), key=lambda d: d.confidence, default=None)
    item = max((d for d in in_zone if not d.is_person), key=lambda d: d.confidence, default=None)
    return person, item


def describe_desk(desk, detections, status=None):
    # e.g. "Desk 1: OCCUPIED  person 0.82  item 0.41 (backpack)"
    person, item = best_in_zone(detections, desk["zone"])
    state = f": {status.state}" if status else ""
    person_text = f"person {person.confidence:.2f}" if person else "person --"
    item_text = f"item {item.confidence:.2f} ({item.label})" if item else "item --"
    return f"{desk['name']}{state}  {person_text}  {item_text}"


# debug view stuff - only for testing, not the dashboard

PERSON_BOX_COLOR = (219, 152, 52)   # blue (BGR)
ITEM_BOX_COLOR = (15, 196, 241)     # yellow
ZONE_COLOR = (255, 255, 255)


def draw_debug(frame, desks, detections, statuses=None):
    view = frame.copy()
    for desk in desks:
        zone = np.array(desk["zone"], dtype=np.int32)
        cv2.polylines(view, [zone], isClosed=True, color=ZONE_COLOR, thickness=2)
        x, y = desk["zone"][0]
        cv2.putText(view, desk["name"], (int(x) + 6, int(y) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)

    for detection in detections:
        color = PERSON_BOX_COLOR if detection.is_person else ITEM_BOX_COLOR
        name = "person" if detection.is_person else f"item ({detection.label})"
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(view, f"{name} {detection.confidence:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # dot = the point that decides which desk it counts for
        center_x, center_y = (int(value) for value in detection.center)
        cv2.circle(view, (center_x, center_y), 5, color, -1)

    # status lines in the top left
    statuses = statuses or {}
    for row, desk in enumerate(desks):
        line = describe_desk(desk, detections, statuses.get(desk["id"]))
        y = 30 + row * 32
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.rectangle(view, (0, y - 24), (text_width + 20, y + 8), (30, 30, 30), -1)
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2)
    return view
