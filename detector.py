"""
detector.py - Finds people and items in a camera frame using YOLO.

YOLO draws boxes around things it recognizes. We only ask it about two
kinds of things: people, and a short list of "items" (bags, laptops...).
We never ask WHO a person is - YOLO's "person" class has no identity.
The rest of desk-watch only ever sees simple True/False answers per desk.
"""

from ultralytics import YOLO

# ---- 1. What we look for ----

MODEL_FILE = "yolov8n.pt"   # small, fast model; downloads automatically the first time

PERSON_CLASS = "person"


# ---- 2. Load the model once ----

class DeskDetector:
    """Wraps YOLO so the rest of the app never touches raw model objects."""

    def __init__(self, desks, confidence=0.25, item_classes=None):
        self.desks = desks
        self.confidence = confidence
        self.model = YOLO(MODEL_FILE)
