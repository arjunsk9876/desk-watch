"""
dashboard.py - Draws the shared status screen: one colored card per desk.

This is what students or staff would see on a shared screen. On purpose,
it contains NO camera image - only each desk's name, state, and timer.
"""

import math
import time

import cv2
import numpy as np

from desk_state import FREE, OCCUPIED, RESERVED_EMPTY

# ---- 1. Look and layout ----

FONT = cv2.FONT_HERSHEY_DUPLEX
BACKGROUND = (36, 30, 30)       # near-black (colors are Blue, Green, Red)
WHITE = (255, 255, 255)
DARK = (40, 34, 34)
GREY = (170, 165, 160)

CARD_WIDTH, CARD_HEIGHT = 420, 240
GAP = 24          # space between cards
MARGIN = 32       # space around the edge of the window
HEADER = 110      # height of the title area
FOOTER = 50       # height of the privacy note at the bottom

# Card color and text for each state.
STYLES = {
    FREE:     {"color": (96, 174, 39),  "text": WHITE, "label": "Free"},
    OCCUPIED: {"color": (219, 152, 52), "text": WHITE, "label": "Occupied"},
    RESERVED_EMPTY: {"color": (15, 196, 241), "text": DARK, "label": "Reserved - empty"},
}
UNKNOWN_STYLE = {"color": (90, 90, 90), "text": WHITE, "label": "..."}


# ---- 2. Draw the whole dashboard ----

def draw_dashboard(desks, statuses):
    """Return an image with one card per desk. `statuses` maps desk_id -> DeskStatus."""
    columns = 1 if len(desks) == 1 else 2
    rows = math.ceil(len(desks) / columns)
    width = MARGIN * 2 + columns * CARD_WIDTH + (columns - 1) * GAP
    height = HEADER + rows * CARD_HEIGHT + (rows - 1) * GAP + FOOTER + MARGIN
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    draw_header(canvas)
    for index, desk in enumerate(desks):
        row, column = divmod(index, columns)
        x = MARGIN + column * (CARD_WIDTH + GAP)
        y = HEADER + row * (CARD_HEIGHT + GAP)
        draw_card(canvas, x, y, desk["name"], statuses.get(desk["id"]))

    note = "Status only - no video, no faces, no images stored."
    put_text(canvas, note, (MARGIN, height - MARGIN + 4), 0.55, GREY, 1)
    return canvas


def draw_header(canvas):
    put_text(canvas, "desk-watch", (MARGIN, 52), 1.3, WHITE, 2)
    put_text(canvas, "Is this seat taken?", (MARGIN, 76), 0.55, GREY, 1)
    clock = time.strftime("%H:%M")
    clock_width = cv2.getTextSize(clock, FONT, 1.0, 2)[0][0]
    put_text(canvas, clock, (canvas.shape[1] - MARGIN - clock_width, 52), 1.0, GREY, 2)


# ---- 3. Draw one desk card ----

def draw_card(canvas, x, y, desk_name, status):
    style = STYLES.get(status.state, UNKNOWN_STYLE) if status else UNKNOWN_STYLE
    rounded_rectangle(canvas, x, y, CARD_WIDTH, CARD_HEIGHT, 18, style["color"])

    put_text(canvas, desk_name, (x + 24, y + 44), 0.8, style["text"], 1)

    if status and status.state == RESERVED_EMPTY:
        # Reserved desks get a live timer: how long has the owner been gone?
        put_centered(canvas, style["label"], x, y + 125, 1.2, style["text"], 2)
        put_centered(canvas, format_duration(status.seconds_in_state), x, y + 185,
                     1.4, style["text"], 3)
    else:
        put_centered(canvas, style["label"], x, y + 140, 1.6, style["text"], 3)


def format_duration(seconds):
    """Turn 125.4 seconds into '2m 05s'."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s"


# ---- 4. Small drawing helpers ----

def put_text(canvas, text, origin, scale, color, thickness):
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def put_centered(canvas, text, card_x, baseline_y, scale, color, thickness):
    """Draw text centered horizontally inside a card."""
    text_width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    put_text(canvas, text, (card_x + (CARD_WIDTH - text_width) // 2, baseline_y),
             scale, color, thickness)


def rounded_rectangle(canvas, x, y, width, height, radius, color):
    """OpenCV has no rounded rectangle, so build one from 2 rectangles + 4 circles."""
    cv2.rectangle(canvas, (x + radius, y), (x + width - radius, y + height), color, -1)
    cv2.rectangle(canvas, (x, y + radius), (x + width, y + height - radius), color, -1)
    for corner_x in (x + radius, x + width - radius):
        for corner_y in (y + radius, y + height - radius):
            cv2.circle(canvas, (corner_x, corner_y), radius, color, -1, cv2.LINE_AA)
