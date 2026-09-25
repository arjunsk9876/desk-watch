# desk-watch

Is this seat taken? desk-watch uses one webcam to show which desks in a shared space are
**free**, **occupied**, or **reserved but empty** (someone left a bag and walked away) —
without ever identifying who is sitting there.

## Setup

You need Python 3.11 or newer and a webcam.

```bash
cd desk-watch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first time desk-watch runs, YOLO downloads its small model file (`yolov8n.pt`, about 6 MB),
so be online for the first run.

Check that everything works:

```bash
pytest                           # runs the tests - no camera needed
python main.py --check-items     # prints how confident YOLO is about people and items
```

With `--check-items` running, sit in view and hold up your bag or laptop. Pick items that score
well above the `confidence` setting (0.25), and remove unreliable ones from `item_classes` in
`settings.json`. (A backpack usually works well. There is no "jacket" class in YOLO's standard model.)

On a Mac, the first run asks for camera permission for your terminal app - allow it, then run again.

## Calibration: telling desk-watch where the desks are

```bash
python calibrate.py
```

1. A live camera window opens.
2. Click the **4 corners** around the first desk. Include the chair: desk-watch decides which
   desk a person belongs to by the **center** of their body, so the zone must cover where a
   seated person's body is, not just the desktop.
3. Press **n** to start the next desk (up to 4).
4. Press **s** to save to `settings.json`, **r** to start over, **q** to quit.

Zones are saved as pixel positions, so don't move the camera after calibrating.
For filming, calibrate the demo settings file too:

```bash
python calibrate.py --settings demo_settings.json
```

## Running

```bash
python main.py                                  # the public dashboard
python main.py --debug                          # also show the camera view with boxes
python main.py --settings demo_settings.json    # short timers for filming
python main.py --record out.mp4                 # DEMO ONLY: record the debug view to a video
```

Press **q** in a window (or Ctrl+C in the terminal) to quit.

## How it works

Every frame from the camera goes through four steps:

1. **Detect** (`detector.py`) - YOLO finds boxes around people and a short list of items
   (backpack, handbag, suitcase, laptop, book). It only knows "a person", never *who*.
2. **Assign to desks** (`detector.py`) - a box belongs to a desk if the center of the box is
   inside that desk's zone. Each desk gets two yes/no answers: *is a person here?* and
   *is an item here?*
3. **Update the state machine** (`desk_state.py`) - each desk is in one of three states:

   | Now | What happens | Next |
   |---|---|---|
   | FREE | a person sits down | OCCUPIED |
   | OCCUPIED | the person leaves and takes everything | FREE |
   | OCCUPIED | the person leaves, an item stays | RESERVED_EMPTY (timer starts) |
   | RESERVED_EMPTY | the person comes back | OCCUPIED (timer resets) |
   | RESERVED_EMPTY | the item is removed, nobody sat down | FREE |
   | RESERVED_EMPTY | the timer passes `long_reserved_minutes` | still RESERVED_EMPTY, but flagged **long** |

   To stop flickering when YOLO misses a frame, each yes/no answer is a vote over the last
   `smoothing_frames` frames (default 10): it counts as "yes" only if more than half say yes.
4. **Show and log** - `dashboard.py` draws one colored card per desk, and `logger.py` writes a
   line to `logs/events.csv` whenever a desk changes state.

| Card | Meaning |
|---|---|
| Green - Free | nothing there |
| Blue - Occupied | someone is sitting there |
| Yellow - Reserved, empty 2m 05s | their stuff is there, they are not (live timer) |
| Red-orange - Empty 30+ min, Available | left too long - fair game |

`desk_state.py` has no camera or AI code at all, so it can be tested with made-up inputs
(see `tests/test_desk_state.py`).

## Settings

| Field | Default | Meaning |
|---|---|---|
| `camera_index` | 0 | which webcam to use |
| `desks` | - | list of `{id, name, zone}`; `zone` is 4 `[x, y]` corners (made by `calibrate.py`) |
| `confidence` | 0.25 | how sure YOLO must be before a box counts |
| `smoothing_frames` | 10 | frames used for the anti-flicker vote |
| `long_reserved_minutes` | 30 | when a reserved-but-empty desk becomes "available" (0.5 in `demo_settings.json`) |
| `item_classes` | backpack, handbag, suitcase, laptop, book | YOLO classes that count as "an item" |

## Privacy

desk-watch never identifies anyone and never saves images. See [PRIVACY.md](PRIVACY.md).
