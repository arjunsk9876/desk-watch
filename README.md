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
