# Privacy

desk-watch answers one question - "is this seat taken?" - and nothing more.

## What desk-watch does NOT do

- **No faces.** It never looks for, crops, or stores faces.
- **No identity.** It cannot tell who anyone is. YOLO's "person" class means "a human shape is
  here", nothing more. There is no face recognition, no name, no ID, no tracking of a person
  from desk to desk.
- **No images stored.** Camera frames are processed in memory and thrown away. No photo or
  video is written to disk during normal use.
- **No video on the dashboard.** The shared screen shows only colored cards with a desk name,
  a state, and a timer. It never shows the camera image.
- **No item details.** It does not record *what* was left on a desk - only that "an item" is there.

## What desk-watch DOES store

One file: `logs/events.csv`. Each line has exactly four things:

```
timestamp, desk_id, old_state, new_state
2026-09-25T20:05:00, desk_1, OCCUPIED, RESERVED_EMPTY
```

That's it. No pictures, no names, no descriptions of people.

## The two exceptions (both off by default, both for building the project)

- `--debug` shows the camera with boxes in a separate window so the builder can check
  detection. It is never meant for a public screen, and it saves nothing.
- `--record out.mp4` saves that debug view to a video file, **only** to make the demo
  video for this project. It prints a large warning when it starts. Normal use never records.

## How to check this yourself

- After a normal run (without `--record`), look in the project folder: the only new file
  is `logs/events.csv`.
