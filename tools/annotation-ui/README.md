# SetOptics Annotation Tool

A desktop tool for labeling **volleyball rally segments** (ball in-play vs.
break) to build ground-truth data for SetOptics. Load a video, mark the
rallies on a timeline, and export a corrected annotation JSON.

You can annotate **fully by hand with no Python and no API key** — just Node.
The optional AI draft (Gemini) needs the project's Python env; see below.

## Quick start (one command)

From a fresh clone:

```bash
cd tools/annotation-ui
npm run local
```

`npm run local` installs dependencies and launches the app. (Node 18+ and,
for export, `ffmpeg` on your PATH.)

## Annotating

1. Drag a video onto the window (MP4 / MOV / WebM).
2. Pick a starting point:
   - **Start from scratch** — empty timeline, no Python needed.
   - **Run Detection** — optional Gemini AI draft (~90% right) to correct.
     Requires the project Python env: run `./setup.sh` at the repo root and
     set `GEMINI_API_KEY` in `.env`.
   - **Load File…** — open an existing annotation JSON.
3. Correct the timeline:
   - Green = `in-play` (rally), gray = `break`.
   - Drag a segment's edges to fix boundaries.
   - `1` = mark `in-play`, `2` = mark `break`, `B` = split at the playhead.
4. **Export** — produces `<video>_annotations_corrected.json`.

## Output format

```json
{
  "video_metadata": { "path": "...", "duration_seconds": 1234.5 },
  "segments": [
    { "segment_id": 1, "type": "in-play", "start_ms": 12345,
      "end_ms": 67890, "rally_number": 1 }
  ]
}
```

Only `type`, `start_ms`, and `end_ms` are required by the eval framework.

## Attribution & license

This tool is built on [OpenScreen](https://github.com/siddharthvaddem/openscreen)
by Siddharth Vaddem, used and extended under the MIT License. The original
MIT license and copyright are retained in [`LICENSE`](./LICENSE).
