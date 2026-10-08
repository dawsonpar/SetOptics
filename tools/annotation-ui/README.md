# SetOptics Annotation Tool

A desktop tool for labeling **volleyball rally segments** (ball in-play vs.
break) to build ground-truth data for SetOptics. Load a video, mark the
rallies on a timeline, and export a corrected annotation JSON.

You can annotate **fully by hand with no Python and no API key**, just Node.
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
   - **Start from scratch**: empty timeline, no Python needed.
   - **Run Detection**: optional Gemini AI draft (~90% right) to correct.
     Requires the project Python env: run `./setup.sh` at the repo root and
     set `GEMINI_API_KEY` in `.env`.
   - **Load File…**: open an existing annotation JSON.
3. Correct the timeline:
   - Green = `in-play` (rally), gray = `break`.
   - Drag a segment's edges to fix boundaries.
   - `1` = mark `in-play`, `2` = mark `break`, `B` = split at the playhead.
4. **Export** produces `<video>_annotations_corrected.json`.

## Making a raw draft

A raw draft is a `<video>_raw_annotations.json` you correct in the app.
`<video>` is the video's file name without the extension, matched exactly.
Put the draft next to the video and the app offers **Load Raw**.

| Way | Command | Needs |
|---|---|---|
| Gemini, ground-truth quality | `python tools/annotation/annotate_sliding_window.py VIDEO.mp4` (writes next to the video) | Python env, `GEMINI_API_KEY` |
| Local signal detector, rougher | `python scripts/signal_rally_detector.py --video VIDEO.mp4 --output VIDEO_raw_annotations.json` | Python env, no key |
| In the app | **Run Detection** (runs `tools/annotation/annotate_fast.py`) | Python env, `GEMINI_API_KEY` |
| By hand | Write the JSON yourself, see [docs/annotation.md](../../docs/annotation.md) | Nothing |

Run the commands from the repo root with the venv active. The Python env
is `./setup.sh`; the key goes in the repo-root `.env`.

The app looks for files in this order and uses the first it finds:

- Corrected: next to the video, then `data/rally-gt/`, then `data/samples/`.
- Raw: `<video>_raw_annotations.json` next to the video, then in
  `data/rally-gt/`; then `<video>_rally_annotations.json` next to the
  video, then in `tools/annotation/annotations/<video>/rally/` (where
  **Run Detection** writes).

## Browser mode

Runs the editor in a plain browser instead of Electron, on this machine
only.

```bash
ANNOTATION_WEB_TOKEN=$(openssl rand -hex 16) npm run dev:web
```

Open `http://localhost:5173/?key=<token>` once; a cookie keeps the session.
The token must be at least 16 characters.

- Videos are listed from `data/rally-gt/` at the repo root. Set
  `ANNOTATION_VIDEO_ROOT` to use another folder.
- Saving writes `<video>_annotations_corrected.json` next to the video and
  keeps the previous file as `.backup`.
- The AI draft and **Load File…** are Electron only. Start from scratch,
  or put a raw annotation JSON next to the video so it is detected.

## Troubleshooting

| What you see | Cause and fix |
|---|---|
| `npm run local` fails | Node is older than 18. Upgrade Node. |
| Detected **Unknown**, "No annotation files found" | No file with the video's exact name in the places above. Check the name and folder, or use Start from scratch. |
| "AI draft needs the Python environment" | Run `./setup.sh` at the repo root. |
| "Gemini API key not found" | Add `GEMINI_API_KEY=...` to the repo-root `.env`. |
| Run Detection fails with "All chunks failed" | Every Gemini call failed: invalid key or quota (400 or 429 in the log). Fix the key, or wait out the quota. |
| A sliding-window draft is one long break | Same cause: every Gemini call failed and the script still wrote a file. Check its log, fix the key, rerun with `--force`. |
| Progress stays at 0% during Run Detection | The bar does not track this script yet. Wait for it to finish. |
| "Failed to load video" | The file moved, or its codec does not play. Pick it again with **Select Different Video**, or re-encode: `ffmpeg -i IN.mov -c:v libx264 -c:a aac OUT.mp4`. |
| "Saved with issues: N segment(s) extend past video end" | The draft belongs to a longer video with the same name, usually a stale raw file. Delete it and redraft, or remove the extra segments before saving. |
| Browser mode: "Unauthorized" | Open the link with `?key=<token>` once. |
| Browser mode: server will not start | `ANNOTATION_WEB_TOKEN` is unset or shorter than 16 characters. |
| Browser mode: "No videos found" | Videos must be `.mp4`, `.mov` or `.webm`, at most two folders deep under `data/rally-gt/` or `ANNOTATION_VIDEO_ROOT`. |

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
