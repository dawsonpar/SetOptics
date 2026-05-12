# Annotation

Two scripts. They share a directory but differ by ~20pp F1.

| Script | Model | Speed | F1 | Use for |
|--------|-------|-------|----|---------|
| `annotate_sliding_window.py` | `gemini-2.5-flash` | Slower | ~94% | Ground truth |
| `annotate_fast.py` | `gemini-3-flash-preview` (configurable) | Faster | ~75% | Prompt iteration |

If you are building ground truth for evaluation, use the sliding-window
script. The fast script is useful when you are tuning prompts or
exploring; do not promote its output to "corrected" status without
review.

## Ground-truth pipeline

```bash
python tools/annotation/annotate_sliding_window.py FOOTAGE.mp4
```

Writes `FOOTAGE_raw_annotations.json` next to the input. Treat the output
as **unverified**. After human review, rename the file to
`FOOTAGE_annotations_corrected.json` to mark it as gold-standard for
evaluation.

### Annotation JSON format

```json
{
  "video_metadata": {
    "path": "...",
    "duration_seconds": 1234.5,
    "fps": 0,
    "total_frames": 0
  },
  "segments": [
    {
      "segment_id": 1,
      "type": "in-play",
      "start_ms": 12345,
      "end_ms": 67890,
      "start_frame": 0,
      "end_frame": 0,
      "rally_number": 1
    }
  ]
}
```

The eval framework reads `segments[].type == "in-play"`, `start_ms`,
`end_ms`. Other fields are informational.

## Fast pipeline (prompt iteration)

```bash
python tools/annotation/annotate_fast.py FOOTAGE.mp4
```

Reads `tools/annotation/config.yaml` for prompt and model selection. Edit
that file, not the script.

## Review and correction with Label Studio

The repo does not ship a Label Studio config; the format is simple
enough that you can import the JSON directly as task definitions. The
"label name" you need is `in-play` (segment-level temporal label).

## Exporting clips for training

`tools/annotation/export_clips.py` splits a video into per-rally clips
using a corrected annotation file. Useful for building training data for
your own classifier.

```bash
python tools/annotation/export_clips.py \
    --video FOOTAGE.mp4 \
    --annotations FOOTAGE_annotations_corrected.json \
    --output-dir clips/
```
