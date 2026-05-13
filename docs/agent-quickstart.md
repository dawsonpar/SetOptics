# Agent quickstart

The whole "clone → run on my footage → get a rallies-only MP4" workflow,
runnable through any agent (Claude Code, Cursor, Codex, Gemini). Paste the
prompt below verbatim and let the agent do the rest.

## Prerequisites

The agent has shell access on a machine with `git`, `python3.11+`, and
`ffmpeg` already installed. (The agent will install Python packages
itself via `./setup.sh`.)

## The one-shot prompt

```text
Clone https://github.com/dawsonpar/SetOptics into the current directory,
run ./setup.sh to set it up (you can skip the GEMINI_API_KEY prompt for
now), then run signal-based rally detection on the video at PATH/TO/MY/
VIDEO.mp4 and export a rallies-only MP4 next to the input video. Use the
canonical scripts in scripts/ — do not invent new wrappers. Read AGENTS.md
first.
```

The agent will:

1. `git clone https://github.com/dawsonpar/SetOptics.git && cd SetOptics`
2. `./setup.sh` (creates `.venv`, installs deps, skips the API-key prompt)
3. `source .venv/bin/activate`
4. `python scripts/signal_rally_detector.py --video VIDEO.mp4 --output VIDEO_signal.json`
5. `python scripts/export_rallies.py --segments VIDEO_signal.json --video VIDEO.mp4 --output VIDEO_rallies.mp4`

Total time on a 30-second clip: under 90 seconds, end to end.

## If you want higher accuracy

The signal-only detector is the free, local, dependency-free path. F1
hovers around 47% on indoor footage (good enough for highlight reels;
miss-rate is mostly false-positives on long stretches of warm-up motion).

For higher accuracy, point the agent at the LLM path instead:

```text
Same as above, but use tools/annotation/annotate_sliding_window.py instead
of the signal detector. Put my GEMINI_API_KEY in .env first (I'll paste it).
```

This bumps F1 to ~94% but costs API tokens and runs 3-4x slower.

## If the rally boundaries feel tight

Add padding so the export keeps a half-second of context on each side:

```text
Re-run scripts/export_rallies.py on the same JSON but pass --pad 0.5 so
the output keeps a beat of context before and after each rally.
```

## Where things land

After a clean run, your project directory looks like:

```
.
├── SetOptics/                       # cloned repo
│   ├── .venv/                       # python env (gitignored)
│   ├── ...
│   └── scripts/
└── VIDEO.mp4                        # your input
    VIDEO_signal.json                # detection output (segments)
    VIDEO_rallies.mp4                # rallies concatenated, breaks cut
```

If `VIDEO_rallies.mp4` is shorter than expected, the signal detector
likely missed a rally — re-run with `--fusion-threshold 0.35` (default is
`0.45`) or try the ensemble/LLM modes per `docs/rally-detection.md`.
