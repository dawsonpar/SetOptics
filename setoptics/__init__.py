"""SetOptics rally detection — public package.

Importable modules:
  - ball_tracker.BallTracker — YOLO + BoT-SORT ball tracking
  - gemini_rally_detector.GeminiRallyDetector — 3-stage Gemini rally pipeline
  - gemini_pipeline — helpers (clip extraction, timeline merge, IoU)
  - prompts — Gemini prompt templates

CLI entry points live under scripts/ at the repo root.
"""

__version__ = "0.1.0"
