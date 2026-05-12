"""Gemini-based rally detection — 3-stage sliding window pipeline.

Pipeline (proven via RL experiment, F1=94.6% on indoor-game-005):
  Stage 1 — Sliding window: 2-min clips at 30s stride; avoids full-video
             timestamp hallucination by querying clip-local coordinates.
  Stage 2 — Boundary refinement: 45s clip per detection, structured output
             for exact start/end ms. Mean IoU=0.789 on matched segments.
  Stage 3 — Gap fill: scan break segments > 15s for missed rallies.

Reference scripts: experiments/downtime-rl/workspace/iter10–iter15.
"""

import json
import logging
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .rally_detector_base import RallyDetectorBase
from .gemini_pipeline import (
    ClipRallies,
    RallyBoundary,
    build_timeline,
    compute_iou,
    delete_file,
    extract_clip,
    get_duration_ms,
    merge_windows,
    upload_and_wait,
)

# Repo layout: <repo>/setoptics/gemini_rally_detector.py and <repo>/tools/
_repo_root = Path(__file__).resolve().parent.parent
_tools_dir = _repo_root / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

logger = logging.getLogger(__name__)

_WINDOW_S = 120          # Stage 1: clip duration
_STRIDE_S = 30           # Stage 1: window stride
_STAGE1_WORKERS = 4      # Stage 1: concurrent window queries
_MIN_RALLY_MS = 2_000
_REFINE_CLIP_S = 45      # Stage 2: clip duration
_REFINE_PRE_S = 2.0      # Stage 2: seconds before anchor to start clip
_REFINE_MAX_START = 20_000
_GAP_MIN_MS = 15_000     # Stage 3: minimum break duration to scan
_DEDUP_IOU = 0.3
_MAX_RETRIES = 3


class GeminiRallyDetector(RallyDetectorBase):
    """Rally detection via 3-stage Gemini sliding window pipeline.

    Achieves F1=94.6% on indoor volleyball (RL experiment, 19 iterations).
    Uses gemini-2.5-flash with structured output for each stage.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int | None = 2048,
        media_resolution: str = "low",
        # Legacy params kept for interface compatibility
        chunk_duration_sec: int = 600,
        video_crf: int = 28,
        video_scale: str = "1280:720",
    ):
        self.api_key = api_key
        self.model = model
        self.thinking_budget = thinking_budget
        self.media_resolution = media_resolution

    def _resolve_media_resolution(self, types_module):
        """Map the string config to the SDK enum."""
        mapping = {
            "low": types_module.MediaResolution.MEDIA_RESOLUTION_LOW,
            "medium": types_module.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            "high": types_module.MediaResolution.MEDIA_RESOLUTION_HIGH,
        }
        if self.media_resolution not in mapping:
            raise ValueError(
                f"Invalid media_resolution: {self.media_resolution!r}. "
                f"Expected one of {list(mapping.keys())}"
            )
        return mapping[self.media_resolution]

    def detect(self, video_path: Path | str) -> tuple[list, dict, list]:
        """Run 3-stage pipeline and return canonical segments.

        ``video_path`` may be a local Path or an ``http(s)://`` URL (e.g.
        a GCS signed URL) — ffmpeg/ffprobe handle both via range requests.

        Returns:
            (canonical_segments, {"segments": canonical_segments}, raw_responses)
        """
        from google import genai

        client = genai.Client(api_key=self.api_key)
        raw_responses: list[str] = []
        if isinstance(video_path, str):
            # Strip query string so log lines stay readable for signed URLs.
            self._label = video_path.split("?", 1)[0].rsplit("/", 1)[-1]
        else:
            self._label = video_path.name

        duration_ms = get_duration_ms(video_path)
        t_start = time.time()
        logger.info("[%s] duration: %.1fs", self._label, duration_ms / 1000)

        with tempfile.TemporaryDirectory(prefix="rally_pipeline_") as tmp:
            work_dir = Path(tmp)

            logger.info("[%s] Stage 1/3 — sliding window (%ds window, %ds stride, %d workers)",
                        self._label, _WINDOW_S, _STRIDE_S, _STAGE1_WORKERS)
            stage1 = self._stage1(client, video_path, duration_ms, work_dir, raw_responses)
            logger.info("[%s] Stage 1/3 done — %d segments (%.0fs elapsed)",
                        self._label, len(stage1), time.time() - t_start)

            logger.info("[%s] Stage 2/3 — boundary refinement (%d segments)",
                        self._label, len(stage1))
            stage2 = self._stage2(client, video_path, duration_ms, stage1, work_dir, raw_responses)
            logger.info("[%s] Stage 2/3 done (%.0fs elapsed)", self._label, time.time() - t_start)

            logger.info("[%s] Stage 3/3 — gap fill", self._label)
            stage3 = self._stage3(client, video_path, duration_ms, stage2, work_dir, raw_responses)
            logger.info("[%s] Stage 3/3 done — %d total segments (%.0fs elapsed)",
                        self._label, len(stage3), time.time() - t_start)

        timeline = build_timeline(stage3, duration_ms)
        rally_num = 0
        for seg in timeline:
            if seg["type"] == "in-play":
                rally_num += 1
                seg.update(rally_number=rally_num, confidence=1.0, description="")
            else:
                seg.update(rally_number=None, confidence=0.0, description="")

        merged = {"segments": timeline}
        return timeline, merged, raw_responses

    # ------------------------------------------------------------------
    # Stage 1: sliding window detection (concurrent)
    # ------------------------------------------------------------------
    def _stage1(self, client, video_path, duration_ms, work_dir, raw_responses):
        from google import genai
        from google.genai import types

        from .prompts import build_sliding_window_prompt

        duration_s = duration_ms / 1000

        # Build the full window list upfront so we know total count for progress
        windows: list[tuple[int, float, float]] = []  # (idx, start_s, clip_dur_s)
        start_s = 0.0
        while start_s < duration_s:
            clip_dur_s = min(_WINDOW_S, duration_s - start_s)
            if clip_dur_s < 5:
                break
            windows.append((len(windows), start_s, clip_dur_s))
            start_s += _STRIDE_S

        total = len(windows)
        lock = threading.Lock()
        completed = [0]
        t0 = time.time()

        def query_window(win_idx: int, start_s: float, clip_dur_s: float) -> list[tuple[int, int]]:
            clip_path = work_dir / f"w{win_idx:04d}.mp4"
            extract_clip(video_path, start_s, clip_dur_s, clip_path)
            prompt = build_sliding_window_prompt(clip_dur_s, start_s)
            clip_start_ms = int(start_s * 1000)
            clip_dur_ms = int(clip_dur_s * 1000)

            found: list[tuple[int, int]] = []
            clip_file = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    clip_file = upload_and_wait(client, clip_path)
                    resp = client.models.generate_content(
                        model=self.model,
                        contents=[
                            types.Part.from_uri(file_uri=clip_file.uri, mime_type="video/mp4"),
                            types.Part.from_text(text=prompt),
                        ],
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ClipRallies,
                            media_resolution=self._resolve_media_resolution(types),
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=self.thinking_budget or 0,
                            ),
                        ),
                    )
                    with lock:
                        raw_responses.append(resp.text)
                    for r in json.loads(resp.text).get("rallies", []):
                        s = max(0, min(clip_dur_ms, int(r["start_ms"])))
                        e = max(s, min(clip_dur_ms, int(r["end_ms"])))
                        if (e - s) >= _MIN_RALLY_MS:
                            found.append((clip_start_ms + s, clip_start_ms + e))
                    break
                except Exception as exc:
                    logger.warning("[%s] window %d attempt %d: %s",
                                   self._label, win_idx, attempt, exc)
                    if attempt < _MAX_RETRIES:
                        time.sleep(60 if "429" in str(exc) else 10)
                finally:
                    if clip_file:
                        delete_file(client, clip_file.name)
                        clip_file = None

            try:
                clip_path.unlink()
            except Exception:
                pass

            with lock:
                completed[0] += 1
                logger.info("[%s] Stage 1/3: window %d/%d done (%.0fs elapsed)",
                            self._label, completed[0], total, time.time() - t0)

            return found

        all_windows: list[tuple[int, int]] = []
        with ThreadPoolExecutor(max_workers=_STAGE1_WORKERS) as executor:
            futures = [executor.submit(query_window, *w) for w in windows]
            for future in as_completed(futures):
                try:
                    all_windows.extend(future.result())
                except Exception as exc:
                    logger.error("[%s] Stage 1 window failed: %s", self._label, exc)

        return merge_windows(all_windows)

    # ------------------------------------------------------------------
    # Stage 2: boundary refinement (concurrent)
    # ------------------------------------------------------------------
    def _stage2(self, client, video_path, duration_ms, segments, work_dir, raw_responses):
        total = len(segments)
        lock = threading.Lock()
        completed = [0]
        t0 = time.time()

        def refine_one(idx: int, anchor_s: int, anchor_e: int) -> tuple[int, tuple[int, int]]:
            clip_start_s = max(0.0, anchor_s / 1000 - _REFINE_PRE_S)
            clip_dur_s = min(_REFINE_CLIP_S, duration_ms / 1000 - clip_start_s)
            clip_path = work_dir / f"r{idx:04d}.mp4"
            extract_clip(video_path, clip_start_s, clip_dur_s, clip_path)
            result = self._refine_boundary(
                client, clip_path, clip_start_s, clip_dur_s, anchor_s, raw_responses,
            )
            try:
                clip_path.unlink()
            except Exception:
                pass
            with lock:
                completed[0] += 1
                logger.info("[%s] Stage 2/3: segment %d/%d refined (%.0fs elapsed)",
                            self._label, completed[0], total, time.time() - t0)
            return (idx, result if result is not None else (anchor_s, anchor_e))

        results: dict[int, tuple[int, int]] = {}
        with ThreadPoolExecutor(max_workers=_STAGE1_WORKERS) as executor:
            futures = [
                executor.submit(refine_one, idx, s, e)
                for idx, (s, e) in enumerate(segments)
            ]
            for future in as_completed(futures):
                try:
                    idx, refined = future.result()
                    results[idx] = refined
                except Exception as exc:
                    logger.error("[%s] Stage 2 segment failed: %s", self._label, exc)

        return [results.get(i, segments[i]) for i in range(total)]

    # ------------------------------------------------------------------
    # Stage 3: gap fill (concurrent scan + concurrent refinement)
    # ------------------------------------------------------------------
    def _stage3(self, client, video_path, duration_ms, segments, work_dir, raw_responses):
        from google import genai
        from google.genai import types

        from .prompts import build_gap_scan_prompt

        timeline = build_timeline(segments, duration_ms)
        breaks = [
            s for s in timeline
            if s["type"] == "break" and (s["end_ms"] - s["start_ms"]) >= _GAP_MIN_MS
        ]
        total_breaks = len(breaks)
        logger.info("[%s] Stage 3/3: scanning %d break segments", self._label, total_breaks)

        lock = threading.Lock()
        all_in_play = list(segments)
        new_found: list[tuple[int, int]] = []
        completed = [0]
        t0 = time.time()

        def scan_gap(idx: int, brk: dict) -> list[tuple[int, int]]:
            brk_start_ms = brk["start_ms"]
            brk_dur_s = (brk["end_ms"] - brk_start_ms) / 1000
            brk_dur_ms = int(brk_dur_s * 1000)
            clip_path = work_dir / f"g{idx:04d}.mp4"
            extract_clip(video_path, brk_start_ms / 1000, brk_dur_s, clip_path)
            prompt = build_gap_scan_prompt(brk_dur_s)

            found: list[tuple[int, int]] = []
            clip_file = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    clip_file = upload_and_wait(client, clip_path)
                    resp = client.models.generate_content(
                        model=self.model,
                        contents=[
                            types.Part.from_uri(file_uri=clip_file.uri, mime_type="video/mp4"),
                            types.Part.from_text(text=prompt),
                        ],
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ClipRallies,
                            media_resolution=self._resolve_media_resolution(types),
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=self.thinking_budget or 0,
                            ),
                        ),
                    )
                    with lock:
                        raw_responses.append(resp.text)
                    for r in json.loads(resp.text).get("rallies", []):
                        s_off = max(0, min(brk_dur_ms, int(r["start_ms"])))
                        e_off = max(s_off, min(brk_dur_ms, int(r["end_ms"])))
                        if (e_off - s_off) < _MIN_RALLY_MS:
                            continue
                        abs_s, abs_e = brk_start_ms + s_off, brk_start_ms + e_off
                        with lock:
                            if any(compute_iou(abs_s, abs_e, es, ee) >= _DEDUP_IOU for es, ee in all_in_play):
                                continue
                            all_in_play.append((abs_s, abs_e))
                        found.append((abs_s, abs_e))
                    break
                except Exception as exc:
                    logger.warning("[%s] gap %d attempt %d: %s",
                                   self._label, idx, attempt, exc)
                    if attempt < _MAX_RETRIES:
                        time.sleep(60 if "429" in str(exc) else 10)
                finally:
                    if clip_file:
                        delete_file(client, clip_file.name)
                        clip_file = None
            try:
                clip_path.unlink()
            except Exception:
                pass
            with lock:
                completed[0] += 1
                logger.info("[%s] Stage 3/3: gap %d/%d scanned (%.0fs elapsed)",
                            self._label, completed[0], total_breaks, time.time() - t0)
            return found

        with ThreadPoolExecutor(max_workers=_STAGE1_WORKERS) as executor:
            futures = [
                executor.submit(scan_gap, idx, brk)
                for idx, brk in enumerate(breaks)
            ]
            for future in as_completed(futures):
                try:
                    new_found.extend(future.result())
                except Exception as exc:
                    logger.error("[%s] Stage 3 gap scan failed: %s", self._label, exc)

        logger.info("[%s] Stage 3/3: %d new segments found in gaps",
                    self._label, len(new_found))
        if not new_found:
            return segments

        # Refine new detections (concurrent)
        refined_lock = threading.Lock()
        refined_completed = [0]
        refined_total = len(new_found)

        def refine_gap(idx: int, anchor_s: int, anchor_e: int) -> tuple[int, tuple[int, int]]:
            clip_start_s = max(0.0, anchor_s / 1000 - _REFINE_PRE_S)
            clip_dur_s = min(_REFINE_CLIP_S, duration_ms / 1000 - clip_start_s)
            clip_path = work_dir / f"gn{idx:04d}.mp4"
            extract_clip(video_path, clip_start_s, clip_dur_s, clip_path)
            result = self._refine_boundary(
                client, clip_path, clip_start_s, clip_dur_s, anchor_s, raw_responses,
            )
            try:
                clip_path.unlink()
            except Exception:
                pass
            with refined_lock:
                refined_completed[0] += 1
                logger.info("[%s] Stage 3/3: refine %d/%d done (%.0fs elapsed)",
                            self._label, refined_completed[0], refined_total, time.time() - t0)
            return (idx, result if result is not None else (anchor_s, anchor_e))

        results: dict[int, tuple[int, int]] = {}
        with ThreadPoolExecutor(max_workers=_STAGE1_WORKERS) as executor:
            futures = [
                executor.submit(refine_gap, idx, s, e)
                for idx, (s, e) in enumerate(new_found)
            ]
            for future in as_completed(futures):
                try:
                    idx, refined = future.result()
                    results[idx] = refined
                except Exception as exc:
                    logger.error("[%s] Stage 3 refine failed: %s", self._label, exc)

        refined_new = [results.get(i, new_found[i]) for i in range(refined_total)]
        return list(segments) + refined_new

    # ------------------------------------------------------------------
    # Shared: refine a single segment's boundaries via 45s clip
    # ------------------------------------------------------------------
    def _refine_boundary(self, client, clip_path, clip_start_s, clip_dur_s, anchor_s, raw_responses):
        from google import genai
        from google.genai import types

        from .prompts import build_boundary_refine_prompt

        prompt = build_boundary_refine_prompt(clip_dur_s, _REFINE_PRE_S)
        clip_start_ms = int(clip_start_s * 1000)

        clip_file = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                clip_file = upload_and_wait(client, clip_path)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_uri(file_uri=clip_file.uri, mime_type="video/mp4"),
                        types.Part.from_text(text=prompt),
                    ],
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RallyBoundary,
                        media_resolution=self._resolve_media_resolution(types),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget or 0,
                        ),
                    ),
                )
                raw_responses.append(resp.text)
                raw = json.loads(resp.text)
                s_off = int(raw.get("start_offset_ms", -1))
                e_off = int(raw.get("end_offset_ms", -1))
                if s_off < 0 or e_off < 0:
                    return None
                dur = e_off - s_off
                if dur < _MIN_RALLY_MS or dur > 40_000 or s_off > _REFINE_MAX_START:
                    return None
                return (clip_start_ms + s_off, clip_start_ms + e_off)
            except Exception as exc:
                logger.warning("Refine attempt %d: %s", attempt, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(60 if "429" in str(exc) else 10)
            finally:
                if clip_file:
                    delete_file(client, clip_file.name)
                    clip_file = None
        return None
