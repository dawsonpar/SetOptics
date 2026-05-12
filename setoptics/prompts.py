"""Video-based rally detection prompts for Gemini Files API."""


def build_sliding_window_prompt(clip_duration_s: float, clip_start_s: float) -> str:
    """Stage 1 prompt: detect all rallies in a 2-minute sliding window clip.

    Proven approach from RL experiment (iter10): avoids full-video timestamp
    hallucination by querying short clips with clip-local ms coordinates.
    Achieved F1=68.6% alone; +11pp over prior single-pass methods.
    """
    dur_ms = int(clip_duration_s * 1000)
    return (
        f"You are watching a {clip_duration_s:.0f}-second clip of an "
        f"indoor volleyball game. The clip starts at {clip_start_s:.0f}s "
        f"into the full match.\n\n"
        "TASK: Identify every volleyball rally (in-play period) in this clip.\n\n"
        "DEFINITIONS:\n"
        "- A RALLY starts when the server strikes the ball (immediately after "
        "the referee's short start whistle).\n"
        "- A RALLY ends when the ball becomes dead: hits floor, goes out, fault "
        "called, or referee blows the end-of-point whistle.\n"
        "- A rally is ONE continuous segment, NOT individual hits.\n"
        "- Typical rally duration: 3–25 seconds. Min 2 seconds.\n\n"
        "OUTPUT: List all rallies as (start_ms, end_ms) pairs in milliseconds "
        "from the START of this clip (NOT from the full video start).\n\n"
        f"This clip is {clip_duration_s:.0f} seconds long, so all timestamps "
        f"must be between 0 and {dur_ms}ms.\n\n"
        "If there are no rallies in this clip, return an empty list.\n"
        "Do NOT merge consecutive rallies into one segment.\n\n"
        f'Output format: {{"rallies": [{{"start_ms": 5000, "end_ms": 18000}}, ...]}}'
    )


def build_boundary_refine_prompt(
    clip_duration_s: float,
    pre_anchor_s: float = 2.0,
) -> str:
    """Stage 2 prompt: refine start/end boundaries of a single predicted rally.

    Upload a 45s clip starting ~2s before the predicted rally start.
    Achieved mean IoU=0.789 on matched segments in RL experiment (iter11).
    """
    dur_ms = int(clip_duration_s * 1000)
    lookahead_s = pre_anchor_s + 3
    return (
        f"You are watching a {clip_duration_s:.0f}-second clip of an "
        "indoor volleyball game.\n\n"
        f"A volleyball rally is predicted to start near the beginning of this clip "
        f"(within the first {lookahead_s:.0f} seconds). Your task is to find the "
        "EXACT start and end of that rally.\n\n"
        "DEFINITIONS:\n"
        "- RALLY START: the moment the server strikes the ball (immediately after "
        "the referee's short start whistle). NOT when players are walking or "
        "standing — the instant the ball is served.\n"
        "- RALLY END: the moment the ball becomes dead — it hits the floor, goes "
        "out of bounds, a fault is called, or the referee blows the end-of-point "
        "whistle.\n"
        "- ONE RALLY ONLY: identify the single main rally that begins near the "
        "start of this clip. Do not merge consecutive rallies.\n\n"
        "OUTPUT: Return start_offset_ms and end_offset_ms as milliseconds from "
        f"the START of this clip (not from the full video). Both values must be "
        f"between 0 and {dur_ms}ms.\n\n"
        "If no clear rally is visible near the start of this clip, return -1 for "
        "both fields."
    )


def build_gap_scan_prompt(clip_duration_s: float) -> str:
    """Stage 3 prompt: check a detected break segment for missed rallies.

    Used in gap fill pass (iter15): scanned break segments > 15s and recovered
    missed rallies, adding +13.2pp F1 (70.6% → 83.8%).
    """
    dur_ms = int(clip_duration_s * 1000)
    return (
        f"You are analyzing a {clip_duration_s:.0f}-second clip from an "
        "indoor volleyball game.\n\n"
        "This clip is a BREAK period — the full-video detector did NOT detect "
        "any rallies here. Your task is to double-check: are there any volleyball "
        "rallies in this clip that were missed?\n\n"
        "DEFINITIONS:\n"
        "- RALLY / IN-PLAY: the ball is being actively played — from the server "
        "striking the ball until the ball becomes dead (hits the floor, goes out, "
        "fault is called, or referee whistles end of point).\n"
        "- BREAK: players are standing, walking, waiting between points, during "
        "timeouts, between sets, or during substitutions.\n\n"
        "TASK: List every rally segment in this clip. For each rally, provide "
        "start_ms and end_ms as milliseconds from the START of this clip "
        f"(0 to {dur_ms}ms).\n\n"
        "If there are NO rallies in this clip, return an empty list.\n"
        "Be conservative: only include segments where the ball is clearly "
        "in active play."
    )


def build_video_rally_prompt() -> str:
    """Build the single-pass video rally detection prompt.

    Proven to achieve avg F1=75.3-75.6% on indoor volleyball games
    using native video understanding (Gemini Files API + 10min chunks).
    """
    return """You are an expert volleyball analyst. Identify every rally \
in this video.

## Rally Definition

A rally:
- STARTS at serve contact. Look for: a player standing alone behind \
the back line, ball toss upward followed by arm swing, ball arcing \
from one side of the net toward the other, other players in receiving \
formation (spread out, low stance).
- CONTINUES while the ball is legally in play, including multiple \
attack-dig exchanges, scramble plays, and saves.
- ENDS when the ball hits the ground, goes out of bounds, or the \
referee signals (arm raised straight up = rally over, arm extended \
to one side = point awarded, players immediately relaxing posture).

## Temporal Continuity (CRITICAL)

A rally is ONE continuous segment. Do NOT split a single rally into \
multiple segments even if:
- The ball crosses the net multiple times
- There are multiple attack-dig sequences
- The rally is long or chaotic

If uncertain whether a rally has ended, assume it CONTINUES unless \
there are clear break signals.

## Output Format

Output JSON with contiguous segments covering the ENTIRE video from \
0:00 to the end. Every second must belong to either "in-play" or "break".

{
  "segments": [
    {
      "type": "in-play" | "break",
      "timestamp_start": "MM:SS",
      "timestamp_end": "MM:SS",
      "rally_number": <int or null>,
      "description": "brief summary of what happens"
    }
  ]
}

Rules:
- Segments must be contiguous with no gaps or overlaps
- Rally numbers start at 1, increment for each in-play segment
- Break segments have rally_number: null
- Be precise with timestamps — within 1 second accuracy"""
