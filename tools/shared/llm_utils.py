"""Shared LLM utilities for tools that interact with language model APIs."""

import json
import re


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and parse JSON from LLM response, handling common formatting issues.

    Handles fenced code blocks (closed or not), text around the JSON, and
    several JSON objects (takes the first valid one).

    Args:
        response_text: Raw response text from LLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Strategy 1: Try complete markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try incomplete code block (```json without closing ```)
    code_block_start = re.search(r"```(?:json)?\s*", response_text)
    if code_block_start:
        # Extract everything after the opening fence
        json_str = response_text[code_block_start.end() :]
        # Remove trailing ``` if present anywhere
        if "```" in json_str:
            json_str = json_str[: json_str.index("```")]
        # Try to find the JSON object within
        brace_match = re.search(r"\{[\s\S]*\}", json_str)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find JSON object anywhere in text (greedy - find largest)
    brace_match = re.search(r"\{[\s\S]*\}", response_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try to find the outermost braces and parse
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        json_str = response_text[first_brace : last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 5: Try the whole response as-is
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Truncate response for error message if too long
    preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
    raise ValueError(f"Failed to extract valid JSON from response:\n{preview}")
