import os
import time
from pathlib import Path

from .base import BaseProvider
from shared.llm_utils import extract_json_from_response


class GoogleProvider(BaseProvider):
    """Google Gemini provider for annotation using the new google-genai SDK."""

    def __init__(self, config: dict):
        super().__init__(config)

        api_key = config.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")

        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment "
                "variable or provide api_key in config."
            )

        try:
            from google import genai
            from google.genai import types
            self.types = types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = config.get("model", "gemini-2.5-flash")
        self.thinking_budget = config.get("thinking_budget")

    @property
    def name(self) -> str:
        return "google"

    @property
    def max_images_per_request(self) -> int:
        return self.config.get("max_images_per_request", 3600)

    def annotate_batch(
        self,
        image_paths: list[Path],
        prompt: str,
    ) -> tuple[dict, str]:
        """Send images to Gemini and get annotation response."""
        contents = []

        for image_path in image_paths:
            mime_type = self._get_media_type(image_path)
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            contents.append(
                self.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )

        contents.append(prompt)

        generation_config = {
            "max_output_tokens": self.max_tokens,
        }
        if self.thinking_budget is not None:
            generation_config["thinking_config"] = self.types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        media_resolution = self.config.get("media_resolution")
        if media_resolution:
            resolution_map = {
                "low": self.types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": self.types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": self.types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            resolved = resolution_map.get(media_resolution)
            if resolved:
                generation_config["media_resolution"] = resolved

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.types.GenerateContentConfig(**generation_config),
        )

        response_text = response.text
        try:
            result = extract_json_from_response(response_text)
            return result, response_text
        except ValueError as e:
            # Re-raise with raw response attached for debugging
            raise ValueError(str(e), response_text) from e

    def generate_text(self, prompt: str) -> tuple[dict, str]:
        """Send a text-only prompt to Gemini and get parsed JSON response."""
        generation_config = {
            "max_output_tokens": self.max_tokens,
        }
        if self.thinking_budget is not None:
            generation_config["thinking_config"] = self.types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=self.types.GenerateContentConfig(**generation_config),
        )

        response_text = response.text
        try:
            result = extract_json_from_response(response_text)
            return result, response_text
        except ValueError as e:
            raise ValueError(str(e), response_text) from e

    def upload_video(self, video_path: str | Path, poll_interval: float = 5.0):
        """Upload a video via the Files API and poll until ACTIVE.

        Args:
            video_path: Path to the video file.
            poll_interval: Seconds between status checks.

        Returns:
            A genai File object in ACTIVE state.
        """
        video_path = Path(video_path)
        mime_map = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
        }
        mime_type = mime_map.get(video_path.suffix.lower(), "video/mp4")

        print(f"  Uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)...")
        uploaded = self.client.files.upload(
            file=str(video_path),
            config={"mime_type": mime_type},
        )

        while uploaded.state.name == "PROCESSING":
            time.sleep(poll_interval)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"Video upload failed: state={uploaded.state.name} "
                f"for {video_path.name}"
            )
        print(f"  Upload complete: {uploaded.name}")
        return uploaded

    def delete_file(self, file_name: str) -> None:
        """Delete a file previously uploaded via the Files API."""
        try:
            self.client.files.delete(name=file_name)
            print(f"  Deleted uploaded file: {file_name}")
        except Exception as e:
            print(f"  Warning: failed to delete {file_name}: {e}")

    def generate_with_video(
        self,
        video_file,
        prompt: str,
        parse_json: bool = True,
    ) -> tuple[dict | None, str]:
        """Send a video file + prompt to Gemini and return the response.

        Args:
            video_file: A genai File object (from upload_video).
            prompt: The text prompt.
            parse_json: If True, attempt to parse JSON from response.

        Returns:
            (parsed_json_or_None, raw_response_text)
        """
        generation_config = {
            "max_output_tokens": self.max_tokens,
        }
        if self.thinking_budget is not None:
            generation_config["thinking_config"] = self.types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        media_resolution = self.config.get("media_resolution")
        if media_resolution:
            resolution_map = {
                "low": self.types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": self.types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": self.types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            resolved = resolution_map.get(media_resolution)
            if resolved:
                generation_config["media_resolution"] = resolved

        response = self.client.models.generate_content(
            model=self.model,
            contents=[video_file, prompt],
            config=self.types.GenerateContentConfig(**generation_config),
        )

        response_text = response.text

        if not parse_json:
            return None, response_text

        try:
            result = extract_json_from_response(response_text)
            return result, response_text
        except ValueError:
            return None, response_text

    def _get_media_type(self, image_path: Path) -> str:
        """Get MIME type from file extension."""
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/jpeg")
