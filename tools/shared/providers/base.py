from abc import ABC, abstractmethod
from pathlib import Path


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens", 4096)

    @abstractmethod
    def annotate_batch(
        self,
        image_paths: list[Path],
        prompt: str,
    ) -> tuple[dict, str]:
        """
        Send a batch of images with a prompt and get annotation response.

        Args:
            image_paths: List of paths to frame images
            prompt: The annotation prompt

        Returns:
            Tuple of (parsed JSON response, raw response text)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    def generate_text(self, prompt: str) -> tuple[dict, str]:
        """Send a text-only prompt and get a parsed JSON response.

        Args:
            prompt: The text prompt (no images).

        Returns:
            Tuple of (parsed JSON response, raw response text).
        """
        raise NotImplementedError(
            f"{self.name} provider does not support text-only generation"
        )

    @property
    def max_images_per_request(self) -> int:
        """Maximum images the provider supports per request."""
        return 20  # Default, override in subclasses
