from .base import BaseProvider
from .google_provider import GoogleProvider

PROVIDERS = {
    "google": GoogleProvider,
}


def get_provider(name: str, config: dict) -> BaseProvider:
    """Get a provider instance by name."""
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")

    provider_class = PROVIDERS[name]
    provider_config = config.get(name, {})
    return provider_class(provider_config)
