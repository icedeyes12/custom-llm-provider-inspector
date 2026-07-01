"""Provider detection and scanning logic."""

from __future__ import annotations

from typing import Any

from core.models import ModelInfo, ProviderSession


KNOWN_PROVIDERS: dict[str, str] = {
    "api.openai.com": "OpenAI",
    "openrouter.ai": "OpenRouter",
    "api.litellm.ai": "LiteLLM",
    "api.cerebras.ai": "Cerebras",
    "integrate.api.nvidia.com": "NVIDIA NIM",
    "api.anthropic.com": "Anthropic (OpenAI-compat proxy)",
    "generativelanguage.googleapis.com": "Google (OpenAI-compat proxy)",
    "api.mistral.ai": "Mistral (OpenAI-compat proxy)",
}


def detect_provider(base_url: str) -> str | None:
    """Heuristic provider name from base URL host."""
    for host, name in KNOWN_PROVIDERS.items():
        if host in base_url:
            return name
    return None


def parse_model(raw: dict[str, Any]) -> ModelInfo:
    """Convert a raw /models entry into a ModelInfo."""
    return ModelInfo(
        id=raw.get("id", "unknown"),
        owner=raw.get("owned_by", raw.get("owner", "unknown")),
        created=int(raw.get("created", 0)),
        raw=raw,
    )


def refresh_models(
    session_obj: ProviderSession, raw_models: list[dict[str, Any]]
) -> None:
    """Populate session.models from raw /models response."""
    session_obj.models = [parse_model(m) for m in raw_models]


def get_provider_summary(session_obj: ProviderSession) -> dict[str, Any]:
    """Build a summary dict for display."""
    provider = detect_provider(session_obj.base_url)
    return {
        "base_url": session_obj.base_url,
        "provider": provider or "unknown",
        "total_models": len(session_obj.models),
        "models": session_obj.models,
    }
