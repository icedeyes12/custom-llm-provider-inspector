"""Core data models for Provider Inspector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Represents a single model returned by GET /models."""

    id: str
    owner: str = "unknown"
    created: int = 0
    capabilities: dict[str, bool] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.id


@dataclass
class LatencyResult:
    """Result of a latency measurement."""

    first_token_ms: float | None = None
    total_ms: float = 0.0
    status_code: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code < 400


@dataclass
class ChatResult:
    """Result of a chat completion request."""

    content: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamResult:
    """Result of a streaming request."""

    content: str = ""
    chunks: int = 0
    first_token_ms: float | None = None
    total_ms: float = 0.0
    status_code: int = 0
    error: str | None = None


@dataclass
class ToolCallResult:
    """Result of a tool calling test."""

    supported: bool = False
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None


@dataclass
class VisionResult:
    """Result of a vision (multimodal) test."""

    supported: bool = False
    response_text: str = ""
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None


@dataclass
class JsonModeResult:
    """Result of a JSON mode test."""

    supported: bool = False
    response_text: str = ""
    parsed_json: bool = False
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None


@dataclass
class CapabilityScan:
    """Full capability scan result for a single model."""

    model_id: str
    chat: ChatResult = field(default_factory=ChatResult)
    streaming: StreamResult = field(default_factory=StreamResult)
    tools: ToolCallResult = field(default_factory=ToolCallResult)
    vision: VisionResult = field(default_factory=VisionResult)
    json_mode: JsonModeResult = field(default_factory=JsonModeResult)


@dataclass
class ProviderSession:
    """Holds the connection state and results for the current session."""

    base_url: str = ""
    api_key: str = ""
    models: list[ModelInfo] = field(default_factory=list)
    selected_model: str = ""
    scan_results: list[CapabilityScan] = field(default_factory=list)
    benchmark_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_connected(self) -> bool:
        return bool(self.base_url and self.api_key)

    @property
    def model_ids(self) -> list[str]:
        return [m.id for m in self.models]

    def get_model(self, model_id: str) -> ModelInfo | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None
