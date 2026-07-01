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

    @property
    def observed(self) -> bool:
        return self.latency.ok and bool(self.content)


@dataclass
class StreamResult:
    """Result of a streaming request."""

    content: str = ""
    chunks: int = 0
    first_token_ms: float | None = None
    total_ms: float = 0.0
    status_code: int = 0
    error: str | None = None

    @property
    def observed(self) -> bool:
        return self.first_token_ms is not None and not self.error


@dataclass
class ToolCallResult:
    """Result of a tool calling test."""

    supported: bool = False
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None

    @property
    def observed(self) -> bool:
        return self.supported


@dataclass
class VisionResult:
    """Result of a vision (multimodal) test."""

    supported: bool = False
    response_text: str = ""
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None

    @property
    def observed(self) -> bool:
        return self.supported


@dataclass
class JsonModeResult:
    """Result of a JSON mode test."""

    supported: bool = False
    response_text: str = ""
    parsed_json: bool = False
    latency: LatencyResult = field(default_factory=lambda: LatencyResult())
    error: str | None = None

    @property
    def observed(self) -> bool:
        return self.supported


@dataclass
class CapabilityFinding:
    """One discovered capability entry inside an inspection report."""

    name: str
    observed: bool
    note: str = ""
    sample: str = ""
    latency_ms: float | None = None
    first_token_ms: float | None = None
    chunks: int = 0


@dataclass
class InspectionReport:
    """Presentation-oriented inspection report for one model."""

    model_id: str
    owner: str = "unknown"
    provider: str = ""
    base_url: str = ""
    latency_ms: float | None = None
    findings: list[CapabilityFinding] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def observed_count(self) -> int:
        return sum(1 for finding in self.findings if finding.observed)

    @property
    def discovered_summary(self) -> str:
        parts: list[str] = []
        for finding in self.findings:
            icon = "✓" if finding.observed else "✗"
            parts.append(f"{icon} {finding.name}")
        return "  ".join(parts)


@dataclass
class CapabilityScan:
    """Compatibility container kept for scan logic and exports."""

    model_id: str
    chat: ChatResult = field(default_factory=ChatResult)
    streaming: StreamResult = field(default_factory=StreamResult)
    tools: ToolCallResult = field(default_factory=ToolCallResult)
    vision: VisionResult = field(default_factory=VisionResult)
    json_mode: JsonModeResult = field(default_factory=JsonModeResult)

    @property
    def report(self) -> InspectionReport:
        findings = [
            CapabilityFinding(
                name="Chat",
                observed=self.chat.observed,
                note="chat completion observed" if self.chat.observed else self.chat.latency.error or "no chat response",
                sample=self.chat.content,
                latency_ms=self.chat.latency.total_ms,
            ),
            CapabilityFinding(
                name="Stream",
                observed=self.streaming.observed,
                note="streaming observed" if self.streaming.observed else self.streaming.error or "no stream output",
                sample=self.streaming.content,
                latency_ms=self.streaming.total_ms,
                first_token_ms=self.streaming.first_token_ms,
                chunks=self.streaming.chunks,
            ),
            CapabilityFinding(
                name="Tools",
                observed=self.tools.observed,
                note="tool calls observed" if self.tools.observed else self.tools.error or "tool calls unsupported",
                sample=self.tools.response_text,
                latency_ms=self.tools.latency.total_ms,
            ),
            CapabilityFinding(
                name="Vision",
                observed=self.vision.observed,
                note="vision observed" if self.vision.observed else self.vision.error or "vision unsupported",
                sample=self.vision.response_text,
                latency_ms=self.vision.latency.total_ms,
            ),
            CapabilityFinding(
                name="JSON",
                observed=self.json_mode.observed,
                note="json mode observed" if self.json_mode.observed else self.json_mode.error or "json mode unsupported",
                sample=self.json_mode.response_text,
                latency_ms=self.json_mode.latency.total_ms,
            ),
        ]
        return InspectionReport(model_id=self.model_id, findings=findings)


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
