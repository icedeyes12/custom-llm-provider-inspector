"""Low-level API client for OpenAI-compatible providers."""

from __future__ import annotations

import time
from typing import Any

import requests

from core.models import LatencyResult


class APIClient:
    """Synchronous HTTP client for OpenAI-compatible API endpoints."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # -- raw helpers -----------------------------------------------------

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.get(url, timeout=self.timeout, **kwargs)

    def post(
        self, path: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.post(url, json=json, timeout=self.timeout, **kwargs)

    # -- typed helpers ---------------------------------------------------

    def list_models(self) -> tuple[list[dict[str, Any]], LatencyResult]:
        """GET /models. Returns (raw_model_list, latency)."""
        lat = LatencyResult()
        start = time.monotonic()
        try:
            resp = self.get("/models")
            lat.status_code = resp.status_code
            resp.raise_for_status()
            data = resp.json()
            lat.total_ms = (time.monotonic() - start) * 1000
            return data.get("data", []), lat
        except requests.RequestException as e:
            lat.total_ms = (time.monotonic() - start) * 1000
            lat.error = str(e)
            return [], lat

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[requests.Response, LatencyResult]:
        """POST /chat/completions. Returns (response, latency)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if max_tokens:
            payload["max_tokens"] = max_tokens

        lat = LatencyResult()
        start = time.monotonic()
        try:
            resp = self.post("/chat/completions", json=payload)
            lat.status_code = resp.status_code
            lat.total_ms = (time.monotonic() - start) * 1000
            return resp, lat
        except requests.RequestException as e:
            lat.total_ms = (time.monotonic() - start) * 1000
            lat.error = str(e)
            # Return a synthetic response-like object
            raise

    def streaming_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[requests.Response, LatencyResult]:
        """POST /chat/completions with stream=True. Caller iterates lines."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        lat = LatencyResult()
        start = time.monotonic()
        try:
            resp = self.post("/chat/completions", json=payload, stream=True)
            lat.status_code = resp.status_code
            lat.total_ms = (time.monotonic() - start) * 1000
            return resp, lat
        except requests.RequestException as e:
            lat.total_ms = (time.monotonic() - start) * 1000
            lat.error = str(e)
            raise
