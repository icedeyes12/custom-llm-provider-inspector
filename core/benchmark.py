"""Benchmark and capability test functions."""

from __future__ import annotations

import json
import time

from core.api import APIClient
from core.models import (
    CapabilityScan,
    ChatResult,
    JsonModeResult,
    StreamResult,
    ToolCallResult,
    VisionResult,
)

# A tiny 1x1 PNG hosted on a public CDN for vision testing
VISION_TEST_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png"

SIMPLE_PROMPT = "Say hello in one short sentence."

TOOL_PAYLOAD = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

JSON_MODE_PAYLOAD = {"type": "json_object"}

JSON_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Respond with a JSON object containing a single key 'greeting' "
            "and a short greeting string."
        ),
    },
]


def test_chat(client: APIClient, model: str, prompt: str = SIMPLE_PROMPT) -> ChatResult:
    """Non-streaming chat completion test."""
    result = ChatResult()
    messages = [{"role": "user", "content": prompt}]
    try:
        resp, lat = client.chat_completion(model, messages, max_tokens=128)
        result.latency = lat
        if resp is None or resp.status_code >= 400:
            error_text = resp.text if resp else "No response"
            result.latency.error = (
                f"HTTP {resp.status_code if resp else '?'}: {error_text[:200]}"
            )
            return result
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        result.content = msg.get("content", "")
        result.finish_reason = choice.get("finish_reason", "")
        result.usage = data.get("usage", {})
        result.raw = data
    except Exception as e:  # noqa: BLE001
        result.latency.error = str(e)
    return result


def test_streaming(client: APIClient, model: str) -> StreamResult:
    """Streaming chat completion test. Measures first-token latency."""
    result = StreamResult()
    messages = [{"role": "user", "content": SIMPLE_PROMPT}]
    try:
        resp, lat = client.streaming_chat(model, messages, max_tokens=128)
        result.status_code = lat.status_code
        result.latency = lat

        first_token_time: float | None = None
        full_text: list[str] = []
        chunk_count = 0
        start = time.monotonic()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                if first_token_time is None:
                    first_token_time = (time.monotonic() - start) * 1000
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_text.append(token)
                        chunk_count += 1
            except json.JSONDecodeError:
                continue

        result.content = "".join(full_text)
        result.chunks = chunk_count
        result.first_token_ms = first_token_time
        result.total_ms = (time.monotonic() - start) * 1000

        if resp.status_code >= 400:
            result.error = f"HTTP {resp.status_code}"
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
    return result


def test_tool_calling(client: APIClient, model: str) -> ToolCallResult:
    """Send a prompt that requires tool use. Detect provider behaviour."""
    result = ToolCallResult()
    try:
        resp, lat = client.chat_completion(
            model, TOOL_MESSAGES, tools=TOOL_PAYLOAD, max_tokens=256
        )
        result.latency = lat
        if resp is None or resp.status_code >= 400:
            text = resp.text if resp else ""
            if resp and resp.status_code == 422:
                result.error = "Tool calling rejected (HTTP 422)"
                return result
            if resp and resp.status_code >= 400:
                result.error = f"HTTP {resp.status_code}: {text[:200]}"
                return result
            result.error = "No response"
            return result
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        result.response_text = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        result.tool_calls = tool_calls
        result.supported = len(tool_calls) > 0 or "function_call" in str(data)
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
    return result


def test_vision(client: APIClient, model: str) -> VisionResult:
    """Send an image_url payload. Detect if vision is supported."""
    result = VisionResult()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image? Reply in one sentence.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": VISION_TEST_IMAGE},
                },
            ],
        }
    ]
    try:
        resp, lat = client.chat_completion(model, messages, max_tokens=64)
        result.latency = lat
        if resp is None or resp.status_code >= 400:
            text = resp.text if resp else ""
            result.supported = False
            result.error = f"HTTP {resp.status_code if resp else '?'}: {text[:200]}"
            return result
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        result.response_text = choice.get("message", {}).get("content", "")
        result.supported = False
        if result.response_text:
            lower = result.response_text.lower()
            rejection_phrases = [
                "cannot",
                "can't see",
                "no image",
                "don't support",
                "does not support",
                "unsupported",
                "unable to process",
            ]
            result.supported = not any(p in lower for p in rejection_phrases)
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
    return result


def test_json_mode(client: APIClient, model: str) -> JsonModeResult:
    """Test response_format=json_object support."""
    result = JsonModeResult()
    try:
        resp, lat = client.chat_completion(
            model,
            JSON_MESSAGES,
            response_format=JSON_MODE_PAYLOAD,
            max_tokens=128,
        )
        result.latency = lat
        if resp is None or resp.status_code >= 400:
            text = resp.text if resp else ""
            result.error = f"HTTP {resp.status_code if resp else '?'}: {text[:200]}"
            return result
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        result.response_text = choice.get("message", {}).get("content", "")
        if result.response_text:
            try:
                json.loads(result.response_text)
                result.parsed_json = True
                result.supported = True
            except json.JSONDecodeError:
                result.supported = True
                result.parsed_json = False
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
    return result


def full_scan(client: APIClient, model: str) -> CapabilityScan:
    """Run all capability tests for one model."""
    scan = CapabilityScan(model_id=model)
    scan.chat = test_chat(client, model)
    scan.streaming = test_streaming(client, model)
    scan.tools = test_tool_calling(client, model)
    scan.vision = test_vision(client, model)
    scan.json_mode = test_json_mode(client, model)
    return scan
