"""Capability detection heuristics."""

from __future__ import annotations

from core.api import APIClient
from core.benchmark import (
    test_chat,
    test_json_mode,
    test_streaming,
    test_tool_calling,
    test_vision,
)


def detect_capabilities(client: APIClient, model: str) -> dict[str, bool]:
    """Quick capability detection for a single model (cached lightweight version).

    Runs only chat + tools + vision to give a fast summary.
    """
    caps: dict[str, bool] = {
        "chat": False,
        "streaming": False,
        "tools": False,
        "vision": False,
        "json_mode": False,
    }
    chat = test_chat(client, model)
    if chat.latency.ok and chat.content:
        caps["chat"] = True

    stream = test_streaming(client, model)
    if stream.first_token_ms is not None and not stream.error:
        caps["streaming"] = True

    tools = test_tool_calling(client, model)
    if tools.supported:
        caps["tools"] = True

    vision = test_vision(client, model)
    if vision.supported:
        caps["vision"] = True

    jm = test_json_mode(client, model)
    if jm.supported:
        caps["json_mode"] = True

    return caps
