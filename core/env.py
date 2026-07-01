"""Resolve values from ~/.env file (no external dependencies)."""

from __future__ import annotations

import os
from pathlib import Path

_env_cache: dict[str, str] | None = None


def _load_dotenv() -> dict[str, str]:
    """Parse ~/.env into a dict. Skips comments and blank lines."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache

    env_path = Path.home() / ".env"
    result: dict[str, str] = {}
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    _env_cache = result
    return _env_cache


def resolve_env(value: str) -> str:
    """Resolve a value that may be an env var reference.

    If *value* matches a key in ~/.env or os.environ, return the resolved value.
    Otherwise return *value* unchanged.

    Supports:
      - Bare var name:  "OPENAI_URL" -> looks up ~/.env then os.environ
      - $ prefix:       "$OPENAI_KEY" -> same lookup, strips the $
    """
    if not value:
        return value

    key = value.removeprefix("$")
    if key == value and " " in key:
        return value

    dotenv = _load_dotenv()
    resolved = dotenv.get(key) or os.environ.get(key)
    return resolved if resolved is not None else value


def resolve_url_key_pair(url_input: str) -> tuple[str, str | None]:
    """Resolve a URL input. If it's an env var, auto-attempt the matching KEY.

    Returns (resolved_url, resolved_key_or_None).

    Resolution logic:
      1. Resolve url_input as a direct value or env var
      2. If the input was an env var (resolved != raw), try to find a matching
         KEY by replacing _URL with _KEY in the original var name
      3. If the matching KEY var exists, auto-resolve it
    """
    resolved_url = resolve_env(url_input)
    auto_key = None

    # Did the input resolve? (i.e. was it an env var name?)
    if resolved_url != url_input:
        # Try to auto-resolve the paired key
        raw_key = url_input.removeprefix("$")
        # Replace _URL with _KEY (case-insensitive pattern)
        if "_URL" in raw_key.upper():
            # Find where _URL is (case-sensitive match on position)
            idx = raw_key.upper().find("_URL")
            candidate = raw_key[:idx] + "_KEY" + raw_key[idx + 4:]
            auto_key = resolve_env(candidate)
            if auto_key == candidate:
                # Not found — don't return the candidate as-is
                auto_key = None

    return resolved_url, auto_key


def has_env_file() -> bool:
    """Check whether ~/.env exists."""
    return (Path.home() / ".env").is_file()


def has_url_vars() -> bool:
    """Check whether ~/.env contains any *-URL-like keys."""
    dotenv = _load_dotenv()
    return any("_URL" in k.upper() for k in dotenv)
