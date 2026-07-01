"""UI theme and shared Rich styling."""

from __future__ import annotations

from rich.theme import Theme

PALETTE = Theme(
    {
        "info": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red on default",
        "muted": "dim white",
        "accent": "bold magenta",
        "header": "bold white on dark_blue",
        "panel.border": "bright_blue",
        "table.header": "bold bright_white",
        "status.ok": "bold green",
        "status.fail": "bold red",
        "status.partial": "bold yellow",
        "status.active": "bold cyan",
        "status.pending": "dim white",
    }
)

SPINNER_STYLE = "dots"
