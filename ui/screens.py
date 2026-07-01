"""Screen renderers — cohesive terminal UX.

Navigation contract:
  9 = Back,  0 = Exit  (every screen)
  Main menu: 9 = Disconnect, 0 = Exit
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.align import Align
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.table import Table

from core.benchmark import (
    test_chat,
    test_json_mode,
    test_streaming,
    test_tool_calling,
    test_vision,
)
from core.env import has_env_file, resolve_url_key_pair
from core.models import CapabilityScan
from core.scanner import detect_provider
from core.session import session
from ui.app import Screen
from ui.theme import SPINNER_STYLE

if TYPE_CHECKING:
    from ui.app import ProviderInspectorApp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAV_BACK = 9
NAV_EXIT = 0
REPORTS_DIR = Path("reports")
MODEL_ID_MAX = 22

SINGLE_TESTS = [
    ("Chat", "test_chat"),
    ("Stream", "test_streaming"),
    ("Tools", "test_tool_calling"),
    ("Vision", "test_vision"),
    ("JSON Mode", "test_json_mode"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _truncate(s: str, max_len: int = MODEL_ID_MAX) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 2] + ".."


def _ask(valid: set[int] | None = None) -> int:
    """Universal `>` prompt — no choices shown, no default."""
    while True:
        raw = Prompt.ask("  >")
        raw = raw.strip()
        if not raw:
            continue
        try:
            val = int(raw)
        except ValueError:
            app_console = _ask._console  # type: ignore[attr-defined]
            if app_console:
                app_console.print("  [dim]?[/]")
            continue
        if valid is not None and val not in valid:
            app_console = _ask._console  # type: ignore[attr-defined]
            if app_console:
                app_console.print("  [dim]?[/]")
            continue
        return val


def _nav(console, back_label: str = "Back") -> None:
    """Standard nav footer."""
    console.print()
    console.rule(style="dim")
    console.print(f"  [bold]9.[/] {back_label}    [bold]0.[/] Exit")


def _provider_header(console) -> None:
    """One-line header — provider name, status, model count, base URL."""
    provider = detect_provider(session.base_url) or "Custom"
    short = provider.split("(")[0].strip()
    host = session.base_url.replace("https://", "").replace("http://", "").split("/")[0]
    console.print()
    console.print(f"  [bold]{short}[/]  [dim]Connected · {len(session.models)} models · {host}[/]")


def _auto_filename(extension: str = "json") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"provider_scan_{ts}.{extension}"


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR


def _save_report(data: object, fmt: str = "json") -> str:
    _ensure_reports_dir()
    fname = _auto_filename(fmt)
    path = REPORTS_DIR / fname
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif fmt == "csv":
        _save_csv(list(data) if isinstance(data, list) else [], path)  # type: ignore[arg-type]
    return fname


def _save_csv(scans: list[CapabilityScan], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "chat", "streaming", "tools", "vision", "json_mode", "latency_ms"])
        for s in scans:
            writer.writerow([
                s.model_id,
                "Y" if s.chat.latency.ok and s.chat.content else "N",
                "Y" if s.streaming.first_token_ms is not None and not s.streaming.error else "N",
                "Y" if s.tools.supported else "N",
                "Y" if s.vision.supported else "N",
                "Y" if s.json_mode.supported else "N",
                f"{s.chat.latency.total_ms:.0f}" if s.chat.latency.ok else "",
            ])


def _list_saved_reports() -> list[Path]:
    if not REPORTS_DIR.is_dir():
        return []
    return sorted(REPORTS_DIR.iterdir(), reverse=True)


def _set_console(console) -> None:
    """Stash console ref for _ask feedback."""
    _ask._console = console  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Splash
# ---------------------------------------------------------------------------


def _splash(app: ProviderInspectorApp) -> str:
    _set_console(app.console)

    app.console.print()
    app.console.print()
    app.console.print(Align.center("[bold bright_white]CUSTOM PROVIDER INSPECTOR[/]"))
    app.console.print(Align.center("[dim]Inspect · Probe · Benchmark[/]"))
    app.console.print()
    app.console.rule(style="bright_blue")
    app.console.print()
    app.console.print(Align.center("[dim]crafted by icedeyes12[/]"))
    app.console.print(Align.center("[dim]co-developed with ฅReina ฅ^•ﻌ•^ฅ[/]"))
    app.console.print(
        Align.center("[link=https://github.com/icedeyes12/provider-inspector][dim]github.com/icedeyes12/provider-inspector[/][/]")
    )
    app.console.print()
    # Auto-advance — no Enter needed
    return "connect"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _connection(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Connect[/bold]")
    app.console.print()

    if has_env_file():
        app.console.print("  [dim]Enter a Base URL or env var ending _URL[/]")
        app.console.print("  [dim]e.g. OPENAI_URL  PROVIDER_URL  TEST_URL[/]")
        app.console.print()

    base_url_raw = Prompt.ask("  [accent]Base URL[/]")
    if base_url_raw.strip().lower() in ("0", "exit", "q"):
        return "exit"

    base_url, auto_key = resolve_url_key_pair(base_url_raw)

    if base_url != base_url_raw:
        app.console.print("  [success]✓ resolved URL[/]")

    if auto_key:
        api_key = auto_key
        suffix = api_key[-4:] if len(api_key) >= 4 else "****"
        app.console.print(f"  [success]✓ auto-resolved key[/]  [dim]{'*' * 8}{suffix}[/]")
    else:
        api_key_raw = Prompt.ask("  [accent]API Key[/]", password=True)
        if api_key_raw.strip().lower() in ("0", "exit", "q"):
            return "exit"
        from core.env import resolve_env
        api_key = resolve_env(api_key_raw)
        if api_key != api_key_raw:
            app.console.print("  [success]✓ resolved key[/]")

    if not base_url or not api_key:
        app.console.print(Panel("[error]Both Base URL and API key are required.[/]", border_style="red"))
        _nav(app.console, "Retry")
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "connect"

    app.console.print()
    with app.console.status("Connecting…", spinner=SPINNER_STYLE):
        ok = app.connect(base_url, api_key)

    if ok:
        return "main_menu"
    return "error"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


def _error(app: ProviderInspectorApp) -> str:
    app.console.print()
    app.console.print(Panel(
        app.last_error_friendly,
        title=app.last_error_title,
        border_style="red",
    ))
    _nav(app.console, "Retry")
    choice = _ask({NAV_BACK, NAV_EXIT})
    if choice == NAV_EXIT:
        return "exit"
    return "connect"


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def _main_menu(app: ProviderInspectorApp) -> str | None:
    _set_console(app.console)
    _provider_header(app.console)

    items: list[tuple[str, str]] = [
        ("Browse Models", "browse_models"),
        ("Full Capability Scan", "full_scan"),
    ]
    if app.scan_results:
        items.append(("Scan Results", "scan_results"))
    items.append(("Reports", "reports"))
    items.append(("Settings", "settings"))

    app.console.print()
    for i, (label, _) in enumerate(items, 1):
        app.console.print(f"  [bold]{i}.[/] {label}")

    _nav(app.console, "Disconnect")

    valid = set(range(1, len(items) + 1)) | {NAV_BACK, NAV_EXIT}
    choice = _ask(valid)

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "disconnect"
    if 1 <= choice <= len(items):
        return items[choice - 1][1]
    return None


# ---------------------------------------------------------------------------
# Browse models
# ---------------------------------------------------------------------------


def _browse_models(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Browse Models[/bold]")

    if not session.models:
        app.console.print("\n  [warning]No models loaded.[/]")
        _nav(app.console)
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "back"

    tbl = Table(show_header=True, header_style="table.header")
    tbl.add_column("#", style="dim white", justify="right", width=4)
    tbl.add_column("Model ID", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Owner", style="dim white", max_width=16, no_wrap=True, overflow="ellipsis")
    for i, m in enumerate(session.models, 1):
        tbl.add_row(str(i), m.id, m.owner)
    app.console.print(tbl)

    _nav(app.console)
    model_range = set(range(1, len(session.models) + 1))
    choice = _ask(model_range | {NAV_BACK, NAV_EXIT})

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "back"

    app._selected_model_idx = choice - 1
    return "model_detail"


# ---------------------------------------------------------------------------
# Model detail — accumulates results below menu
# ---------------------------------------------------------------------------


def _model_detail(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    idx = app._selected_model_idx
    model = session.models[idx]
    # Track test results for this model so we can accumulate
    if not hasattr(app, "_model_results"):
        app._model_results = {}  # type: ignore[attr-defined]
    results_cache = app._model_results.setdefault(idx, [])

    # Show header
    app.console.print()
    app.console.rule(f"[bold]{model.id}[/bold]")
    app.console.print(f"  [dim]{model.owner}[/]")

    # Show previous results (accumulated)
    for kind, data in results_cache:
        _display_single_result(app, kind, data)

    app.console.print()
    for i, (label, _) in enumerate(SINGLE_TESTS, 1):
        app.console.print(f"  [bold]{i}.[/] {label}")

    _nav(app.console)
    test_range = set(range(1, len(SINGLE_TESTS) + 1))
    choice = _ask(test_range | {NAV_BACK, NAV_EXIT})

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        # Clear cache for this model when leaving
        results_cache.clear()
        return "back"

    if not app.client:
        app.console.print("\n  [error]Not connected.[/]")
        return "model_detail"

    action = SINGLE_TESTS[choice - 1][1]
    with app.console.status(f"  Running {SINGLE_TESTS[choice - 1][0]}…", spinner=SPINNER_STYLE):
        result = _run_single_test(app.client, model.id, action)

    # Store result — will show on next render
    results_cache.append(result)
    return "model_detail"


def _run_single_test(client, model_id: str, action: str):
    match action:
        case "test_chat":
            return ("chat", test_chat(client, model_id))
        case "test_streaming":
            return ("streaming", test_streaming(client, model_id))
        case "test_tool_calling":
            return ("tools", test_tool_calling(client, model_id))
        case "test_vision":
            return ("vision", test_vision(client, model_id))
        case "test_json_mode":
            return ("json_mode", test_json_mode(client, model_id))
        case _:
            return ("unknown", None)


def _display_single_result(app: ProviderInspectorApp, kind: str, data) -> None:
    if data is None:
        app.console.print("\n  [error]Unknown test.[/]")
        return

    app.console.print()
    tbl = Table(show_header=False, box=None, padding=(0, 2), title=f"[bold]{kind}[/]")
    tbl.add_column("P", style="dim")
    tbl.add_column("V")

    lat = getattr(data, "latency", None)
    if lat:
        status = getattr(lat, "status_code", 0)
        if status and status >= 400:
            tbl.add_row("Status", f"[error]{status}[/]")
        tbl.add_row("Latency", f"{getattr(lat, 'total_ms', 0):.0f}ms")
        if getattr(lat, "first_token_ms", None) is not None:
            tbl.add_row("TTFT", f"{lat.first_token_ms:.0f}ms")
        if getattr(lat, "error", None):
            tbl.add_row("Error", f"[error]{lat.error}[/]")

    match kind:
        case "chat":
            tbl.add_row("Content", (data.content or "")[:200])
            tbl.add_row("Finish", data.finish_reason or "n/a")
            if data.usage:
                tbl.add_row("Tokens", f"in={data.usage.get('prompt_tokens', '?')} out={data.usage.get('completion_tokens', '?')}")
        case "streaming":
            tbl.add_row("Chunks", str(data.chunks))
            tbl.add_row("TTFT", f"{data.first_token_ms:.0f}ms" if data.first_token_ms else "n/a")
            tbl.add_row("Total", f"{data.total_ms:.0f}ms")
            if data.error:
                tbl.add_row("Error", f"[error]{data.error}[/]")
        case "tools":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            if data.tool_calls:
                tbl.add_row("Calls", str(len(data.tool_calls)))
        case "vision":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            if data.response_text:
                tbl.add_row("Response", data.response_text[:200])
        case "json_mode":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            tbl.add_row("Parsed", "[success]Yes[/]" if getattr(data, "parsed_json", False) else "[error]No[/]")
            if data.response_text:
                tbl.add_row("Response", data.response_text[:200])

    app.console.print(tbl)


# ---------------------------------------------------------------------------
# Full scan — Rich Progress with model counter + truncated IDs
# ---------------------------------------------------------------------------


def _full_scan(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Full Capability Scan[/bold]")

    if not session.models:
        app.console.print("\n  [warning]No models loaded.[/]")
        _nav(app.console)
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "back"

    if not app.client:
        app.console.print("\n  [error]Not connected.[/]")
        _nav(app.console)
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "back"

    total = len(session.models)
    total_tasks = total * 5
    results: list[CapabilityScan] = []

    app.console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=app.console,
    ) as progress:
        task = progress.add_task("Scanning", total=total_tasks)

        for i, m in enumerate(session.models, 1):
            scan = CapabilityScan(model_id=m.id)
            # Prioritize model name: "Model Name | Capability"
            tag = f"[bold]{m.id}[/] [dim]({i}/{total})[/]"

            progress.update(task, description=f"{tag} ❯ chat")
            scan.chat = test_chat(app.client, m.id)
            progress.advance(task)

            progress.update(task, description=f"{tag} ❯ stream")
            scan.streaming = test_streaming(app.client, m.id)
            progress.advance(task)

            progress.update(task, description=f"{tag} ❯ tools")
            scan.tools = test_tool_calling(app.client, m.id)
            progress.advance(task)

            progress.update(task, description=f"{tag} ❯ vision")
            scan.vision = test_vision(app.client, m.id)
            progress.advance(task)

            progress.update(task, description=f"{tag} ❯ json")
            scan.json_mode = test_json_mode(app.client, m.id)
            progress.advance(task)

            results.append(scan)

    app.scan_results = results

    # Results + actions immediately
    app.console.print()
    _print_scan_table(app, results)

    app.console.print()
    app.console.print("  [bold]1.[/] Export JSON")
    app.console.print("  [bold]2.[/] Export CSV")
    app.console.print("  [bold]3.[/] View Results")
    _nav(app.console)
    choice = _ask({1, 2, 3, NAV_BACK, NAV_EXIT})

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "back"
    if choice == 1:
        fname = _save_report([asdict(s) for s in results], "json")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_results"
    if choice == 2:
        fname = _save_report(results, "csv")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_results"
    if choice == 3:
        return "scan_results"
    return "scan_results"


def _print_scan_table(app: ProviderInspectorApp, results: list[CapabilityScan], title: str = "Results") -> None:
    tbl = Table(show_header=True, header_style="table.header", title=title)
    tbl.add_column("Model", style="bold magenta", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Chat", justify="center")
    tbl.add_column("Stream", justify="center")
    tbl.add_column("Tools", justify="center")
    tbl.add_column("Vision", justify="center")
    tbl.add_column("JSON Mode", justify="center")
    tbl.add_column("Latency", justify="right", style="dim white")

    for s in results:
        chat_ok = "[success]Y[/]" if s.chat.latency.ok and s.chat.content else "[error]N[/]"
        stream_ok = "[success]Y[/]" if s.streaming.first_token_ms is not None and not s.streaming.error else "[error]N[/]"
        tools_ok = "[success]Y[/]" if s.tools.supported else "[error]N[/]"
        vision_ok = "[success]Y[/]" if s.vision.supported else "[error]N[/]"
        json_ok = "[success]Y[/]" if s.json_mode.supported else "[error]N[/]"
        lat = f"{s.chat.latency.total_ms:.0f}ms" if s.chat.latency.ok else "—"
        tbl.add_row(s.model_id, chat_ok, stream_ok, tools_ok, vision_ok, json_ok, lat)

    app.console.print(tbl)


# ---------------------------------------------------------------------------
# Scan Results (in-memory)
# ---------------------------------------------------------------------------


def _scan_results(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    results = app.scan_results

    if not results:
        app.console.print("\n  [warning]No scan results in memory.[/]")
        app.console.print("  [dim]Run a Full Capability Scan first.[/]")
        _nav(app.console)
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "back"

    app.console.print()
    app.console.rule("[bold]Scan Results[/bold]")
    app.console.print()
    app.console.print("  [bold]1.[/] Browse")
    app.console.print("  [bold]2.[/] Sort by Latency")
    app.console.print("  [bold]3.[/] Sort by Capabilities")
    app.console.print("  [bold]4.[/] Export JSON")
    app.console.print("  [bold]5.[/] Export CSV")
    _nav(app.console)
    choice = _ask({1, 2, 3, 4, 5, NAV_BACK, NAV_EXIT})

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "back"

    if choice == 1:
        _browse_scan_results(app, results)
        return "scan_results"
    if choice == 2:
        sorted_r = sorted(results, key=lambda s: s.chat.latency.total_ms if s.chat.latency.ok else 99999)
        _print_scan_table(app, sorted_r, "Sorted by Latency")
        _nav(app.console)
        nav = _ask({NAV_BACK, NAV_EXIT})
        return "exit" if nav == NAV_EXIT else "scan_results"
    if choice == 3:
        sorted_r = sorted(results, key=lambda s: sum([
            s.chat.latency.ok and bool(s.chat.content),
            s.streaming.first_token_ms is not None and not s.streaming.error,
            s.tools.supported,
            s.vision.supported,
            s.json_mode.supported,
        ]), reverse=True)
        _print_scan_table(app, sorted_r, "Sorted by Capabilities")
        _nav(app.console)
        nav = _ask({NAV_BACK, NAV_EXIT})
        return "exit" if nav == NAV_EXIT else "scan_results"
    if choice == 4:
        fname = _save_report([asdict(s) for s in results], "json")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_results"
    if choice == 5:
        fname = _save_report(results, "csv")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_results"

    return "scan_results"


def _browse_scan_results(app: ProviderInspectorApp, results: list[CapabilityScan]) -> None:
    tbl = Table(show_header=True, header_style="table.header")
    tbl.add_column("#", style="dim white", justify="right", width=4)
    tbl.add_column("Model ID", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Chat", justify="center")
    tbl.add_column("Latency", justify="right", style="dim white")
    for i, s in enumerate(results, 1):
        chat_ok = "[success]Y[/]" if s.chat.latency.ok and s.chat.content else "[error]N[/]"
        lat = f"{s.chat.latency.total_ms:.0f}ms" if s.chat.latency.ok else "—"
        tbl.add_row(str(i), s.model_id, chat_ok, lat)
    app.console.print(tbl)

    _nav(app.console)
    model_range = set(range(1, len(results) + 1))
    choice = _ask(model_range | {NAV_BACK, NAV_EXIT})
    if choice in {NAV_BACK, NAV_EXIT}:
        return

    s = results[choice - 1]
    app.console.print()
    app.console.rule(f"[bold]{s.model_id}[/bold]")

    dtbl = Table(show_header=False, box=None, padding=(0, 2))
    dtbl.add_column("Cap", style="dim")
    dtbl.add_column("Status")
    dtbl.add_column("Detail", style="dim")

    dtbl.add_row("Chat", "[success]Y[/]" if s.chat.latency.ok and s.chat.content else "[error]N[/]",
                 f"{s.chat.latency.total_ms:.0f}ms" if s.chat.latency.ok else "failed")
    dtbl.add_row("Streaming", "[success]Y[/]" if s.streaming.first_token_ms is not None and not s.streaming.error else "[error]N[/]",
                 f"ttft={s.streaming.first_token_ms:.0f}ms" if s.streaming.first_token_ms else "n/a")
    dtbl.add_row("Tools", "[success]Y[/]" if s.tools.supported else "[error]N[/]",
                 f"{len(s.tools.tool_calls)} calls" if s.tools.tool_calls else "")
    dtbl.add_row("Vision", "[success]Y[/]" if s.vision.supported else "[error]N[/]", "")
    dtbl.add_row("JSON Mode", "[success]Y[/]" if s.json_mode.supported else "[error]N[/]",
                 "parsed" if getattr(s.json_mode, "parsed_json", False) else "unparsed")
    app.console.print(dtbl)

    _nav(app.console)
    _ask({NAV_BACK, NAV_EXIT})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _reports(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Reports[/bold]")

    has_results = bool(app.scan_results)
    existing = _list_saved_reports()

    app.console.print()
    item_num = 1
    items: list[str] = []

    if existing:
        app.console.print(f"  [bold]{item_num}.[/] View Reports")
        items.append("view_existing")
        item_num += 1

    if has_results:
        app.console.print(f"  [bold]{item_num}.[/] Export Current Scan")
        items.append("export_current")
        item_num += 1

    if existing:
        app.console.print(f"  [bold]{item_num}.[/] Delete Report")
        items.append("delete_report")
        item_num += 1

    if not items:
        app.console.print("  [warning]No reports found.[/]")
        app.console.print("  [dim]Run a scan first, then export.[/]")
        _nav(app.console)
        choice = _ask({NAV_BACK, NAV_EXIT})
        if choice == NAV_EXIT:
            return "exit"
        return "back"

    _nav(app.console)
    valid = set(range(1, item_num)) | {NAV_BACK, NAV_EXIT}
    choice = _ask(valid)

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "back"

    action = items[choice - 1]

    if action == "view_existing":
        _view_existing_reports(app)
        return "reports"

    if action == "export_current":
        app.console.print("\n  [bold]1.[/] JSON    [bold]2.[/] CSV")
        fmt_choice = _ask({1, 2})
        fmt = "json" if fmt_choice == 1 else "csv"
        fname = _save_report([asdict(s) for s in app.scan_results] if fmt == "json" else app.scan_results, fmt)
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "reports"

    if action == "delete_report":
        _delete_report(app)
        return "reports"

    return "reports"


def _view_existing_reports(app: ProviderInspectorApp) -> None:
    existing = _list_saved_reports()
    if not existing:
        app.console.print("\n  [warning]No saved reports.[/]")
        return

    app.console.print()
    tbl = Table(show_header=True, header_style="table.header")
    tbl.add_column("#", style="dim white", justify="right", width=4)
    tbl.add_column("Filename")
    tbl.add_column("Size", justify="right", style="dim white")
    for i, p in enumerate(existing, 1):
        size = p.stat().st_size
        size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
        tbl.add_row(str(i), p.name, size_str)
    app.console.print(tbl)

    _nav(app.console)
    report_range = set(range(1, len(existing) + 1))
    choice = _ask(report_range | {NAV_BACK, NAV_EXIT})
    if choice in {NAV_BACK, NAV_EXIT}:
        return

    chosen = existing[choice - 1]
    try:
        content = chosen.read_text()
        lines = content.splitlines()[:60]
        app.console.print()
        app.console.rule(f"[bold]{chosen.name}[/bold]")
        for line in lines:
            app.console.print(f"  [dim]{line}[/]")
        total_lines = len(content.splitlines())
        if total_lines > 60:
            app.console.print(f"\n  [dim]… {total_lines - 60} more lines[/]")
    except Exception as exc:  # noqa: BLE001
        app.console.print(f"\n  [error]Cannot read: {exc}[/]")

    _nav(app.console)
    _ask({NAV_BACK, NAV_EXIT})


def _delete_report(app: ProviderInspectorApp) -> None:
    existing = _list_saved_reports()
    if not existing:
        app.console.print("\n  [warning]Nothing to delete.[/]")
        return

    app.console.print()
    for i, p in enumerate(existing, 1):
        app.console.print(f"  [bold]{i}.[/] {p.name}")

    _nav(app.console)
    report_range = set(range(1, len(existing) + 1))
    choice = _ask(report_range | {NAV_BACK, NAV_EXIT})
    if choice in {NAV_BACK, NAV_EXIT}:
        return

    chosen = existing[choice - 1]
    # Confirmation
    app.console.print(f"\n  [warning]Delete {chosen.name}?[/]")
    app.console.print("  [bold]1.[/] Yes    [bold]2.[/] No")
    confirm = _ask({1, 2})
    if confirm == 2:
        app.console.print("  [dim]Cancelled.[/]")
        return

    try:
        chosen.unlink()
        app.console.print(f"\n  [success]Deleted[/] {chosen.name}")
    except OSError as exc:
        app.console.print(f"\n  [error]Failed: {exc}[/]")


# ---------------------------------------------------------------------------
# Settings — no About (credits on splash)
# ---------------------------------------------------------------------------


def _settings(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Settings[/bold]")
    app.console.print()
    app.console.print("  [bold]1.[/] Reconnect")
    app.console.print("  [bold]2.[/] Disconnect")
    _nav(app.console)

    choice = _ask({1, 2, NAV_BACK, NAV_EXIT})

    if choice == NAV_EXIT:
        return "exit"
    if choice == NAV_BACK:
        return "back"
    if choice == 1:
        if app.scan_results:
            app.console.print("\n  [warning]Reconnecting will discard scan results.[/]")
            app.console.print("  [bold]1.[/] Continue    [bold]2.[/] Cancel")
            c = _ask({1, 2})
            if c == 2:
                return "settings"
        return "connect"
    if choice == 2:
        return "disconnect"

    return "back"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SCREEN_MAP: dict[Screen, Callable] = {
    Screen.SPLASH: _splash,
    Screen.CONNECTION: _connection,
    Screen.ERROR_PANEL: _error,
    Screen.MAIN_MENU: _main_menu,
    Screen.BROWSE_MODELS: _browse_models,
    Screen.MODEL_DETAIL: _model_detail,
    Screen.FULL_SCAN: _full_scan,
    Screen.SCAN_RESULTS: _scan_results,
    Screen.REPORTS: _reports,
    Screen.SETTINGS: _settings,
}

_ACTION_TO_SCREEN: dict[str, Screen] = {
    "connect": Screen.CONNECTION,
    "error": Screen.ERROR_PANEL,
    "main_menu": Screen.MAIN_MENU,
    "browse_models": Screen.BROWSE_MODELS,
    "model_detail": Screen.MODEL_DETAIL,
    "full_scan": Screen.FULL_SCAN,
    "scan_results": Screen.SCAN_RESULTS,
    "reports": Screen.REPORTS,
    "settings": Screen.SETTINGS,
    "disconnect": Screen.CONNECTION,
}


def render(app: ProviderInspectorApp) -> str | None:
    """Render current screen, return action string or None to quit."""
    # Clear screen between renders for clean UX
    app.console.clear()

    handler = _SCREEN_MAP.get(app.screen)
    if handler is None:
        return "exit"

    action = handler(app)
    if action is None or action == "exit":
        return None
    if action == "disconnect":
        app.disconnect()
        app.go(Screen.CONNECTION)
        return action
    if action == "back":
        app.back()
        return action

    target = _ACTION_TO_SCREEN.get(action)
    if target is not None:
        app.go(target)
    return action
