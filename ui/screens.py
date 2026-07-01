"""Screen renderers for Provider Inspector.

Navigation:
  b = Back
  q = Quit
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich import box
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from core.benchmark import (
    test_chat,
    test_json_mode,
    test_streaming,
    test_tool_calling,
    test_vision,
)
from core.env import has_env_file, resolve_url_key_pair
from core.models import CapabilityScan, ChatResult, JsonModeResult, ModelInfo, StreamResult, ToolCallResult, VisionResult
from core.scanner import detect_provider
from core.session import session
from ui.app import Screen
from ui.theme import SPINNER_STYLE

if TYPE_CHECKING:
    from ui.app import ProviderInspectorApp

REPORTS_DIR = Path("reports")
MODEL_ID_MAX = 24
RECENT_ACTIVITY_LIMIT = 8
CAPABILITY_SEQUENCE = [
    ("Chat", "chat"),
    ("Stream", "streaming"),
    ("Tools", "tools"),
    ("Vision", "vision"),
    ("JSON", "json_mode"),
]
SINGLE_TESTS = [
    ("Chat", "test_chat"),
    ("Stream", "test_streaming"),
    ("Tools", "test_tool_calling"),
    ("Vision", "test_vision"),
    ("JSON Mode", "test_json_mode"),
]

CAPABILITY_LABELS = [
    ("Chat", "chat"),
    ("Stream", "streaming"),
    ("Tools", "tools"),
    ("Vision", "vision"),
    ("JSON", "json_mode"),
]


@dataclass
class _RecentModelResult:
    model_id: str
    success: bool
    reason: str = ""


@dataclass
class _ScanDashboardState:
    total_models: int
    current_model_index: int = 0
    current_model_id: str = ""
    current_model_owner: str = ""
    current_capability_index: int = 0
    current_capability_name: str = ""
    completed_models: int = 0
    completed_capabilities: int = 0
    current_scan: CapabilityScan | None = None
    recent: list[_RecentModelResult] = field(default_factory=list)
    results: list[CapabilityScan] = field(default_factory=list)
    running: bool = True
    started_at: float = field(default_factory=time.monotonic)
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 1:
        return text[:max_len]
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    seconds = int(round(ms / 1000.0))
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _format_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms:.0f}ms"


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}" if minutes else f"{secs:02d}"


def _spinner_frame() -> str:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    return frames[int(time.monotonic() * 6) % len(frames)]


def _set_console(console) -> None:
    _ask._console = console  # type: ignore[attr-defined]


def _ask(valid: set[int] | None = None) -> int | str:
    while True:
        raw = Prompt.ask("  >").strip().lower()
        if not raw:
            continue
        if raw in {"b", "q"}:
            return raw
        try:
            value = int(raw)
        except ValueError:
            console = getattr(_ask, "_console", None)
            if console:
                console.print("  [dim]?[/]")
            continue
        if valid is not None and value not in valid:
            console = getattr(_ask, "_console", None)
            if console:
                console.print("  [dim]?[/]")
            continue
        return value


def _nav(console, back_label: str = "Back") -> None:
    console.print()
    console.rule(style="dim")
    console.print(f"  [bold]b[/] {back_label}    [bold]q[/] Quit")


def _provider_header(console) -> None:
    provider = detect_provider(session.base_url) or "Custom"
    short = provider.split("(")[0].strip()
    host = session.base_url.replace("https://", "").replace("http://", "").split("/")[0]
    console.print()
    header = Text()
    header.append(short, style="bold")
    header.append("  ")
    header.append("Connected", style="muted")
    if session.models:
        header.append(f" · {len(session.models)} models", style="muted")
    if host:
        header.append(f" · {host}", style="muted")
    console.print(header)


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)


def _auto_filename(extension: str = "json") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"provider_scan_{ts}.{extension}"


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
        for scan in scans:
            writer.writerow(
                [
                    scan.model_id,
                    "Y" if scan.chat.latency.ok and scan.chat.content else "N",
                    "Y" if scan.streaming.first_token_ms is not None and not scan.streaming.error else "N",
                    "Y" if scan.tools.supported else "N",
                    "Y" if scan.vision.supported else "N",
                    "Y" if scan.json_mode.supported else "N",
                    f"{scan.chat.latency.total_ms:.0f}" if scan.chat.latency.ok else "",
                ]
            )


def _list_saved_reports() -> list[Path]:
    if not REPORTS_DIR.is_dir():
        return []
    return sorted(REPORTS_DIR.iterdir(), reverse=True)


def _model_scan_for(model_id: str) -> CapabilityScan | None:
    for scan in session.scan_results:
        if scan.model_id == model_id:
            return scan
    return None


def _capability_ok(value) -> bool:
    if isinstance(value, ChatResult):
        return value.latency.ok and bool(value.content)
    if isinstance(value, StreamResult):
        return value.first_token_ms is not None and not value.error
    if isinstance(value, ToolCallResult):
        return value.supported
    if isinstance(value, VisionResult):
        return value.supported
    if isinstance(value, JsonModeResult):
        return value.supported
    return False


def _result_reason(value) -> str:
    if isinstance(value, ChatResult):
        if value.latency.error:
            return value.latency.error
        if value.latency.status_code >= 400:
            return f"HTTP {value.latency.status_code}"
        if not value.content:
            return "Empty response"
        return ""
    if isinstance(value, StreamResult):
        if value.error:
            return value.error
        if value.status_code >= 400:
            return f"HTTP {value.status_code}"
        return "No stream output"
    if isinstance(value, ToolCallResult):
        if value.error:
            return value.error
        return "Tool calls not returned"
    if isinstance(value, VisionResult):
        if value.error:
            return value.error
        return "Image handling rejected"
    if isinstance(value, JsonModeResult):
        if value.error:
            return value.error
        if not value.parsed_json:
            return "Returned non-JSON text"
        return ""
    return ""


def _primary_latency(scan: CapabilityScan) -> float | None:
    candidates = [
        scan.chat.latency.total_ms if scan.chat.latency.ok else None,
        scan.streaming.total_ms if scan.streaming.first_token_ms is not None and not scan.streaming.error else None,
        scan.tools.latency.total_ms if scan.tools.latency.ok else None,
        scan.vision.latency.total_ms if scan.vision.latency.ok else None,
        scan.json_mode.latency.total_ms if scan.json_mode.latency.ok else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _scan_status(scan: CapabilityScan) -> tuple[str, str]:
    passes = [
        _capability_ok(scan.chat),
        _capability_ok(scan.streaming),
        _capability_ok(scan.tools),
        _capability_ok(scan.vision),
        _capability_ok(scan.json_mode),
    ]
    count = sum(1 for item in passes if item)
    if count == 5:
        return ("OK", "status.ok")
    if count == 0:
        return ("FAILED", "status.fail")
    return ("PARTIAL", "status.partial")


def _capability_summary(scan: CapabilityScan) -> str:
    enabled: list[str] = []
    if _capability_ok(scan.chat):
        enabled.append("Chat")
    if _capability_ok(scan.streaming):
        enabled.append("Stream")
    if _capability_ok(scan.tools):
        enabled.append("Tools")
    if _capability_ok(scan.vision):
        enabled.append("Vision")
    if _capability_ok(scan.json_mode):
        enabled.append("JSON")
    return " ".join(enabled)


def _finding_summary(scan: CapabilityScan) -> list[tuple[str, bool]]:
    return [
        ("Chat", scan.chat.observed),
        ("Stream", scan.streaming.observed),
        ("Tools", scan.tools.observed),
        ("Vision", scan.vision.observed),
        ("JSON", scan.json_mode.observed),
    ]


def _finding_reason(scan: CapabilityScan) -> list[tuple[str, str]]:
    reasons: list[tuple[str, str]] = []
    for label, capability_key in CAPABILITY_LABELS:
        result = getattr(scan, capability_key)
        if not result.observed:
            reason = _result_reason(result)
            if reason:
                reasons.append((label, reason))
    return reasons


def _failure_reason(scan: CapabilityScan) -> str:
    reasons = _finding_reason(scan)
    if reasons:
        label, reason = reasons[0]
        return f"{label}: {_truncate(reason, 64)}"
    return ""


def _status_style(status: str) -> str:
    if status == "OK":
        return "status.ok"
    if status == "FAILED":
        return "status.fail"
    return "status.partial"


def _scan_progress_layout(state: _ScanDashboardState, width: int) -> Layout:
    with state.lock:
        total_models = state.total_models
        current_model_index = state.current_model_index
        current_model_id = state.current_model_id
        current_model_owner = state.current_model_owner
        current_capability_index = state.current_capability_index
        current_capability_name = state.current_capability_name
        completed_models = state.completed_models
        completed_capabilities = state.completed_capabilities
        recent = list(state.recent)
        current_scan = state.current_scan
        started_at = state.started_at
        running = state.running
        error = state.error

    elapsed = _format_elapsed(time.monotonic() - started_at)
    provider = detect_provider(session.base_url) or "Custom"
    host = session.base_url.replace("https://", "").replace("http://", "").split("/")[0]
    spinner = _spinner_frame() if running else "✓"

    layout = Layout(name="root")
    layout.split_column(
        Layout(_scan_title_panel(provider, host, total_models, completed_models, elapsed, spinner), size=5),
        Layout(
            _scan_current_model_panel(
                current_model_id,
                current_model_owner,
                current_model_index,
                total_models,
                current_capability_name,
                running,
                error,
            ),
            size=9,
        ),
        Layout(
            _scan_current_capability_panel(
                current_scan,
                current_capability_name,
                current_capability_index,
                running,
            ),
            size=10,
        ),
        Layout(
            _scan_progress_panel(
                completed_models,
                total_models,
                completed_capabilities,
                current_capability_index,
                elapsed,
            ),
            size=8,
        ),
        Layout(_scan_recent_panel(recent), ratio=1),
    )
    return layout


def _scan_title_panel(provider: str, host: str, total_models: int, completed_models: int, elapsed: str, spinner: str) -> Panel:
    lines = Table.grid(expand=True)
    lines.add_column()
    lines.add_row(Text("Full Capability Scan", style="bold bright_white"))
    summary = Text()
    summary.append(f"{spinner} ", style="status.active")
    summary.append(f"{provider}", style="bold")
    if host:
        summary.append(f" · {host}", style="muted")
    summary.append(f" · {completed_models} / {total_models} models", style="muted")
    summary.append(f" · {elapsed} elapsed", style="muted")
    lines.add_row(summary)
    return Panel(lines, border_style="panel.border")


def _scan_current_model_panel(
    model_id: str,
    owner: str,
    index: int,
    total: int,
    capability: str,
    running: bool,
    error: str,
) -> Panel:
    body = Table.grid(expand=True)
    body.add_column(ratio=1)
    if model_id:
        body.add_row(Text(model_id, style="bold bright_white", overflow="fold"))
        context = Text()
        if owner:
            context.append(f"Owner      : {owner}", style="muted")
            if capability:
                context.append("\n")
        if capability:
            context.append(f"Capability : {capability}", style="muted")
        if index and total:
            context.append("\n")
            context.append(f"Model      : {index} / {total}", style="muted")
        if running:
            context.append("\n")
            context.append("Status     : Testing...", style="status.active")
        body.add_row(context)
    else:
        body.add_row(Text("Waiting for the first model…", style="muted"))
    if error:
        body.add_row(Text(_truncate(error, 120), style="status.fail"))
    return Panel(body, title="Current Model", border_style="panel.border")


def _capability_marker(active: bool, result: CapabilityScan | None, label: str, field: str) -> tuple[str, str]:
    if active:
        return ("⟳", "status.active")
    if result is None:
        return ("•", "status.pending")
    scan_result = getattr(result, field)
    if _capability_ok(scan_result):
        return ("✓", "status.ok")
    return ("✗", "status.fail")


def _scan_current_capability_panel(
    scan: CapabilityScan | None,
    capability: str,
    capability_index: int,
    running: bool,
) -> Panel:
    rows = Table.grid(expand=True)
    rows.add_column(ratio=1)
    rows.add_column(width=4, justify="right")

    current_label = capability or "Waiting"
    if running and current_label:
        header = Text()
        header.append("Status     : ", style="muted")
        header.append(f"Testing {current_label}...", style="status.active")
    else:
        header = Text("Status     : Idle", style="muted")
    rows.add_row(header, Text(_spinner_frame() if running else "", style="status.active"))

    for label, capability_key in CAPABILITY_SEQUENCE:
        active = running and label == capability
        marker, marker_style = _capability_marker(active, scan, label, capability_key)
        label_text = Text(label, style="status.active" if active else "muted")
        rows.add_row(label_text, Text(marker, style=marker_style))

    return Panel(rows, title="Current Capability", border_style="panel.border")


def _progress_bar(label: str, completed: int, total: int, suffix: str, style: str) -> Progress:
    progress = Progress(
        TextColumn(f"[muted]{label}[/]"),
        BarColumn(bar_width=None, complete_style=style, finished_style=style),
        TextColumn(f"[bold]{completed} / {total} {suffix}[/]"),
        expand=True,
    )
    task_id = progress.add_task(label, total=max(total, 1))
    progress.update(task_id, completed=min(completed, total))
    return progress


def _scan_progress_panel(
    completed_models: int,
    total_models: int,
    completed_capabilities: int,
    capability_index: int,
    elapsed: str,
) -> Panel:
    capability_total = len(CAPABILITY_SEQUENCE)
    current_capability = CAPABILITY_SEQUENCE[max(0, min(capability_index - 1, capability_total - 1))][0] if capability_index else "Starting"
    body = Group(
        _progress_bar("Model", completed_models, total_models, "models", "status.ok"),
        _progress_bar("Capability", max(0, capability_index), capability_total, current_capability, "status.active"),
        Text(f"Elapsed     : {elapsed}", style="muted"),
    )
    return Panel(body, title="Progress", border_style="panel.border")


def _scan_recent_panel(recent: list[_RecentModelResult]) -> Panel:
    if not recent:
        return Panel(Text("Waiting for completed models…", style="muted"), title="Recent Activity", border_style="panel.border")

    table = Table(show_header=False, box=box.SIMPLE, expand=True, pad_edge=False)
    table.add_column("Mark", width=3)
    table.add_column("Model", overflow="fold")
    table.add_column("Reason", overflow="fold")

    for item in recent[-RECENT_ACTIVITY_LIMIT:][::-1]:
        if item.success:
            table.add_row(Text("✓", style="status.ok"), Text(item.model_id, style="bold"), Text("", style="muted"))
        else:
            table.add_row(Text("✗", style="status.fail"), Text(item.model_id, style="bold"), Text(_truncate(item.reason, 48), style="status.fail"))
    return Panel(table, title="Recent Activity", border_style="panel.border")


def _record_model_completion(state: _ScanDashboardState, scan: CapabilityScan) -> None:
    status, _ = _scan_status(scan)
    success = status == "OK"
    reason = "" if success else _failure_reason(scan)
    with state.lock:
        state.results.append(scan)
        state.completed_models = len(state.results)
        state.current_scan = scan
        state.recent.append(_RecentModelResult(scan.model_id, success, reason))
        state.recent = state.recent[-RECENT_ACTIVITY_LIMIT:]


def _scan_model_worker(app: ProviderInspectorApp, state: _ScanDashboardState) -> None:
    if not app.client:
        with state.lock:
            state.error = "Not connected."
            state.running = False
        return

    try:
        for model_index, model in enumerate(session.models, 1):
            scan = CapabilityScan(model_id=model.id)
            with state.lock:
                state.current_model_index = model_index
                state.current_model_id = model.id
                state.current_model_owner = model.owner
                state.current_capability_index = 0
                state.current_capability_name = ""
                state.current_scan = scan

            for capability_index, (label, action) in enumerate(CAPABILITY_SEQUENCE, 1):
                with state.lock:
                    state.current_capability_index = capability_index
                    state.current_capability_name = label

                if action == "chat":
                    scan.chat = test_chat(app.client, model.id)
                elif action == "streaming":
                    scan.streaming = test_streaming(app.client, model.id)
                elif action == "tools":
                    scan.tools = test_tool_calling(app.client, model.id)
                elif action == "vision":
                    scan.vision = test_vision(app.client, model.id)
                elif action == "json_mode":
                    scan.json_mode = test_json_mode(app.client, model.id)

                with state.lock:
                    state.completed_capabilities += 1
                    state.current_scan = scan

            _record_model_completion(state, scan)

    except Exception as exc:  # noqa: BLE001
        with state.lock:
            state.error = str(exc)
    finally:
        with state.lock:
            state.running = False
            if not state.current_model_id and session.models:
                state.current_model_index = 1
                state.current_model_id = session.models[0].id
                state.current_model_owner = session.models[0].owner


def _render_scan_dashboard(state: _ScanDashboardState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="title", size=5),
        Layout(name="summary", size=8),
        Layout(name="progress", size=9),
        Layout(name="recent", ratio=1),
    )

    current_model = state.current_model_id or "Starting…"
    owner = state.current_model_owner
    capability = state.current_capability_name or "Preparing"
    total_models = state.total_models
    model_progress = f"Model {state.completed_models} / {total_models}" if total_models else "Model 0 / 0"
    capability_progress = f"Capability: {capability} ({state.completed_capabilities % 5 + 1} / 5)" if state.current_capability_name else "Capability: Preparing"
    elapsed = _format_elapsed(time.monotonic() - state.started_at)
    spinner = _spinner_frame() if state.running else ""

    title = Table.grid(padding=(0, 1))
    title.add_column()
    title.add_row(Text("Full Capability Scan", style="bold white"))
    title.add_row(Text("Live dashboard", style="muted"))
    layout["title"].update(Panel(title, border_style="panel.border"))

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="muted", no_wrap=True)
    summary.add_column()
    summary.add_row("Current Model", Text(current_model, style="bold white"))
    if owner:
        summary.add_row("Owner", Text(owner, style="muted"))
    summary.add_row("Current Capability", Text(capability, style="bold"))
    summary.add_row("Status", Text(f"{spinner} Testing…" if state.running else "Done", style="status.active" if state.running else "status.ok"))
    layout["summary"].update(Panel(summary, border_style="panel.border"))

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.percentage:>3.0f}%"),
        expand=True,
    )
    task = progress.add_task("", total=100)
    percent = 0
    if total_models:
        percent = int((state.completed_models / total_models) * 100)
    progress.update(task, completed=percent, description=model_progress)
    progress_group = Table.grid(expand=True)
    progress_group.add_column()
    progress_group.add_row(progress)
    progress_group.add_row(Text(capability_progress, style="muted"))
    progress_group.add_row(Text(f"Elapsed: {elapsed}", style="muted"))
    layout["progress"].update(Panel(progress_group, border_style="panel.border"))

    recent_tbl = Table(show_header=False, box=box.SIMPLE, expand=True, padding=(0, 1))
    recent_tbl.add_column("Status", width=3)
    recent_tbl.add_column("Model", overflow="ellipsis")
    recent_tbl.add_column("Detail", overflow="ellipsis")
    for entry in state.recent[-RECENT_ACTIVITY_LIMIT:]:
        icon = "✓" if entry.success else "✗"
        detail = entry.reason
        recent_tbl.add_row(icon, entry.model_id, detail)
    if not state.recent:
        recent_tbl.add_row("•", "", "Waiting for first result")
    layout["recent"].update(Panel(recent_tbl, title="Recent Activity", border_style="panel.border"))
    return layout


def _render_scan_progress(model: str, capability: str, done: int, total: int, elapsed_ms: float, console_width: int) -> Panel:
    inner_width = max(30, console_width - 6)
    model_width = max(18, inner_width - 10)
    cap_width = max(10, inner_width - 12)
    model_value = _truncate(model, model_width)
    cap_value = _truncate(capability, cap_width)
    progress_value = f"{done}/{total} ({(done / total * 100):.0f}%)" if total else "0/0"
    elapsed_value = _format_duration(elapsed_ms)

    body = Text()
    body.append("Model: ", style="muted")
    body.append(model_value, style="bold")
    body.append("\n")
    body.append("Capability: ", style="muted")
    body.append(cap_value)
    body.append("\n")
    body.append("Progress: ", style="muted")
    body.append(progress_value)
    body.append("\n")
    body.append("Elapsed: ", style="muted")
    body.append(elapsed_value)

    return Panel.fit(body, title="Scanning", border_style="panel.border")


def _render_scan_table(results: list[CapabilityScan], title: str = "Results") -> Table:
    tbl = Table(show_header=True, header_style="table.header", title=title, box=box.SIMPLE, expand=True)
    tbl.add_column("#", style="dim", justify="right", width=4)
    tbl.add_column("Model", style="bold magenta", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Observed", justify="center", width=16)
    tbl.add_column("Latency", justify="right", width=10)
    tbl.add_column("Findings", max_width=24, overflow="ellipsis")

    for idx, scan in enumerate(results, 1):
        findings = _finding_summary(scan)
        observed = len([item for item in findings if item[1]])
        latency = _format_ms(_primary_latency(scan))
        finding_text = " ".join([f"✓ {name}" if ok else f"✗ {name}" for name, ok in findings])
        tbl.add_row(str(idx), scan.model_id, f"{observed} / {len(findings)}", latency, finding_text)
    return tbl


def _render_model_detail(app: ProviderInspectorApp, model: ModelInfo, scan: CapabilityScan | None) -> list[object]:
    blocks: list[object] = []

    overview = Table.grid(padding=(0, 1))
    overview.add_column(style="muted", no_wrap=True)
    overview.add_column()
    overview.add_row("Model", f"[bold]{model.id}[/]")
    if scan is not None:
        latency = _format_ms(_primary_latency(scan))
        if latency:
            overview.add_row("Latency", latency)
    if model.owner:
        overview.add_row("Owner", model.owner)
    if model.created:
        overview.add_row("Created", datetime.fromtimestamp(model.created).strftime("%Y-%m-%d %H:%M"))
    blocks.append(Panel(overview, title="Model", border_style="panel.border"))

    if scan is not None:
        findings = Table(show_header=False, box=box.SIMPLE, expand=True, pad_edge=False)
        findings.add_column("Mark", width=3)
        findings.add_column("Capability", style="bold", width=12)
        findings.add_column("Status", width=12)
        findings.add_column("Detail", overflow="ellipsis")

        rows = [
            ("Chat", scan.chat, "content" if scan.chat.observed else _result_reason(scan.chat)),
            ("Stream", scan.streaming, "first token" if scan.streaming.observed else _result_reason(scan.streaming)),
            ("Tools", scan.tools, "tool calls" if scan.tools.observed else _result_reason(scan.tools)),
            ("Vision", scan.vision, "image input" if scan.vision.observed else _result_reason(scan.vision)),
            ("JSON", scan.json_mode, "json output" if scan.json_mode.observed else _result_reason(scan.json_mode)),
        ]
        for label, result, detail in rows:
            ok = result.observed
            findings.add_row("✓" if ok else "✗", label, "observed" if ok else "unsupported", detail)
        blocks.append(Panel(findings, title="Inspection Findings", border_style="panel.border"))
    else:
        blocks.append(Panel("Run a full scan to inspect this model.", border_style="panel.border"))

    meta = Table.grid(padding=(0, 1))
    meta.add_column(style="muted", no_wrap=True)
    meta.add_column()
    provider = detect_provider(session.base_url)
    if provider:
        meta.add_row("Provider", provider)
    if session.base_url:
        meta.add_row("Base URL", session.base_url)
    raw_keys = sorted(model.raw.keys())
    if raw_keys:
        meta.add_row("Raw fields", str(len(raw_keys)))
    blocks.append(Panel(meta, title="Provider Metadata", border_style="panel.border"))

    return blocks


def _model_result_menu(app: ProviderInspectorApp) -> None:
    app.console.print()
    for idx, (label, _) in enumerate(SINGLE_TESTS, 1):
        app.console.print(f"  [bold]{idx}.[/] {label}")


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
        app.console.print(Panel("Unknown test.", border_style="red"))
        return

    tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), expand=True)
    tbl.add_column("Field", style="muted", no_wrap=True, width=12)
    tbl.add_column("Value")

    lat = getattr(data, "latency", None)
    if lat:
        if getattr(lat, "status_code", 0) >= 400:
            tbl.add_row("Status", f"[error]{lat.status_code}[/]")
        if getattr(lat, "total_ms", None) is not None:
            tbl.add_row("Latency", _format_ms(lat.total_ms))
        if getattr(lat, "first_token_ms", None) is not None:
            tbl.add_row("TTFT", _format_ms(lat.first_token_ms))
        if getattr(lat, "error", None):
            tbl.add_row("Error", f"[error]{lat.error}[/]")

    match kind:
        case "chat":
            if data.content:
                tbl.add_row("Content", _truncate(data.content, 220))
            if data.finish_reason:
                tbl.add_row("Finish", data.finish_reason)
            if data.usage:
                tbl.add_row(
                    "Tokens",
                    f"in={data.usage.get('prompt_tokens', 0)} out={data.usage.get('completion_tokens', 0)}",
                )
        case "streaming":
            tbl.add_row("Chunks", str(data.chunks))
            if data.error:
                tbl.add_row("Error", f"[error]{data.error}[/]")
        case "tools":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            if data.tool_calls:
                tbl.add_row("Calls", str(len(data.tool_calls)))
            if data.error:
                tbl.add_row("Error", f"[error]{data.error}[/]")
        case "vision":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            if data.response_text:
                tbl.add_row("Response", _truncate(data.response_text, 220))
            if data.error:
                tbl.add_row("Error", f"[error]{data.error}[/]")
        case "json_mode":
            tbl.add_row("Supported", "[success]Yes[/]" if data.supported else "[error]No[/]")
            tbl.add_row("Parsed", "[success]Yes[/]" if getattr(data, "parsed_json", False) else "[error]No[/]")
            if data.response_text:
                tbl.add_row("Response", _truncate(data.response_text, 220))
            if data.error:
                tbl.add_row("Error", f"[error]{data.error}[/]")

    app.console.print(Panel(tbl, title=kind.capitalize(), border_style="panel.border"))


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
    return "connect"


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
    if base_url_raw.strip().lower() in {"b", "q"}:
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
        if api_key_raw.strip().lower() in {"b", "q"}:
            return "exit"
        from core.env import resolve_env

        api_key = resolve_env(api_key_raw)
        if api_key != api_key_raw:
            app.console.print("  [success]✓ resolved key[/]")

    if not base_url or not api_key:
        app.console.print(Panel("[error]Both Base URL and API key are required.[/]", border_style="red"))
        _nav(app.console, "Retry")
        choice = _ask()
        if choice == "q":
            return "exit"
        return "connect"

    app.console.print()
    with app.console.status("Connecting…", spinner=SPINNER_STYLE):
        ok = app.connect(base_url, api_key)

    if ok:
        return "main_menu"
    return "error"


def _error(app: ProviderInspectorApp) -> str:
    app.console.print()
    app.console.print(
        Panel(
            app.last_error_friendly,
            title=app.last_error_title,
            border_style="red",
        )
    )
    _nav(app.console, "Retry")
    choice = _ask()
    if choice == "q":
        return "exit"
    return "connect"


def _main_menu(app: ProviderInspectorApp) -> str | None:
    _set_console(app.console)
    _provider_header(app.console)

    items: list[tuple[str, str]] = [
        ("Browse Models", "browse_models"),
        ("Full Capability Scan", "full_scan"),
    ]
    if app.scan_results:
        items.append(("Browse Results", "scan_results"))
    items.append(("Reports", "reports"))
    items.append(("Settings", "settings"))

    app.console.print()
    for i, (label, _) in enumerate(items, 1):
        app.console.print(f"  [bold]{i}.[/] {label}")

    _nav(app.console, "Disconnect")
    choice = _ask(set(range(1, len(items) + 1)))

    if choice == "q":
        return "exit"
    if choice == "b":
        return "disconnect"

    if isinstance(choice, int) and 1 <= choice <= len(items):
        return items[choice - 1][1]
    return None


def _browse_models(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Browse Models[/bold]")

    if not session.models:
        app.console.print("\n  [warning]No models loaded.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    tbl = Table(show_header=True, header_style="table.header", box=box.SIMPLE, expand=True)
    tbl.add_column("#", style="dim", justify="right", width=4)
    tbl.add_column("Model", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Owner", style="dim", max_width=18, no_wrap=True, overflow="ellipsis")
    for i, model in enumerate(session.models, 1):
        tbl.add_row(str(i), model.id, model.owner)
    app.console.print(tbl)

    _nav(app.console)
    choice = _ask(set(range(1, len(session.models) + 1)))
    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"

    if isinstance(choice, int):
        app._selected_model_idx = choice - 1
        return "model_detail"
    return "back"


def _model_detail(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    idx = app._selected_model_idx
    if idx < 0 or idx >= len(session.models):
        app.console.print("\n  [warning]No model selected.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    model = session.models[idx]
    scan = _model_scan_for(model.id)

    app.console.print()
    app.console.rule(f"[bold]{model.id}[/bold]")
    app.console.print()

    for block in _render_model_detail(app, model, scan):
        app.console.print(block)
        app.console.print()

    app.console.print("  [bold]1.[/] Chat")
    app.console.print("  [bold]2.[/] Stream")
    app.console.print("  [bold]3.[/] Tools")
    app.console.print("  [bold]4.[/] Vision")
    app.console.print("  [bold]5.[/] JSON Mode")
    _nav(app.console)

    choice = _ask(set(range(1, len(SINGLE_TESTS) + 1)))
    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"

    if not app.client or not isinstance(choice, int):
        return "model_detail"

    action = SINGLE_TESTS[choice - 1][1]
    with app.console.status(f"Running {SINGLE_TESTS[choice - 1][0]}…", spinner=SPINNER_STYLE):
        result = _run_single_test(app.client, model.id, action)

    if not hasattr(app, "_model_results"):
        app._model_results = {}  # type: ignore[attr-defined]
    results_cache = app._model_results.setdefault(idx, [])
    results_cache.append(result)
    _display_single_result(app, *result)
    _nav(app.console)
    follow = _ask()
    if follow == "q":
        return "exit"
    return "model_detail"


def _full_scan(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Full Capability Scan[/bold]")

    if not session.models:
        app.console.print("\n  [warning]No models loaded.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    if not app.client:
        app.console.print("\n  [error]Not connected.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    state = _ScanDashboardState(total_models=len(session.models))
    worker = threading.Thread(target=_scan_model_worker, args=(app, state), daemon=True)
    worker.start()

    with Live(_scan_progress_layout(state, app.console.width), console=app.console, refresh_per_second=4, transient=True) as live:
        while worker.is_alive():
            time.sleep(1)
            live.update(_scan_progress_layout(state, app.console.width))
        worker.join()
        live.update(_scan_progress_layout(state, app.console.width))

    with state.lock:
        if state.error:
            app.console.print(Panel(state.error, title="Scan Error", border_style="red"))
            _nav(app.console)
            choice = _ask()
            if choice == "q":
                return "exit"
            return "back"
        results = list(state.results)

    session.scan_results = results
    app.scan_results = results
    return "scan_summary"


def _scan_summary(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    results = app.scan_results
    if not results:
        app.console.print("\n  [warning]No scan results in memory.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    total = len(results)
    discovered = sum(len(_finding_summary(scan)) for scan in results)
    observed = sum(1 for scan in results for _, ok in _finding_summary(scan) if ok)

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="muted", no_wrap=True)
    summary.add_column()
    summary.add_row("Models scanned", str(total))
    summary.add_row("Findings observed", str(observed))
    summary.add_row("Findings available", str(discovered))

    app.console.print()
    app.console.print(Panel(summary, title="Inspection Complete", border_style="panel.border"))

    issues = []
    for scan in results:
        for label, reason in _finding_reason(scan):
            issues.append((scan.model_id, label, reason))

    if issues:
        tbl = Table(show_header=True, header_style="table.header", box=box.SIMPLE, expand=True)
        tbl.add_column("Model", max_width=MODEL_ID_MAX, no_wrap=True, overflow="ellipsis")
        tbl.add_column("Capability", width=12)
        tbl.add_column("Note", overflow="ellipsis")
        for model_id, label, reason in issues:
            tbl.add_row(model_id, label, reason)
        app.console.print(tbl)

    app.console.print()
    app.console.print("  [bold]1.[/] Browse Results")
    app.console.print("  [bold]2.[/] Export JSON")
    app.console.print("  [bold]3.[/] Export CSV")
    _nav(app.console, "Back to Menu")

    choice = _ask({1, 2, 3})
    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"

    if choice == 1:
        return "scan_results"
    if choice == 2:
        fname = _save_report([asdict(s) for s in results], "json")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_summary"
    if choice == 3:
        fname = _save_report(results, "csv")
        app.console.print(f"\n  [success]Saved[/] reports/{fname}")
        return "scan_summary"
    return "scan_summary"


def _scan_results(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    results = app.scan_results

    if not results:
        app.console.print("\n  [warning]No scan results in memory.[/]")
        app.console.print("  [dim]Run a full scan first.[/]")
        _nav(app.console)
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    app.console.print()
    app.console.rule("[bold]Browse Results[/bold]")
    app.console.print()
    app.console.print(_render_scan_table(results, "Model Summary"))
    app.console.print()
    app.console.print("  [dim]Select a model for details.[/]")
    _nav(app.console)

    choice = _ask(set(range(1, len(results) + 1)))
    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"

    if isinstance(choice, int):
        app._selected_model_idx = choice - 1
        return "model_detail"
    return "back"


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
        choice = _ask()
        if choice == "q":
            return "exit"
        return "back"

    _nav(app.console)
    valid = set(range(1, item_num))
    choice = _ask(valid)

    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"

    if not isinstance(choice, int):
        return "reports"

    action = items[choice - 1]
    if action == "view_existing":
        _view_existing_reports(app)
        return "reports"
    if action == "export_current":
        app.console.print("\n  [bold]1.[/] JSON    [bold]2.[/] CSV")
        fmt_choice = _ask({1, 2})
        if fmt_choice == "q":
            return "exit"
        if fmt_choice == "b":
            return "reports"
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
    tbl = Table(show_header=True, header_style="table.header", box=box.SIMPLE, expand=True)
    tbl.add_column("#", style="dim", justify="right", width=4)
    tbl.add_column("Filename", overflow="ellipsis")
    tbl.add_column("Size", justify="right", width=10)
    for i, p in enumerate(existing, 1):
        size = p.stat().st_size
        size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
        tbl.add_row(str(i), p.name, size_str)
    app.console.print(tbl)

    _nav(app.console)
    choice = _ask(set(range(1, len(existing) + 1)))
    if choice in {"q", "b"}:
        return
    if not isinstance(choice, int):
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
    _ask()


def _delete_report(app: ProviderInspectorApp) -> None:
    existing = _list_saved_reports()
    if not existing:
        app.console.print("\n  [warning]Nothing to delete.[/]")
        return

    app.console.print()
    for i, p in enumerate(existing, 1):
        app.console.print(f"  [bold]{i}.[/] {p.name}")

    _nav(app.console)
    choice = _ask(set(range(1, len(existing) + 1)))
    if choice in {"q", "b"}:
        return
    if not isinstance(choice, int):
        return

    chosen = existing[choice - 1]
    app.console.print(f"\n  [warning]Delete {chosen.name}?[/]")
    app.console.print("  [bold]1.[/] Yes    [bold]2.[/] No")
    confirm = _ask({1, 2})
    if confirm in {"q", "b"} or confirm == 2:
        app.console.print("  [dim]Cancelled.[/]")
        return

    try:
        chosen.unlink()
        app.console.print(f"\n  [success]Deleted[/] {chosen.name}")
    except OSError as exc:
        app.console.print(f"\n  [error]Failed: {exc}[/]")


def _settings(app: ProviderInspectorApp) -> str:
    _set_console(app.console)
    app.console.print()
    app.console.rule("[bold]Settings[/bold]")
    app.console.print()
    app.console.print("  [bold]1.[/] Reconnect")
    app.console.print("  [bold]2.[/] Disconnect")
    _nav(app.console)

    choice = _ask({1, 2})
    if choice == "q":
        return "exit"
    if choice == "b":
        return "back"
    if choice == 1:
        if app.scan_results:
            app.console.print("\n  [warning]Reconnecting will discard scan results.[/]")
            app.console.print("  [bold]1.[/] Continue    [bold]2.[/] Cancel")
            confirm = _ask({1, 2})
            if confirm == 2:
                return "settings"
        return "connect"
    if choice == 2:
        return "disconnect"
    return "back"


_SCREEN_MAP: dict[Screen, Callable] = {
    Screen.SPLASH: _splash,
    Screen.CONNECTION: _connection,
    Screen.ERROR_PANEL: _error,
    Screen.MAIN_MENU: _main_menu,
    Screen.BROWSE_MODELS: _browse_models,
    Screen.MODEL_DETAIL: _model_detail,
    Screen.FULL_SCAN: _full_scan,
    Screen.SCAN_SUMMARY: _scan_summary,
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
    "scan_summary": Screen.SCAN_SUMMARY,
    "scan_results": Screen.SCAN_RESULTS,
    "reports": Screen.REPORTS,
    "settings": Screen.SETTINGS,
    "disconnect": Screen.CONNECTION,
}


def render(app: ProviderInspectorApp) -> str | None:
    app.console.clear()

    handler = _SCREEN_MAP.get(app.screen)
    if handler is None:
        return None

    action = handler(app)
    if action is None or action == "exit":
        return None
    if action == "disconnect":
        app.disconnect()
        app.screen = Screen.CONNECTION
        return action
    if action == "back":
        app.back()
        return action
    if action == "scan_summary":
        app.screen = Screen.SCAN_SUMMARY
        return action

    target = _ACTION_TO_SCREEN.get(action)
    if target is not None:
        app.go(target)
    return action
