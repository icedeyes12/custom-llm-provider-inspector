"""Provider Inspector — interactive Rich TUI application."""

from __future__ import annotations

from enum import Enum, auto

from rich.console import Console

from core.api import APIClient
from core.models import CapabilityScan
from core.scanner import refresh_models
from core.session import session
from ui.theme import PALETTE


class Screen(Enum):
    SPLASH = auto()
    CONNECTION = auto()
    ERROR_PANEL = auto()
    MAIN_MENU = auto()
    BROWSE_MODELS = auto()
    MODEL_DETAIL = auto()
    FULL_SCAN = auto()
    SCAN_SUMMARY = auto()
    SCAN_RESULTS = auto()
    REPORTS = auto()
    SETTINGS = auto()


class ProviderInspectorApp:
    """State-machine driven TUI that orchestrates core modules."""

    def __init__(self) -> None:
        self.console = Console(theme=PALETTE)
        self.screen: Screen = Screen.SPLASH
        self.client: APIClient | None = None
        self._nav_stack: list[Screen] = []
        self._selected_model_idx: int = 0
        self._last_error: str = ""
        self._last_error_friendly: str = ""
        self._last_error_title: str = "Error"
        self.scan_results: list[CapabilityScan] = []

    # -- navigation --------------------------------------------------------

    def go(self, screen: Screen) -> None:
        self._nav_stack.append(self.screen)
        self.screen = screen

    def back(self) -> None:
        if self._nav_stack:
            self.screen = self._nav_stack.pop()
        else:
            self.screen = Screen.MAIN_MENU

    # -- connection --------------------------------------------------------

    def connect(self, base_url: str, api_key: str) -> bool:
        try:
            client = APIClient(base_url, api_key)
            raw_models, lat = client.list_models()
            if lat.error:
                self._last_error = lat.error
                title, friendly = _translate_error(lat.error, lat.status_code)
                self._last_error_friendly = friendly
                self._last_error_title = title
                return False
            if not raw_models and lat.status_code == 0:
                self._last_error = "Empty response from /models"
                title, friendly = _translate_error("empty", 0)
                self._last_error_friendly = friendly
                self._last_error_title = title
                return False
            session.base_url = base_url
            session.api_key = api_key
            refresh_models(session, raw_models)
            self.client = client
            self.scan_results = []
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            title, friendly = _translate_error(str(exc), 0)
            self._last_error_friendly = friendly
            self._last_error_title = title
            return False

    def disconnect(self) -> None:
        session.base_url = ""
        session.api_key = ""
        session.models = []
        session.selected_model = ""
        session.scan_results = []
        session.benchmark_results = []
        self.client = None
        self.scan_results = []

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_error_friendly(self) -> str:
        return self._last_error_friendly

    @property
    def last_error_title(self) -> str:
        return self._last_error_title

    @property
    def is_connected(self) -> bool:
        return self.client is not None and bool(session.base_url)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        from ui.screens import render

        while True:
            action = render(self)
            if action is None or action == "exit":
                break


def _translate_error(raw: str, status_code: int) -> tuple[str, str]:
    """Return (title, friendly_message) for an error."""
    raw_lower = raw.lower()

    if status_code == 401:
        return ("Authentication Failed", "Please verify your API key.")
    if status_code == 403:
        return ("Access Denied", "Your API key does not have permission\nto access this endpoint.")
    if status_code == 404:
        return ("Not Found", "The endpoint does not appear to implement\nan OpenAI-compatible API.\n\nExpected: /v1/models")
    if status_code == 422:
        return ("Format Rejected", "The provider rejected the request format.\n\nThis endpoint may not be OpenAI-compatible.")
    if status_code and status_code >= 500:
        return ("Provider Error", f"The provider returned HTTP {status_code}.\n\nTry again later.")

    if "name or service not known" in raw_lower or "dns" in raw_lower:
        return ("DNS Failed", "Unable to resolve the provider address.\n\nPlease verify:\n  • Internet connection\n  • Base URL")
    if "timed out" in raw_lower or "timeout" in raw_lower:
        return ("Timeout", "The provider did not respond in time.\n\nCheck your connection or try a different provider.")
    if "connection refused" in raw_lower:
        return ("Connection Refused", "The provider may be down or the URL is incorrect.")
    if "ssl" in raw_lower:
        return ("SSL Error", "SSL/TLS handshake failed.\n\nVerify the Base URL uses HTTPS\nand the provider has a valid certificate.")
    if "not found" in raw_lower and "env" in raw_lower:
        return ("Env Not Found", "Environment variable not found.\n\nVerify your environment or enter the value manually.")

    shortened = raw[:200] if len(raw) > 200 else raw
    return ("Connection Failed", f"{shortened}")
