"""Session state management."""

from __future__ import annotations

from core.models import ProviderSession


# Single session instance — not global mutable state accessible module-wide.
# Created fresh per application run.
session = ProviderSession()
