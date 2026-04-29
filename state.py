"""
Persistent state for the event-driven scheduler.

Stores:
  - current capital (updated after each close cycle)
  - open positions (set during open, cleared during close)

State survives process restarts. On restart, the scheduler closes any
positions still in the file before opening new ones.
"""

import json
import os
from datetime import datetime as dt
from typing import Any

_STATE_DIR  = "state"
_STATE_FILE = os.path.join(_STATE_DIR, "positions.json")

_DEFAULT: dict[str, Any] = {
    "capital":   100_000.0,
    "positions": [],
}


def _load() -> dict[str, Any]:
    os.makedirs(_STATE_DIR, exist_ok=True)
    if not os.path.exists(_STATE_FILE):
        return dict(_DEFAULT)
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)


def _save(state: dict[str, Any]) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_capital() -> float:
    return float(_load()["capital"])


def set_capital(capital: float) -> None:
    state = _load()
    state["capital"] = capital
    _save(state)


def get_open_positions() -> list[dict]:
    return _load()["positions"]


def has_open_positions() -> bool:
    return len(get_open_positions()) > 0


def save_open_positions(positions: list[dict], capital: float) -> None:
    """Atomically record newly opened positions and updated capital."""
    state = _load()
    state["positions"] = positions
    state["capital"] = capital
    _save(state)


def clear_positions(capital: float) -> None:
    """Clear positions after close cycle and record realised capital."""
    state = _load()
    state["positions"] = []
    state["capital"] = capital
    _save(state)


def reset(initial_capital: float = 100_000.0) -> None:
    """Wipe all state. Use at start of a fresh simulation run."""
    _save({"capital": initial_capital, "positions": []})
