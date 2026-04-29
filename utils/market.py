from datetime import datetime as dt
import pytz


def is_market_open() -> bool:
    """True if the NYSE is currently in regular session (Mon–Fri, 09:30–16:00 ET)."""
    eastern = pytz.timezone("US/Eastern")
    now = dt.now(eastern)
    if now.weekday() >= 5:
        return False
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ <= now <= close
