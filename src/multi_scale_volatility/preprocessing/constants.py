"""Preprocessing constants."""

from __future__ import annotations

from datetime import timedelta, timezone

RAW_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
FIXED_EST = timezone(timedelta(hours=-5))
UTC = timezone.utc

