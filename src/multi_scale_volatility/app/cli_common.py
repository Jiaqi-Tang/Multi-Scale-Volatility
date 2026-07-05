"""Shared helpers for command-line handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def print_paths(paths: list[Path]) -> None:
    for path in paths:
        print(path)


def json_ready_summary(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, list):
            summary[key] = [str(item) for item in value]
        else:
            summary[key] = value
    return summary
