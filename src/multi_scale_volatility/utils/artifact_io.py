"""Atomic writers for generated artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        temp_path.write_text(text, encoding=encoding)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path, **to_csv_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        frame.to_csv(temp_path, **to_csv_kwargs)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)

