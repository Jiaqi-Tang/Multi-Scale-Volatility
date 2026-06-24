"""Small report-shape helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingOutputs:
    clean_1m_csv: Path
    ohlc_5m_csv: Path
    clean_returns_csv: Path
    report_json: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "clean_1m_csv": str(self.clean_1m_csv),
            "ohlc_5m_csv": str(self.ohlc_5m_csv),
            "clean_returns_csv": str(self.clean_returns_csv),
            "report_json": str(self.report_json),
        }
