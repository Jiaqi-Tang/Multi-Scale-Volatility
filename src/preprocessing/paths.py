"""Preprocessing path configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.paths import (
    CLEAN_1M_CSV,
    CLEAN_RETURNS_CSV,
    INTERMEDIATE_DIR,
    OHLC_5M_CSV,
    PREPROCESSING_REPORT_JSON,
    RAW_METATRADER_DIR,
)


@dataclass(frozen=True)
class PreprocessingPaths:
    raw_dir: Path = RAW_METATRADER_DIR
    intermediate_dir: Path = INTERMEDIATE_DIR

    @property
    def clean_1m_csv(self) -> Path:
        return CLEAN_1M_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / CLEAN_1M_CSV.name
        )

    @property
    def ohlc_5m_csv(self) -> Path:
        return OHLC_5M_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / OHLC_5M_CSV.name
        )

    @property
    def report_json(self) -> Path:
        return PREPROCESSING_REPORT_JSON if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / PREPROCESSING_REPORT_JSON.name
        )

    @property
    def clean_returns_csv(self) -> Path:
        return CLEAN_RETURNS_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / CLEAN_RETURNS_CSV.name
        )

