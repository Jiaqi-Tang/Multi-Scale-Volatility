"""Helpers for deriving artifact paths from overridable output directories."""

from __future__ import annotations

from pathlib import Path

from multi_scale_volatility.config.bundles import SeriesBundle


def resolve_artifact_path(output_dir: Path, default_dir: Path, default_path: Path) -> Path:
    if output_dir == default_dir:
        return default_path
    return output_dir / default_path.name


def resolve_artifact_bundle(
    output_dir: Path,
    default_dir: Path,
    default_paths: SeriesBundle[Path],
) -> SeriesBundle[Path]:
    return SeriesBundle(
        final=resolve_artifact_path(
            output_dir, default_dir, default_paths.final),
        shuffle=resolve_artifact_path(
            output_dir, default_dir, default_paths.shuffle),
        gaussian=resolve_artifact_path(
            output_dir, default_dir, default_paths.gaussian),
    )
