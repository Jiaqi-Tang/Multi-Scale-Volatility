"""V3 event-study command-line commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_scale_volatility.app.cli_common import print_json
from multi_scale_volatility.core.config.paths import EVENT_STUDY_RESULTS_DIR, FINAL_RETURNS_CSV
from multi_scale_volatility.research.event_study.events import (
    EVENT_K,
    REFERENCE_CHUNK_SIZE,
    REFERENCE_LENGTH,
    EventStudyPaths,
    detect_events,
    extract_event_windows,
)


def add_event_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("events", help="Run V3 volatility-event stages.")
    event_subparsers = parser.add_subparsers(dest="event_step")

    detection = event_subparsers.add_parser("detect", help="Detect volatility events.")
    _add_paths(detection)
    detection.add_argument("--reference-length", type=int, default=REFERENCE_LENGTH)
    detection.add_argument("--chunk-size", type=int, default=REFERENCE_CHUNK_SIZE)
    detection.set_defaults(handler=_handle_detect)

    windows = event_subparsers.add_parser(
        "windows", help="Extract and decompose eligible event windows."
    )
    _add_paths(windows)
    windows.add_argument("--k", type=int, default=EVENT_K)
    windows.set_defaults(handler=_handle_windows)

    all_stages = event_subparsers.add_parser("all", help="Run detection and windows.")
    _add_paths(all_stages)
    all_stages.add_argument("--reference-length", type=int, default=REFERENCE_LENGTH)
    all_stages.add_argument("--chunk-size", type=int, default=REFERENCE_CHUNK_SIZE)
    all_stages.add_argument("--k", type=int, default=EVENT_K)
    all_stages.set_defaults(handler=_handle_all)


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--output-dir", type=Path, default=EVENT_STUDY_RESULTS_DIR)


def _paths(args: argparse.Namespace) -> EventStudyPaths:
    return EventStudyPaths(input_csv=args.input_csv, output_dir=args.output_dir)


def _handle_detect(args: argparse.Namespace) -> None:
    print_json(detect_events(_paths(args), args.reference_length, args.chunk_size))


def _handle_windows(args: argparse.Namespace) -> None:
    print_json(extract_event_windows(_paths(args), k=args.k))


def _handle_all(args: argparse.Namespace) -> None:
    paths = _paths(args)
    print_json({
        "detection": detect_events(paths, args.reference_length, args.chunk_size),
        "windows": extract_event_windows(paths, k=args.k),
    })
