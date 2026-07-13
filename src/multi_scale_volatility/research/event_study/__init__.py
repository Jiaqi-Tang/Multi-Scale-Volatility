"""V3 event detection and event-aligned decomposition."""

from multi_scale_volatility.research.event_study.events import (
    EventStudyPaths,
    detect_events,
    extract_event_windows,
)

__all__ = ["EventStudyPaths", "detect_events", "extract_event_windows"]
