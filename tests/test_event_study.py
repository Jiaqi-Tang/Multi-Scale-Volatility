from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from multi_scale_volatility.research.event_study.events import (
    EVENT_WINDOW_LENGTH,
    POST_EVENT_END_OFFSET,
    PRE_EVENT_OBSERVATIONS,
    annotate_overlaps,
    apply_state_machine,
    robust_causal_scores,
    trailing_rms,
)
from multi_scale_volatility.research.decomposition import decompose_values
from multi_scale_volatility.plotting.event_eda import (
    add_event_calendar_fields,
    build_weekday_hour_counts,
    build_consecutive_event_intervals,
    select_example_events,
)


class EventStudyTests(unittest.TestCase):
    def test_consecutive_event_intervals_exclude_first_event(self) -> None:
        catalog = pd.DataFrame({
            "event_id": [2, 0, 1],
            "anchor_index": [864, 0, 288],
            "event_timestamp_utc": [
                "2026-01-04 00:00:00+00:00",
                "2026-01-01 00:00:00+00:00",
                "2026-01-02 00:00:00+00:00",
            ],
        })
        intervals = build_consecutive_event_intervals(catalog)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals["interval_observations"].tolist(), [288, 576])
        self.assertEqual(intervals["interval_indexed_days"].tolist(), [1.0, 2.0])
    def test_weekday_hour_counts_include_zero_cells(self) -> None:
        catalog = add_event_calendar_fields(pd.DataFrame({
            "event_timestamp_utc": ["2026-07-12 22:05:00+00:00"]
        }))
        counts = build_weekday_hour_counts(catalog)
        self.assertEqual(len(counts), 7 * 24)
        self.assertEqual(int(counts["event_count"].sum()), 1)
        sunday_22 = counts[(counts["weekday"] == "Sunday") & (counts["utc_hour"] == 22)]
        self.assertEqual(int(sunday_22["event_count"].iloc[0]), 1)

    def test_example_selection_is_reproducible_and_excludes_endpoints(self) -> None:
        catalog = pd.DataFrame({
            "event_id": range(10),
            "anchor_index": np.arange(10) * 100,
            "event_timestamp_utc": [f"2026-01-{day:02d} 00:00:00+00:00" for day in range(1, 11)],
        })
        first = select_example_events(catalog, random_seed=7, random_event_count=3)
        second = select_example_events(catalog, random_seed=7, random_event_count=3)
        pd.testing.assert_frame_equal(first, second)
        random_ids = first.loc[first["selection_reason"].str.startswith("random"), "event_id"]
        self.assertFalse(random_ids.isin([0, 9]).any())
    def test_trailing_rms_includes_current_observation(self) -> None:
        actual = trailing_rms(np.array([1.0, 2.0, 3.0]), 2)
        np.testing.assert_allclose(
            actual,
            np.array([np.nan, np.sqrt(2.5), np.sqrt(6.5)]),
            equal_nan=True,
        )

    def test_robust_score_uses_only_preceding_reference(self) -> None:
        volatility = np.exp(np.array([0.0, 1.0, 2.0, 10.0]))
        medians, mads, scores = robust_causal_scores(
            volatility, reference_length=3, chunk_size=2
        )
        self.assertEqual(medians[3], 1.0)
        self.assertEqual(mads[3], 1.0)
        self.assertAlmostEqual(scores[3], 9.0 / 1.4826)

    def test_state_machine_requires_crossing_and_reset(self) -> None:
        primary = np.array([0.0, 4.2, 4.5, 0.5, 0.4, 3.0, 4.1])
        triggers, resets, active, streaks = apply_state_machine(
            primary, reset_length=2
        )
        np.testing.assert_array_equal(np.flatnonzero(triggers), [1, 5])
        np.testing.assert_array_equal(np.flatnonzero(resets), [4])
        self.assertTrue(active[1])
        self.assertFalse(active[4])
        self.assertEqual(streaks[3], 1)

    def test_overlap_clusters_are_transitive(self) -> None:
        catalog = pd.DataFrame(
            {
                "event_id": [0, 1, 2, 3],
                "window_start_index": [0, 8, 16, 40],
                "window_end_index": [10, 18, 26, 50],
                "is_window_eligible": [True, True, True, True],
            }
        )
        actual = annotate_overlaps(catalog)
        self.assertEqual(actual.loc[0, "overlap_cluster_id"], 0)
        self.assertEqual(actual.loc[2, "overlap_cluster_id"], 0)
        self.assertTrue(pd.isna(actual.loc[3, "overlap_cluster_id"]))
        self.assertEqual(actual.loc[1, "overlap_event_count"], 2)
        self.assertEqual(actual.loc[0, "overlap_event_count"], 1)

    def test_event_window_supports_nine_level_reconstruction(self) -> None:
        self.assertEqual(
            PRE_EVENT_OBSERVATIONS + 1 + POST_EVENT_END_OFFSET,
            EVENT_WINDOW_LENGTH,
        )
        self.assertEqual(EVENT_WINDOW_LENGTH % 512, 0)
        values = np.linspace(-1.0, 1.0, EVENT_WINDOW_LENGTH)
        details, approximation = decompose_values(values, k=9)
        reconstruction = approximation + np.sum(details, axis=0)
        np.testing.assert_allclose(reconstruction, values, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
