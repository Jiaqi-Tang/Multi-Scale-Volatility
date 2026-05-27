"""Small containers for values keyed by canonical analysis series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, Mapping, TypeVar

from multi_scale_volatility.config.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_ORDER, SERIES_SHUFFLE

T = TypeVar("T")


@dataclass(frozen=True)
class SeriesBundle(Generic[T]):
    """Values for the final, shuffled, and Gaussian baseline series."""

    final: T
    shuffle: T
    gaussian: T

    def items(self) -> tuple[tuple[str, T], ...]:
        return (
            (SERIES_FINAL, self.final),
            (SERIES_SHUFFLE, self.shuffle),
            (SERIES_GAUSSIAN, self.gaussian),
        )

    def values(self) -> tuple[T, T, T]:
        return (self.final, self.shuffle, self.gaussian)

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self.items())

    def __getitem__(self, series: str) -> T:
        if series == SERIES_FINAL:
            return self.final
        if series == SERIES_SHUFFLE:
            return self.shuffle
        if series == SERIES_GAUSSIAN:
            return self.gaussian
        raise KeyError(series)

    @classmethod
    def from_mapping(cls, values: Mapping[str, T]) -> "SeriesBundle[T]":
        missing = [series for series in SERIES_ORDER if series not in values]
        if missing:
            raise KeyError(f"Missing series values: {missing}")
        return cls(
            final=values[SERIES_FINAL],
            shuffle=values[SERIES_SHUFFLE],
            gaussian=values[SERIES_GAUSSIAN],
        )
