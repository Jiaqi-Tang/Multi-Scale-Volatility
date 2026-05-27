"""Component metadata for dyadic decomposition layers."""

from __future__ import annotations

from dataclasses import dataclass

from multi_scale_volatility.config.constants import BASE_INTERVAL_MINUTES
from multi_scale_volatility.config.types import ComponentType
from multi_scale_volatility.scale_utils import (
    component_repeat_length,
    component_scale,
    component_scale_minutes,
    component_type,
    decomposition_components,
)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: ComponentType
    scale: int
    scale_minutes: int
    repeat_length: int

    @property
    def scale_days(self) -> float:
        return self.scale_minutes / (60 * 24)


def component_spec(
    component: str,
    base_interval_minutes: int = BASE_INTERVAL_MINUTES,
) -> ComponentSpec:
    return ComponentSpec(
        name=component,
        kind=component_type(component),
        scale=component_scale(component),
        scale_minutes=component_scale_minutes(
            component, base_interval_minutes),
        repeat_length=component_repeat_length(component),
    )


def component_specs(
    k: int,
    include_original: bool = False,
    base_interval_minutes: int = BASE_INTERVAL_MINUTES,
) -> list[ComponentSpec]:
    return [
        component_spec(component, base_interval_minutes)
        for component in decomposition_components(k, include_original=include_original)
    ]
