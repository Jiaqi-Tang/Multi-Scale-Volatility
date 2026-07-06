"""Component metadata for dyadic decomposition layers."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from multi_scale_volatility.core.config.names import BASE_INTERVAL_MINUTES
from multi_scale_volatility.core.config.names import ComponentType


def component_type(component: str) -> ComponentType:
    if component == "original":
        return "original"
    if re.fullmatch(r"D_\d{2}", component):
        return "detail"
    if re.fullmatch(r"A_\d{2}", component):
        return "approximation"
    raise ValueError(f"Unrecognized component name: {component}")


def component_scale(component: str) -> int:
    kind = component_type(component)
    if kind == "original":
        return 0
    return int(component.split("_", maxsplit=1)[1])


def component_repeat_length(component: str) -> int:
    kind = component_type(component)
    scale = component_scale(component)
    if kind == "original":
        return 1
    if kind == "detail":
        return 2 ** (scale - 1)
    return 2**scale


def compress_component(values: np.ndarray, component: str) -> np.ndarray:
    return values[::component_repeat_length(component)]


def original_lags_from_compressed_lags(
    compressed_lags: np.ndarray,
    component: str,
) -> np.ndarray:
    return compressed_lags * component_repeat_length(component)


def component_scale_minutes(
    component: str,
    base_interval_minutes: int = BASE_INTERVAL_MINUTES,
) -> int:
    kind = component_type(component)
    scale = component_scale(component)
    if scale == 0:
        return base_interval_minutes
    if kind == "detail":
        return base_interval_minutes * (2 ** (scale - 1))
    return base_interval_minutes * (2**scale)


def decomposition_components(k: int, include_original: bool = False) -> list[str]:
    if k < 1:
        raise ValueError("k must be at least 1")
    components = [f"D_{scale:02d}" for scale in range(1, k + 1)] + [f"A_{k:02d}"]
    if include_original:
        return ["original", *components]
    return components


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
