"""Immutable report style for the final Phase 2E source-scale figure pack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FinalFigureStyle:
    font_family: str = "DejaVu Sans"
    base_font_size: float = 10.0
    title_font_size: float = 12.0
    panel_label_font_size: float = 11.0
    line_width: float = 1.6
    intensity_colormap: str = "magma"
    difference_colormap: str = "RdBu_r"
    axis_unit: str = "mm"
    detail_halfwidth_m: float = 0.25e-3
    snapshot_halfwidth_m: float = 0.25e-3
    surface_halfwidth_m: float = 0.12e-3
    z_limits_m: tuple[float, float] = (0.0, 0.180)
    primary_normalisation: str = "global_linear"
    primary_scaling: str = "linear"
    display_interpolation: str = "bilinear"
    display_interpolation_used_for_metrics: bool = False
    surface_display_upsampling: int = 2
    surface_elevation_deg: float = 32.0
    surface_azimuth_deg: float = -52.0
    surface_intensity_limits: tuple[float, float] = (0.0, 1.0)
    output_dpi: int = 320
    primary_figsize_inches: tuple[float, float] = (16.0, 12.0)
    comparison_figsize_inches: tuple[float, float] = (16.0, 12.0)
    snapshot_figsize_inches: tuple[float, float] = (15.0, 8.0)
    surface_figsize_inches: tuple[float, float] = (12.0, 5.6)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FINAL_FIGURE_STYLE = FinalFigureStyle()
