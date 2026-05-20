from __future__ import annotations

from dataclasses import dataclass

import torch

from differentiableneck.geometry import Plane, initialize_control_points
from differentiableneck.losses import total_neckspline_loss
from differentiableneck.models import NeckSpline


@dataclass
class StageConfig:
    control_points: int
    iterations: int
    profile_radius: float


def default_stages() -> list[StageConfig]:
    return [
        StageConfig(control_points=8, iterations=100, profile_radius=1.0),
        StageConfig(control_points=12, iterations=100, profile_radius=0.6),
        StageConfig(control_points=16, iterations=100, profile_radius=0.3),
    ]


def fit_neckspline(
    plane: Plane,
    image_volume: torch.Tensor,
    boundary_volume: torch.Tensor,
    gradient_volume: torch.Tensor,
    parent_radius: float,
    weights: dict[str, float],
    stages: list[StageConfig] | None = None,
    lr: float = 0.02,
    weight_decay: float = 1e-4,
    num_samples: int = 128,
    reference_curvature: float = 1.0,
) -> tuple[NeckSpline, list[dict[str, float]]]:
    """Coarse-to-fine NeckSpline fitting loop from the paper."""

    stages = stages or default_stages()
    control_points = initialize_control_points(
        plane,
        radius=parent_radius,
        num_points=stages[0].control_points,
    )
    model = NeckSpline(control_points, plane)
    history: list[dict[str, float]] = []

    for stage_id, stage in enumerate(stages):
        if stage_id > 0:
            model = NeckSpline(
                model.spline.subdivide(stage.control_points).control_points.detach(),
                model.plane,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        for _ in range(stage.iterations):
            losses = total_neckspline_loss(
                model,
                image_volume=image_volume,
                boundary_volume=boundary_volume,
                gradient_volume=gradient_volume,
                parent_radius=parent_radius,
                weights=weights,
                num_samples=num_samples,
                profile_radius=stage.profile_radius,
                reference_curvature=reference_curvature,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            history.append(
                {
                    name: float(value.detach().cpu())
                    for name, value in losses.items()
                }
            )

    return model, history
