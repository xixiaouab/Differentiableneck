from __future__ import annotations

import torch

from differentiableneck.geometry import Plane, sample_trilinear


def boundary_likelihood_loss(boundary_volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    values = sample_trilinear(boundary_volume, points).clamp_min(1e-6)
    return -values.log().mean()


def gradient_alignment_loss(
    gradient_volume: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    gradients = sample_trilinear(gradient_volume, points)
    return -(gradients * normals).sum(dim=-1).abs().mean()


def profile_symmetry_loss(
    image_volume: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    radius: float = 0.3,
) -> torch.Tensor:
    inside = sample_trilinear(image_volume, points + radius * normals)
    outside = sample_trilinear(image_volume, points - radius * normals)
    return (inside - outside).pow(2).mean()


def curvature_tightness_loss(
    curvature: torch.Tensor,
    reference_curvature: float,
) -> torch.Tensor:
    return (curvature - reference_curvature).pow(2).mean()


def loop_scale_loss(
    points: torch.Tensor,
    parent_radius: float,
    circumference_target: float = 6.3,
    area_target: float = 3.1,
    area_weight: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    edges = torch.roll(points, -1, dims=0) - points
    circumference = edges.norm(dim=-1).sum()
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=2)
    planar = centered @ basis
    x, y = planar[:, 0], planar[:, 1]
    area = 0.5 * torch.abs((x * torch.roll(y, -1) - y * torch.roll(x, -1)).sum())
    radius = max(parent_radius, eps)
    return (
        (circumference / radius - circumference_target).pow(2)
        + area_weight * (area.sqrt() / radius - area_target).pow(2)
    )


def self_intersection_loss(
    points: torch.Tensor,
    min_distance: float = 0.05,
    exclusion: int = 3,
) -> torch.Tensor:
    n = points.shape[0]
    distances = torch.cdist(points, points)
    ids = torch.arange(n, device=points.device)
    circular_gap = torch.abs(ids[:, None] - ids[None, :])
    circular_gap = torch.minimum(circular_gap, n - circular_gap)
    mask = circular_gap > exclusion
    penalty = torch.relu(min_distance - distances[mask])
    if penalty.numel() == 0:
        return points.new_tensor(0.0)
    return penalty.pow(2).mean()


def plane_distance_loss(points: torch.Tensor, plane: Plane) -> torch.Tensor:
    distances = points @ plane.normal + plane.offset
    return distances.pow(2).mean()


def centerline_anchor_loss(points: torch.Tensor, plane: Plane) -> torch.Tensor:
    centroid = points.mean(dim=0)
    return (centroid - plane.center).pow(2).sum()


def euler_characteristic_proxy_loss(
    points: torch.Tensor,
    closure_weight: float = 1.0,
    separation_weight: float = 1.0,
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Differentiable proxy for the paper's EC topology term.

    Exact Euler characteristic computation requires discrete cubical-complex
    bookkeeping. This scaffold keeps gradients on the control points by using
    loop closure and non-adjacent separation as the trainable topology proxy.
    """

    edges = (torch.roll(points, -1, dims=0) - points).norm(dim=-1)
    closure = (edges[-1] - edges.mean()).pow(2)
    separation = self_intersection_loss(points, min_distance=min_distance)
    return closure_weight * closure + separation_weight * separation


def total_neckspline_loss(
    model,
    image_volume: torch.Tensor,
    boundary_volume: torch.Tensor,
    gradient_volume: torch.Tensor,
    parent_radius: float,
    weights: dict[str, float],
    num_samples: int = 128,
    profile_radius: float = 0.3,
    reference_curvature: float = 1.0,
) -> dict[str, torch.Tensor]:
    points = model.sample(num_samples)
    normals = model.normals_in_plane(num_samples)
    curvature = model.spline.curvature(num_samples)
    plane = model.plane

    losses = {
        "boundary": boundary_likelihood_loss(boundary_volume, points),
        "gradient": gradient_alignment_loss(gradient_volume, points, normals),
        "profile": profile_symmetry_loss(image_volume, points, normals, profile_radius),
        "tightness": curvature_tightness_loss(curvature, reference_curvature),
        "scale": loop_scale_loss(points, parent_radius),
        "self": self_intersection_loss(points),
        "ec": euler_characteristic_proxy_loss(points),
        "plane": plane_distance_loss(points, plane),
        "anchor": centerline_anchor_loss(points, plane),
    }
    total = points.new_tensor(0.0)
    for name, value in losses.items():
        total = total + weights.get(name, 1.0) * value
    losses["total"] = total
    return losses
