from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Plane:
    """Local ostium plane n^T x + d = 0."""

    normal: torch.Tensor
    offset: torch.Tensor
    center: torch.Tensor
    tangent: torch.Tensor


def normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def orthonormal_frame(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a stable local frame with e3 aligned to ``normal``."""

    e3 = normalize(normal.reshape(1, 3)).squeeze(0)
    reference = torch.tensor([1.0, 0.0, 0.0], device=e3.device, dtype=e3.dtype)
    if torch.abs(torch.dot(e3, reference)) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0], device=e3.device, dtype=e3.dtype)
    e1 = normalize(torch.cross(e3, reference, dim=0).reshape(1, 3)).squeeze(0)
    e2 = normalize(torch.cross(e3, e1, dim=0).reshape(1, 3)).squeeze(0)
    return e1, e2, e3


def weighted_pca_plane(
    points: torch.Tensor,
    weights: torch.Tensor,
    tangent: torch.Tensor,
    tangent_weight: float = 0.25,
) -> Plane:
    """Robust-PCA-style plane fit regularized toward the centerline tangent."""

    weights = weights.clamp_min(1e-8)
    weights = weights / weights.sum()
    center = (weights[:, None] * points).sum(dim=0)
    centered = points - center
    covariance = centered.t() @ (centered * weights[:, None])
    _, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    tangent = normalize(tangent.reshape(1, 3)).squeeze(0)
    if torch.dot(normal, tangent).abs() < tangent_weight:
        normal = normalize((normal + tangent_weight * tangent).reshape(1, 3)).squeeze(0)
    if torch.dot(normal, tangent) < 0:
        normal = -normal
    offset = -torch.dot(normal, center)
    return Plane(normal=normal, offset=offset, center=center, tangent=tangent)


def initialize_control_points(
    plane: Plane,
    radius: float,
    num_points: int = 16,
) -> torch.Tensor:
    """Initialize a closed loop as a regular polygon on the ostium plane."""

    e1, e2, _ = orthonormal_frame(plane.normal)
    theta = torch.linspace(
        0,
        2 * torch.pi,
        steps=num_points + 1,
        device=plane.center.device,
        dtype=plane.center.dtype,
    )[:-1]
    offsets = radius * (theta.cos()[:, None] * e1 + theta.sin()[:, None] * e2)
    return plane.center.unsqueeze(0) + offsets


def sample_trilinear(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample a 3D volume at normalized coordinates in [-1, 1]."""

    if volume.ndim == 3:
        volume = volume[None, None]
    elif volume.ndim == 4:
        volume = volume[None]
    grid = points.reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(volume.shape[1], -1).transpose(0, 1)


def project_to_plane(points: torch.Tensor, plane: Plane) -> torch.Tensor:
    distances = points @ plane.normal + plane.offset
    return points - distances[:, None] * plane.normal[None]


def pairwise_chamfer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(left, right).pow(2)
    return distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()


def caliper_width(points: torch.Tensor, num_angles: int = 180) -> torch.Tensor:
    """Maximum projected chord length through the loop centroid."""

    centered = points - points.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=2)
    planar = centered @ basis
    angles = torch.linspace(
        0,
        torch.pi,
        steps=num_angles,
        device=points.device,
        dtype=points.dtype,
    )
    directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
    projections = planar @ directions.t()
    return (projections.max(dim=0).values - projections.min(dim=0).values).max()
