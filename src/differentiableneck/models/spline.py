from __future__ import annotations

import torch
from torch import nn

from differentiableneck.geometry import Plane, caliper_width, normalize


class PeriodicCubicBSpline(nn.Module):
    """Closed uniform cubic B-spline with learnable 3D control points."""

    def __init__(self, control_points: torch.Tensor) -> None:
        super().__init__()
        if control_points.ndim != 2 or control_points.shape[1] != 3:
            raise ValueError("control_points must have shape [num_points, 3].")
        self.control_points = nn.Parameter(control_points.clone())

    @property
    def num_control_points(self) -> int:
        return self.control_points.shape[0]

    def sample(self, num_samples: int = 128) -> torch.Tensor:
        n = self.num_control_points
        u = torch.linspace(
            0,
            n,
            steps=num_samples + 1,
            device=self.control_points.device,
            dtype=self.control_points.dtype,
        )[:-1]
        i = torch.floor(u).long()
        t = (u - i).unsqueeze(-1)

        p0 = self.control_points[(i - 1) % n]
        p1 = self.control_points[i % n]
        p2 = self.control_points[(i + 1) % n]
        p3 = self.control_points[(i + 2) % n]

        b0 = (1 - t).pow(3) / 6
        b1 = (3 * t.pow(3) - 6 * t.pow(2) + 4) / 6
        b2 = (-3 * t.pow(3) + 3 * t.pow(2) + 3 * t + 1) / 6
        b3 = t.pow(3) / 6
        return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

    def tangent(self, num_samples: int = 128) -> torch.Tensor:
        points = self.sample(num_samples)
        return normalize(torch.roll(points, -1, dims=0) - torch.roll(points, 1, dims=0))

    def curvature(self, num_samples: int = 128, eps: float = 1e-8) -> torch.Tensor:
        points = self.sample(num_samples)
        first = torch.roll(points, -1, dims=0) - torch.roll(points, 1, dims=0)
        second = torch.roll(points, -1, dims=0) - 2 * points + torch.roll(points, 1, dims=0)
        cross = torch.cross(first, second, dim=-1).norm(dim=-1)
        return cross / first.norm(dim=-1).pow(3).clamp_min(eps)

    def subdivide(self, new_count: int) -> "PeriodicCubicBSpline":
        if new_count <= self.num_control_points:
            raise ValueError("new_count must increase the number of control points.")
        with torch.no_grad():
            control_points = self.sample(new_count).detach()
        return PeriodicCubicBSpline(control_points)


class NeckSpline(nn.Module):
    """Differentiable aneurysm neck loop anchored to an ostium plane."""

    def __init__(
        self,
        control_points: torch.Tensor,
        plane: Plane,
        optimize_plane: bool = False,
    ) -> None:
        super().__init__()
        self.spline = PeriodicCubicBSpline(control_points)
        self.register_buffer("plane_center", plane.center.clone())
        self.register_buffer("plane_tangent", plane.tangent.clone())
        if optimize_plane:
            self.plane_normal = nn.Parameter(plane.normal.clone())
            self.plane_offset = nn.Parameter(plane.offset.clone())
        else:
            self.register_buffer("plane_normal", plane.normal.clone())
            self.register_buffer("plane_offset", plane.offset.clone())

    @property
    def plane(self) -> Plane:
        normal = normalize(self.plane_normal.reshape(1, 3)).squeeze(0)
        return Plane(
            normal=normal,
            offset=self.plane_offset,
            center=self.plane_center,
            tangent=self.plane_tangent,
        )

    def sample(self, num_samples: int = 128) -> torch.Tensor:
        return self.spline.sample(num_samples)

    def normals_in_plane(self, num_samples: int = 128) -> torch.Tensor:
        tangent = self.spline.tangent(num_samples)
        normal = self.plane.normal.expand_as(tangent)
        return normalize(torch.cross(normal, tangent, dim=-1))

    def clinical_measurements(self, num_samples: int = 256) -> dict[str, torch.Tensor]:
        points = self.sample(num_samples)
        width = caliper_width(points)
        normal = self.plane.normal
        tangent = normalize(self.plane.tangent.reshape(1, 3)).squeeze(0)
        angle = torch.arccos(torch.abs(torch.dot(normal, tangent)).clamp(0, 1))
        return {"width": width, "angle_radians": angle}
