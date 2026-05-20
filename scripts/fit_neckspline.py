from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from differentiableneck.engine import StageConfig, fit_neckspline
from differentiableneck.geometry import weighted_pca_plane


def load_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neckspline_base.yaml")
    parser.add_argument("--output", default="outputs/neckspline.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data_cfg = cfg["data"]
    image_volume = load_tensor(data_cfg["volume_path"])
    boundary_volume = load_tensor(data_cfg["boundary_path"])
    gradient_volume = load_tensor(data_cfg["gradient_path"])
    centerline_points = load_tensor(data_cfg["centerline_points_path"])
    centerline_weights = load_tensor(data_cfg["centerline_weights_path"])
    tangent = load_tensor(data_cfg["tangent_path"])

    plane = weighted_pca_plane(
        centerline_points,
        centerline_weights,
        tangent,
        tangent_weight=cfg["plane"]["tangent_weight"],
    )
    stages = [
        StageConfig(
            control_points=stage["control_points"],
            iterations=stage["iterations"],
            profile_radius=stage["profile_radius_mm"],
        )
        for stage in cfg["spline"]["stages"]
    ]
    model, history = fit_neckspline(
        plane=plane,
        image_volume=image_volume,
        boundary_volume=boundary_volume,
        gradient_volume=gradient_volume,
        parent_radius=cfg["plane"]["initial_parent_radius_mm"],
        weights=cfg["loss"],
        stages=stages,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        num_samples=cfg["spline"]["num_samples"],
        reference_curvature=cfg["spline"]["reference_curvature"],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "control_points": model.spline.control_points.detach().cpu(),
            "plane_normal": model.plane.normal.detach().cpu(),
            "plane_offset": model.plane.offset.detach().cpu(),
            "measurements": {
                key: value.detach().cpu()
                for key, value in model.clinical_measurements().items()
            },
            "history": history,
        },
        output,
    )
    print(f"Saved NeckSpline fit to {output}")


if __name__ == "__main__":
    main()
