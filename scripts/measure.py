from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True)
    args = parser.parse_args()

    result = torch.load(args.fit, map_location="cpu")
    measurements = result["measurements"]
    width = float(measurements["width"])
    angle_deg = float(measurements["angle_radians"] * 180.0 / torch.pi)
    print(f"neck_width={width:.4f}")
    print(f"parent_vessel_angle_deg={angle_deg:.4f}")


if __name__ == "__main__":
    main()
