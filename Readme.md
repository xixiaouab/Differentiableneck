# DifferentiableNeck / NeckSpline

Official PyTorch scaffold for **Differentiable Centerline-Aware Framework for Aneurysm Neck Delineation in Volumetric Angiography** (npj Digital Medicine, 2026).

The paper's method is **NeckSpline**: a geometry-first framework that models an aneurysm neck as a continuous, closed, periodic cubic B-spline anchored by the parent-vessel centerline. This repository exposes the core differentiable pieces: centerline-guided ostium-plane fitting, B-spline control-point optimization, image-evidence losses, tightness and scale priors, topology-preserving penalties, and clinical measurement extraction.

## Method Overview

```text
CTA / TOF-MRA volume
        |
vessel probability + boundary likelihood + centerline graph
        |
weighted PCA ostium-plane estimate near bifurcation
        |
periodic cubic B-spline initialized on the plane
        |
boundary + gradient + profile evidence
        |
tightness + scale + self-intersection + EC topology proxy
        |
neck loop, width, parent-vessel angle
```

## Core Components

- **Centerline-guided plane**: weighted PCA over local wall-transition voxels, regularized toward the parent-vessel tangent.
- **Periodic cubic B-spline**: closed-loop control points with C2 continuity at wrap-around.
- **Differentiable image evidence**: trilinear sampling for boundary likelihood, gradient alignment, and inside/outside profile terms.
- **Geometry priors**: curvature tightness, circumference/area scale control, plane anchoring, and centerline coupling.
- **Topology constraints**: self-intersection repulsion plus a differentiable EC-style proxy for a single closed loop.
- **Measurements**: direct caliper width and parent-vessel angle from the optimized spline.

## Repository Layout

```text
configs/neckspline_base.yaml          Minimal NeckSpline configuration
scripts/fit_neckspline.py             Coarse-to-fine fitting entrypoint
scripts/measure.py                    Width and angle extraction
src/differentiableneck/geometry.py    Plane fitting, frames, sampling, metrics
src/differentiableneck/models/spline.py Periodic cubic B-spline and NeckSpline
src/differentiableneck/losses.py      Image, geometry, and topology losses
src/differentiableneck/engine/optimize.py Coarse-to-fine AdamW optimizer
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/fit_neckspline.py --config configs/neckspline_base.yaml --output outputs/case.pt
python scripts/measure.py --fit outputs/case.pt
```

This release is a research scaffold. It does not include private preprocessing code, trained vessel/boundary predictors, or clinical datasets.

## Citation

```bibtex
@article{liu2026differentiable,
  title={Differentiable centerline-aware framework for aneurysm neck delineation in volumetric angiography},
  author={Liu, Xinyan and Zhou, Jian and Zhang, Hongyue and Tao, Bilin and Xu, Shuogui},
  journal={npj Digital Medicine},
  year={2026},
  doi={10.1038/s41746-026-02613-6}
}
```
