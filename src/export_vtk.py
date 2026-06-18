"""
Roll out a trained policy and export the per-step state as VTK files that
ParaView can open as a time series.

Each step writes a .vti (VTK ImageData) file containing three scalar arrays:
    - field   : the scalar simulation field
    - error   : the per-cell interpolation error
    - refined : 1 where the agent chose to refine, 0 otherwise

In ParaView, open the whole `paraview_output/amr_*.vti` group as a time series,
then colour by "field" and add a Threshold filter on "refined" to see exactly
where the agent spent its refinement budget as the simulation evolves.

Usage:
    python src/export_vtk.py --model models/ppo_amr.zip --steps 30
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from amr_env import AMREnv


def write_vti(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Write a 2D ImageData VTK file in the legacy-friendly XML .vti format.

    Implemented by hand so this script has no hard dependency on the `vtk`
    Python package; ParaView reads the resulting files directly.
    """
    g = next(iter(arrays.values())).shape[0]
    n = g * g

    point_data = []
    for name, arr in arrays.items():
        flat = arr.flatten(order="C")
        values = " ".join(f"{v:.6f}" for v in flat)
        point_data.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'format="ascii">\n          {values}\n        </DataArray>'
        )
    point_data_str = "\n".join(point_data)

    xml = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {g-1} 0 {g-1} 0 0"
             Origin="0 0 0" Spacing="1 1 1">
    <Piece Extent="0 {g-1} 0 {g-1} 0 0">
      <PointData Scalars="field">
{point_data_str}
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
"""
    with open(path, "w") as f:
        f.write(xml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_amr.zip")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--outdir", default="paraview_output")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of the trained model.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env = AMREnv(grid_size=args.grid_size, refine_budget=args.budget, seed=42)
    obs, _ = env.reset(seed=42)

    model = None
    if not args.random:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    for t in range(args.steps):
        state = env.export_state()
        write_vti(os.path.join(args.outdir, f"amr_{t:03d}.vti"), state)

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    print(f"Wrote {t + 1} VTK frames to {args.outdir}/")
    print("Open paraview_output/amr_*.vti in ParaView as a time series.")


if __name__ == "__main__":
    main()
