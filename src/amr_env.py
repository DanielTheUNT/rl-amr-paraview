"""
Adaptive Mesh Refinement (AMR) environment for reinforcement learning.

The agent observes a coarse 2D scalar field (a diffusing "heat" / concentration
blob) and must decide which cells of a grid to refine. Refining a cell costs
budget but reduces the interpolation error in that region. The agent learns to
spend a limited refinement budget on the cells where the field changes fastest
(high gradient), which is exactly the heuristic human-written AMR solvers try to
approximate.

This is a self-contained Gymnasium environment with no external simulator
dependency, so it runs anywhere. The field dynamics are a simple explicit
diffusion step, which keeps things reproducible and fast.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AMREnv(gym.Env):
    """Gymnasium environment for learning where to refine a mesh.

    Observation: the current coarse scalar field, flattened (grid_size**2,).
    Action:      MultiBinary mask over grid cells (1 = refine this cell).
    Reward:      reduction in total interpolation error minus a budget penalty.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size: int = 16, refine_budget: int = 40,
                 max_steps: int = 30, per_step_cap: int = 4,
                 seed: int | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.refine_budget = refine_budget
        self.max_steps = max_steps
        # Cap cells refined per step so the agent must choose *where* to refine
        # at each timestep rather than spending the whole budget at once. This
        # is what makes the problem a sequential decision task worth learning.
        self.per_step_cap = per_step_cap

        n_cells = grid_size * grid_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_cells,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(n_cells)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.field = None
        self.refined = None  # boolean mask of already-refined cells

    # ------------------------------------------------------------------ #
    # Field generation and dynamics
    # ------------------------------------------------------------------ #
    def _make_field(self) -> np.ndarray:
        """Create a smooth field with one or two Gaussian blobs."""
        g = self.grid_size
        xs, ys = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g))
        field = np.zeros((g, g), dtype=np.float32)
        n_blobs = self._rng.integers(1, 3)
        for _ in range(n_blobs):
            cx, cy = self._rng.uniform(0.2, 0.8, size=2)
            sigma = self._rng.uniform(0.05, 0.15)
            field += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        field /= field.max() + 1e-8
        return field.astype(np.float32)

    def _diffuse(self, field: np.ndarray) -> np.ndarray:
        """One explicit diffusion step (5-point Laplacian stencil)."""
        lap = (
            -4 * field
            + np.roll(field, 1, 0) + np.roll(field, -1, 0)
            + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        )
        return np.clip(field + 0.15 * lap, 0.0, 1.0)

    def _interp_error(self) -> np.ndarray:
        """Per-cell interpolation error proxy = local gradient magnitude.

        Refined cells have their error reduced by 80%, modelling the fact that
        a finer mesh resolves the gradient better.
        """
        gx, gy = np.gradient(self.field)
        err = np.sqrt(gx ** 2 + gy ** 2)
        err = err * np.where(self.refined, 0.2, 1.0)
        return err

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.field = self._make_field()
        self.refined = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._step_count = 0
        return self.field.flatten(), {}

    def step(self, action):
        action = np.asarray(action).reshape(self.grid_size, self.grid_size).astype(bool)

        err_before = self._interp_error().sum()

        # Apply refinement, respecting both the per-step cap and the overall
        # remaining budget. When the agent requests more than allowed, keep the
        # highest-gradient requested cells.
        remaining = self.refine_budget - self.refined.sum()
        allowed = int(min(self.per_step_cap, max(remaining, 0)))
        new_cells = action & (~self.refined)
        if new_cells.sum() > allowed:
            err_now = self._interp_error()
            idx = np.argsort(err_now[new_cells])[::-1]
            coords = np.argwhere(new_cells)[idx][:allowed]
            mask = np.zeros_like(new_cells)
            for r, c in coords:
                mask[r, c] = True
            new_cells = mask
        self.refined |= new_cells

        err_after = self._interp_error().sum()

        # Reward: error reduction, minus a small cost per cell refined.
        reward = float(err_before - err_after) - 0.01 * float(new_cells.sum())

        # Advance the simulation so the optimal refinement region moves.
        self.field = self._diffuse(self.field)
        self._step_count += 1

        terminated = self.refined.sum() >= self.refine_budget
        truncated = self._step_count >= self.max_steps
        info = {
            "total_error": float(self._interp_error().sum()),
            "cells_refined": int(self.refined.sum()),
        }
        return self.field.flatten(), reward, terminated, truncated, info

    def render(self):
        """Return the field as an RGB array for quick inspection."""
        f = (self.field * 255).astype(np.uint8)
        rgb = np.stack([f, f, f], axis=-1)
        # Mark refined cells in red.
        rgb[self.refined] = [200, 40, 40]
        return rgb

    # ------------------------------------------------------------------ #
    # Export for ParaView
    # ------------------------------------------------------------------ #
    def export_state(self):
        """Return arrays needed to write a VTK file for ParaView."""
        return {
            "field": self.field.copy(),
            "refined": self.refined.astype(np.float32),
            "error": self._interp_error(),
        }
