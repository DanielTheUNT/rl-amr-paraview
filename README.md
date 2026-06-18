# Reinforcement Learning for Adaptive Mesh Refinement (with ParaView)

A reinforcement learning agent that learns *where* to refine a computational
mesh. The agent watches a 2D scalar field evolve (a diffusing concentration /
heat blob) and, under a limited refinement budget, decides which grid cells to
refine at each timestep. The goal is to minimise interpolation error — which
means learning to spend the budget on the regions where the field changes
fastest, exactly the heuristic human-written adaptive mesh refinement (AMR)
solvers approximate.

Trained policy rollouts are exported to **ParaView** as a `.vti` time series so
you can visually inspect where the agent chose to refine as the simulation
evolves.

## Why this is interesting

Adaptive mesh refinement is a core technique in computational fluid dynamics and
scientific simulation: you can't afford a uniformly fine mesh everywhere, so you
refine only where it matters. Traditional AMR uses fixed gradient/error
thresholds. Framing refinement as a sequential decision problem lets an RL agent
learn a refinement *policy* directly from the reward signal (error reduction per
unit budget). This project is a compact, reproducible demonstration of that idea.

## Method

- **Environment** (`src/amr_env.py`): a self-contained `gymnasium` environment.
  The field evolves by an explicit diffusion step, so the optimal place to
  refine moves over time and the agent can't just memorise one location.
  - *Observation*: the current coarse scalar field (flattened).
  - *Action*: a binary mask over grid cells (refine / don't refine), capped at a
    few cells per step so refinement is a genuine sequential decision.
  - *Reward*: reduction in total interpolation error, minus a small per-cell
    cost — this is what pushes the agent toward an efficient policy.
- **Algorithm**: PPO from `stable-baselines3` (`src/train.py`).
- **Baseline**: training reports the learned policy's error reduction against a
  random-refinement baseline so the result is interpretable.
- **Visualization**: `src/export_vtk.py` rolls out the policy and writes VTK
  ImageData files for ParaView.

## Project structure
