# RL for Adaptive Mesh Refinement — Project Writeup

## The problem

Scientific simulations (fluid dynamics, heat transfer, combustion) can't afford a
uniformly fine mesh everywhere — it's too expensive. **Adaptive mesh refinement
(AMR)** solves this by refining the mesh only where it matters: near shocks,
fronts, and steep gradients. Traditional AMR decides where to refine using fixed
error or gradient thresholds tuned by hand. I wanted to see whether an agent
could *learn* that refinement policy directly, instead of relying on a fixed
rule.

## What I built

I framed mesh refinement as a sequential decision problem and trained a
reinforcement learning agent to solve it.

- **Environment**: a custom `gymnasium` environment simulating a 2D scalar field
  that evolves by diffusion, so the region worth refining moves over time. At
  each step the agent observes the current field and chooses a small set of grid
  cells to refine, under a fixed total budget.
- **Reward**: the reduction in total interpolation error, minus a small cost per
  cell refined — this rewards the agent for being efficient, not just for
  refining everything.
- **Agent**: PPO (Proximal Policy Optimization) via `stable-baselines3`.
- **Baseline**: I compared the learned policy against random refinement using the
  same budget, so the result is interpretable rather than a number in a vacuum.
- **Visualization**: I export trained-policy rollouts to **ParaView** as a VTK
  time series, with separate data arrays for the field, the per-cell error, and
  the agent's refinement choices. This makes the learned policy directly visible:
  you can watch the refined cells track the moving high-gradient front.

## Key design decisions

- **Per-step refinement cap.** My first version let the agent spend its whole
  budget in one step, which collapsed the task into a trivial one-shot choice.
  Capping refinement to a few cells per step turned it into a genuine sequential
  problem where the agent has to follow the front as it moves.
- **Error-reduction reward with a cost term.** Rewarding raw error reduction
  alone encourages over-refinement; the small per-cell penalty forces the agent
  to prioritise the highest-impact cells.
- **Self-contained simulation.** I used an explicit diffusion step rather than
  wiring in a heavy external solver, so the whole project is fast and fully
  reproducible. The refinement is modelled as a reduction in a per-cell error
  proxy, which keeps the core decision problem intact while staying lightweight.

## What I learned

- How to design a reward function that captures a real engineering trade-off
  (accuracy vs. computational budget) rather than a toy objective.
- Why reward shaping and action constraints matter: small changes to the action
  space (the per-step cap) completely changed whether the task was learnable.
- The value of always having a baseline — "the agent gets reward X" means
  nothing until you can say "X versus Y for random."
- Practical scientific-visualization workflow: writing VTK files by hand and
  driving ParaView to inspect a policy, including threshold filters to isolate
  exactly what the agent did.

## Possible extensions

Swap the toy field for output from a real solver (PETSc, FEniCS) and refine on
the actual residual; replace the flattened-field input with a CNN policy;
implement true multi-resolution re-meshing instead of the error-reduction proxy;
and benchmark PPO against a value-based method like DQN.

## Tech stack

Python · Gymnasium · Stable-Baselines3 (PPO) · NumPy · ParaView / VTK ·
Matplotlib
