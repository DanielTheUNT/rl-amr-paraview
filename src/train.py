"""
Train a PPO agent on the AMR environment.

Usage:
    python src/train.py --timesteps 200000 --grid-size 16

The trained policy is saved to models/ppo_amr.zip and a CSV of episode rewards
is written to logs/ for plotting. A random-action baseline is evaluated at the
end so you have a number to compare the learned policy against.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from amr_env import AMREnv


def evaluate(model, env, n_episodes: int = 20, random_policy: bool = False):
    """Return mean episode reward and mean final interpolation error."""
    rewards, errors = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        total_r = 0.0
        info = {}
        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        errors.append(info.get("total_error", np.nan))
    return float(np.mean(rewards)), float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Imported here so the env file stays usable without SB3 installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    def make_env():
        return Monitor(
            AMREnv(grid_size=args.grid_size, refine_budget=args.budget,
                   seed=args.seed),
            filename="logs/monitor.csv",
        )

    env = make_env()
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log="logs/tb",
    )

    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    model.save("models/ppo_amr")
    print("Saved model to models/ppo_amr.zip")

    # Compare learned policy vs random baseline.
    eval_env = AMREnv(grid_size=args.grid_size, refine_budget=args.budget,
                      seed=args.seed + 1)
    learned_r, learned_e = evaluate(model, eval_env, random_policy=False)
    random_r, random_e = evaluate(model, eval_env, random_policy=True)

    print("\n=== Evaluation (20 episodes) ===")
    print(f"Learned policy : reward {learned_r:8.3f} | final error {learned_e:8.3f}")
    print(f"Random policy  : reward {random_r:8.3f} | final error {random_e:8.3f}")
    print(f"Error reduction vs random: "
          f"{100 * (random_e - learned_e) / random_e:5.1f}%")


if __name__ == "__main__":
    main()
