"""
Plot the training reward curve from the Monitor CSV produced during training.

Usage:
    python src/plot_results.py
"""

import csv
import os

import matplotlib.pyplot as plt


def main(log_path: str = "logs/monitor.csv", out_path: str = "logs/reward_curve.png"):
    if not os.path.exists(log_path):
        raise SystemExit(f"No log found at {log_path}. Run training first.")

    rewards = []
    with open(log_path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "r":
                continue
            try:
                rewards.append(float(row[0]))
            except ValueError:
                continue

    # Simple moving average to smooth the curve.
    window = max(1, len(rewards) // 50)
    smooth = [
        sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    plt.plot(smooth, label=f"moving avg (w={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (error reduction)")
    plt.title("PPO learning curve — AMR refinement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
