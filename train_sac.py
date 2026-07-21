"""Train the low-level goal-reaching policy with SAC (stable-baselines3).

Usage (Windows PowerShell):
    python train_sac.py --steps 300000

~300k steps is enough for reliable reaching on CPU (about 1-2 hours).
Checkpoints and TensorBoard logs land in ./runs.
"""

import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from geminibot.env import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    env = Monitor(make_env())
    eval_env = Monitor(make_env())

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.98,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=args.out,
        verbose=1,
    )

    cb = EvalCallback(eval_env, best_model_save_path=args.out,
                      eval_freq=10_000, n_eval_episodes=10)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=True)
    model.save(f"{args.out}/sac_reach_final")
    print(f"Saved to {args.out}/sac_reach_final.zip")


if __name__ == "__main__":
    main()
