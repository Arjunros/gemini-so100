"""Fine-tune the existing SAC checkpoint with the cube-proximate goal
curriculum (see geminibot.env._sample_goal), to fix the ~3.5cm systematic
positioning error the base checkpoint had at low, grasp-relevant targets.

Usage:
    python finetune_grasp.py --steps 150000
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
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--base", type=str, default="runs/best_model.zip")
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    env = Monitor(make_env())
    eval_env = Monitor(make_env())

    model = SAC.load(args.base, env=env)

    cb = EvalCallback(eval_env, best_model_save_path=args.out,
                      eval_freq=10_000, n_eval_episodes=10)
    model.learn(total_timesteps=args.steps, callback=cb,
                reset_num_timesteps=False, progress_bar=True)
    model.save(f"{args.out}/sac_reach_final")
    print(f"Saved to {args.out}/sac_reach_final.zip")


if __name__ == "__main__":
    main()
