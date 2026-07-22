"""Fine-tune the existing SAC checkpoint on the banana+bowl pick-place scene.

The base checkpoint (already curriculum-tuned for cube-grasp precision)
transfers well to banana-proximate targets (~1.6cm) but was ~11cm off on
bowl-proximate targets, since that xy region + the "place" sub-task weren't
covered by the original cube-only curriculum. SO100PickPlaceEnv's goal
curriculum (see _primary_object_xy in env.py) samples near both the banana
and the bowl, so this fine-tune extends precision to both sub-tasks.

Usage:
    python finetune_pickplace.py --steps 150000
"""

import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from geminibot.env import make_pickplace_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--base", type=str, default="runs/best_model.zip")
    ap.add_argument("--out", type=str, default="runs_pickplace")
    args = ap.parse_args()

    env = Monitor(make_pickplace_env())
    eval_env = Monitor(make_pickplace_env())

    model = SAC.load(args.base, env=env)
    model.tensorboard_log = args.out

    cb = EvalCallback(eval_env, best_model_save_path=args.out,
                      eval_freq=10_000, n_eval_episodes=10)
    model.learn(total_timesteps=args.steps, callback=cb,
                reset_num_timesteps=False, progress_bar=True)
    model.save(f"{args.out}/sac_pickplace_final")
    print(f"Saved to {args.out}/sac_pickplace_final.zip")


if __name__ == "__main__":
    main()
