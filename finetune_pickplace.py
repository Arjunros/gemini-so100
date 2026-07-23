"""Fine-tune the existing SAC checkpoint on the banana+bowl pick-place scene.

The base checkpoint (already curriculum-tuned for cube-grasp precision)
transfers well to banana-proximate targets (~1.6cm) but was ~11cm off on
bowl-proximate targets, since that xy region + the "place" sub-task weren't
covered by the original cube-only curriculum. SO100PickPlaceEnv's goal
curriculum (see _primary_object_xy in env.py) samples near both the banana
and the bowl, so this fine-tune extends precision to both sub-tasks.

A first attempt at this (default hyperparameters, 150k steps) diverged: the
entropy coefficient ran away (0.03 -> 30+) a few thousand steps in, and both
banana and bowl reach precision ended up *worse* than the untouched base
checkpoint. Likely cause: bowl targets started out with much larger errors
than anything seen during the original cube fine-tune, and the resulting
large/volatile TD errors destabilized SAC's automatic entropy tuning. This
version uses a lower learning rate and stops automatically -- both on
eval-reward plateauing and directly on entropy-coefficient runaway, the
earliest signal of this specific failure mode -- rather than blindly running
the full step budget.

Usage:
    python finetune_pickplace.py --steps 150000
"""

import argparse

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from geminibot.env import make_pickplace_env


class StopOnEntropyRunaway(BaseCallback):
    """Aborts training if SAC's auto-tuned entropy coefficient exceeds a
    threshold well above its normal operating range (~0.01-0.2 in every
    successful run this project has done) -- the earliest, most direct
    signal of the divergence this script exists to catch."""

    def __init__(self, threshold: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold

    def _on_step(self) -> bool:
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        if log_ent_coef is not None:
            val = float(log_ent_coef.detach().exp().cpu().item())
            if val > self.threshold:
                print(f"\n[ABORT] ent_coef={val:.3f} exceeded threshold={self.threshold} "
                      f"at step {self.num_timesteps} -- stopping before it diverges further.")
                return False
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--base", type=str, default="runs/best_model.zip")
    ap.add_argument("--out", type=str, default="runs_pickplace")
    ap.add_argument("--lr", type=float, default=1e-4, help="lower than the original 3e-4 for stability")
    args = ap.parse_args()

    env = Monitor(make_pickplace_env())
    eval_env = Monitor(make_pickplace_env())

    model = SAC.load(args.base, env=env)
    model.tensorboard_log = args.out
    # SB3 resets every optimizer's LR from model.lr_schedule before each
    # train() call (see BaseAlgorithm._update_learning_rate) -- setting
    # model.learning_rate or the optimizers' param_groups directly gets
    # silently overwritten on the very first gradient step. lr_schedule is
    # what actually has to change.
    model.learning_rate = args.lr
    model.lr_schedule = get_schedule_fn(args.lr)

    stop_on_plateau = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=3, verbose=1)
    cb = EvalCallback(eval_env, best_model_save_path=args.out,
                      eval_freq=10_000, n_eval_episodes=10,
                      callback_after_eval=stop_on_plateau)
    entropy_guard = StopOnEntropyRunaway(threshold=1.0)

    model.learn(total_timesteps=args.steps, callback=[cb, entropy_guard],
                reset_num_timesteps=False, progress_bar=True)
    model.save(f"{args.out}/sac_pickplace_final")
    print(f"Saved to {args.out}/sac_pickplace_final.zip")


if __name__ == "__main__":
    main()
