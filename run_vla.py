"""The full Gemini-VLA + RL control loop.

Slow loop (~1 per 4.5s wall clock, free-tier safe):  camera + state -> Gemini -> subgoal
Fast loop (25 Hz sim):  SAC policy (or IK fallback) chases the current subgoal

Usage:
    python run_vla.py --scripted                          # offline, no API key
    python run_vla.py --task "pick up the red cube" --model runs\\sac_reach_v2.zip --live
"""

import argparse
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from geminibot.env import SO100ReachEnv
from geminibot.gemini_vla import GeminiPlanner, ScriptedPlanner


def jacobian_ik_action(env: SO100ReachEnv) -> np.ndarray:
    """Fallback controller: damped-least-squares differential IK, target
    tethered to measured joint angles so the position servo can't run away."""
    import mujoco

    err = env.goal - env.gripper_pos()
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.model.site("gripper").id)
    J = jacp[:, :5]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), 1.5 * err)
    q = env.data.qpos[:5]
    desired = np.clip(q + dq, q - 0.25, q + 0.25)
    delta = desired - env.data.ctrl[:5]
    span = (env.ctrl_high[:5] - env.ctrl_low[:5]) * env.action_scale
    return np.clip(delta / span, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick up the red cube and lift it 10 cm")
    ap.add_argument("--model", default=None, help="path to SAC .zip checkpoint")
    ap.add_argument("--scripted", action="store_true", help="offline planner, no API")
    ap.add_argument("--plan-every", type=int, default=25, help="ctrl steps per planner call")
    ap.add_argument("--plan-gap", type=float, default=4.5,
                    help="min wall-clock seconds between Gemini calls (free tier: 15/min)")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--video", default="rollout.mp4")
    ap.add_argument("--live", action="store_true", help="show live MuJoCo viewer")
    args = ap.parse_args()

    env = SO100ReachEnv(max_episode_steps=args.max_steps, resample_goals=False)
    planner = ScriptedPlanner() if args.scripted else GeminiPlanner()

    policy = None
    if args.model:
        from stable_baselines3 import SAC
        policy = SAC.load(args.model)

    obs, info = env.reset()

    viewer = None
    if args.live:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    frames = []
    last_plan = 0.0

    try:
        for t in range(args.max_steps):
            if t % args.plan_every == 0:
                if not args.scripted:
                    gap = time.time() - last_plan
                    if gap < args.plan_gap:
                        time.sleep(args.plan_gap - gap)
                    last_plan = time.time()
                frame = env.render("vla_cam")
                sg = planner.plan(frame, args.task, info["gripper_pos"], info["cube_pos"])
                print(f"[t={t:4d}] phase={sg.phase:10s} target={np.round(sg.target_xyz,3)} "
                      f"grip={np.round(info['gripper_pos'],3)} cube={np.round(info['cube_pos'],3)} "
                      f"jaw={sg.jaw:.1f}  {sg.reason}")
                env.set_goal(sg.target_xyz, jaw=sg.jaw)
                obs = env._obs()
                if sg.done:
                    print("Planner reports task complete.")
                    break

            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = jacobian_ik_action(env)
            # jaw is commanded by the planner via env.set_goal(), never learned

            obs, r, term, trunc, info = env.step(action)
            frames.append(env.render("vla_cam"))
            if viewer is not None:
                viewer.sync()
            if trunc:
                break
    except KeyboardInterrupt:
        print("\nInterrupted — saving video of what we have.")
    finally:
        if frames:
            imageio.mimsave(args.video, frames, fps=25)
            print(f"Cube height at end: {info['cube_pos'][2]:.3f} m "
                  f"({'LIFTED!' if info['cube_pos'][2] > 0.08 else 'not lifted'})")
            print(f"Video saved to {args.video} ({len(frames)} frames)")
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
