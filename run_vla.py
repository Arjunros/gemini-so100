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
from geminibot.env import SO100ReachEnv, SO100PickPlaceEnv
from geminibot.gemini_vla import GeminiPlanner, ScriptedPlanner, ScriptedPickPlacePlanner
from geminibot.geometry import pixel_to_world

# Approximate resting heights used only to back-project vision-grounded
# pixel detections to 3D (see geometry.pixel_to_world's plane_z arg) --
# not used for anything else, so a few mm of error here is harmless.
BANANA_REST_Z = 0.01
BOWL_REST_Z = 0.014


def jacobian_ik_action(env) -> np.ndarray:
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
    ap.add_argument("--scene", choices=["cube", "pickplace"], default="cube",
                    help="cube: fixed-cube reach/grasp/lift. "
                         "pickplace: open-vocabulary banana-in-bowl (vision-grounded, no oracle object coords)")
    ap.add_argument("--task", default=None, help="defaults per --scene if omitted")
    ap.add_argument("--model", default=None, help="path to SAC .zip checkpoint")
    ap.add_argument("--scripted", action="store_true", help="offline planner, no API")
    ap.add_argument("--plan-every", type=int, default=25, help="ctrl steps per planner call")
    ap.add_argument("--plan-gap", type=float, default=4.5,
                    help="min wall-clock seconds between Gemini calls (free tier: 15/min)")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--video", default="rollout.mp4")
    ap.add_argument("--live", action="store_true", help="show live MuJoCo viewer")
    args = ap.parse_args()

    pickplace = args.scene == "pickplace"
    if args.task is None:
        args.task = "put the banana in the bowl" if pickplace else "pick up the red cube and lift it 10 cm"

    if pickplace:
        env = SO100PickPlaceEnv(max_episode_steps=args.max_steps, resample_goals=False)
        planner = ScriptedPickPlacePlanner() if args.scripted else GeminiPlanner()
    else:
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
    # Open-vocab pick-place only: cache vision-grounded positions between
    # planning ticks rather than re-grounding every call (cheaper, and the
    # objects don't move on their own) -- refreshed whenever the planner
    # re-enters "hover" (a fresh pick attempt, e.g. after a failed grasp),
    # since the object may have moved since the last grounding.
    grounded_pick_xyz = grounded_place_xyz = None
    last_phase = None
    cam_id = env.model.camera("vla_cam").id if pickplace else None

    try:
        for t in range(args.max_steps):
            if t % args.plan_every == 0:
                if not args.scripted:
                    gap = time.time() - last_plan
                    if gap < args.plan_gap:
                        time.sleep(args.plan_gap - gap)
                    last_plan = time.time()
                frame = env.render("vla_cam")

                if pickplace:
                    if args.scripted:
                        sg = planner.plan(frame, args.task, info["gripper_pos"],
                                          info["banana_pos"], info["bowl_pos"])
                    else:
                        if grounded_pick_xyz is None or last_phase == "hover":
                            det = planner.locate_task_objects(frame, args.task)
                            h, w = frame.shape[:2]
                            grounded_pick_xyz = pixel_to_world(
                                env.model, env.data, cam_id, *det["pick"], w, h, plane_z=BANANA_REST_Z)
                            grounded_place_xyz = pixel_to_world(
                                env.model, env.data, cam_id, *det["place"], w, h, plane_z=BOWL_REST_Z)
                        sg = planner.plan_pickplace(frame, args.task, info["gripper_pos"],
                                                    grounded_pick_xyz, grounded_place_xyz)
                    last_phase = sg.phase
                    print(f"[t={t:4d}] phase={sg.phase:12s} target={np.round(sg.target_xyz,3)} "
                          f"grip={np.round(info['gripper_pos'],3)} jaw={sg.jaw:.1f}  {sg.reason}")
                else:
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
            if pickplace:
                print(f"Banana placed in bowl: {info['placed']}")
            else:
                print(f"Cube height at end: {info['cube_pos'][2]:.3f} m "
                      f"({'LIFTED!' if info['cube_pos'][2] > 0.08 else 'not lifted'})")
            print(f"Video saved to {args.video} ({len(frames)} frames)")
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
