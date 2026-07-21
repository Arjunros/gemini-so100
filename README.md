# gemini-so100: Gemini-as-VLA + RL control for the SO-100/101 arm (MuJoCo)

A minimal, hackable stack where **Gemini acts as the vision-language-action brain**
and a **SAC policy acts as the low-level motor controller** for the SO-ARM100 —
the real, open-hardware robot arm from the LeRobot ecosystem — simulated in
MuJoCo. Everything runs natively on Windows. No robot hardware required; the
model is the same MJCF that hardware owners use, so policies and code transfer.

```
                 every ~1 s                          every 40 ms
 camera frame ─► Gemini 2.5 ─► subgoal {target_xyz, jaw} ─► SAC policy ─► joint targets ─► MuJoCo
                 (semantics)                                (precision)
```

Why the split: VLMs are great at scene understanding and task decomposition but
too slow and imprecise for 25 Hz joint control. RL is the opposite. Gemini
decides *where and when*; the policy handles *how*.

## Quickstart (Windows, PowerShell)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Sanity: open the interactive viewer
python view_env.py

# 2. Test the whole VLA loop OFFLINE (no API key, scripted planner + IK fallback)
python run_vla.py --scripted

# 3. Train the low-level reach policy (~1-2 h on CPU, faster w/ GPU)
python train_sac.py --steps 300000

# 4. The real thing
$env:GOOGLE_API_KEY = "your-key"
python run_vla.py --task "pick up the red cube and lift it" --model runs/best_model.zip
```

Each `run_vla.py` run saves a `rollout.mp4` of what happened.

## Repo layout

```
assets/trs_so_arm100/      SO-ARM100 MJCF + meshes (Apache-2.0, from mujoco_menagerie)
  scene_pick.xml           pick scene: arm + red cube + goal marker + cameras
src/geminibot/env.py       Gymnasium env: goal-conditioned reach/grasp, 25 Hz control
src/geminibot/gemini_vla.py  GeminiPlanner (image+state -> JSON subgoal) + ScriptedPlanner
train_sac.py               SB3 SAC training with eval callback + TensorBoard
run_vla.py                 the full slow/fast control loop, saves video
view_env.py                interactive MuJoCo viewer
```

## Environment details

- **Obs (24):** joint pos (6, incl. jaw), joint vel (6), gripper xyz, gripper
  approach vector, cube xyz, goal xyz
- **Action (5):** delta joint-position targets in [-1, 1] for the 5 arm
  joints only. The jaw is **not** part of the learned action space — it's
  driven directly by `env.set_goal(..., jaw=...)`, both in training (randomized
  per episode/goal-switch so the policy is robust to either state) and at
  deployment (commanded by the Gemini planner). See gotchas below for why.
- **Reward:** −distance(gripper, goal) + 2·success + 0.1·top-down-orientation
  − 0.001·‖a‖²; success radius 2.5 cm
- Physics 500 Hz, control 25 Hz. The goal marker can be moved mid-episode via
  `env.set_goal(xyz, jaw=...)` — that's the hook the Gemini planner drives.

## Gotchas we hit so you don't have to

- **Venv discipline (Windows):** after activating, run `pip -V` and confirm the
  path contains your project's `.venv` BEFORE installing. Otherwise pip mutates
  your global Python and breaks other projects.
- **Gemini model names churn.** `gemini-2.5-flash` is retired for new API users.
  Default here is `gemini-3.1-flash-lite` — `gemini-3.5-flash` looked
  appealing but was returning server-side 503s ("high demand") for this exact
  multimodal+JSON-mode call when we checked; `flash-lite` answered instantly
  and correctly every time. List what YOUR key can use with
  `client.models.list()` and swap the default in `GeminiPlanner.__init__` if
  needed. `gemini-robotics-er-*` models are worth benchmarking as planners.
- **Free tier = 15 requests/min.** The sim runs faster than real time, so an
  unthrottled loop burns the quota in seconds. `run_vla.py` enforces a wall
  clock gap between calls (`--plan-gap`, default 4.5 s).
- **Goal sampler must respect the arm's reach.** Sampling goals from a box that
  exceeds the reach sphere silently caps your success rate (`_clip_to_reach`).
- **Static-goal training breaks under VLM goal switching.** A policy trained on
  fixed per-episode goals developed a "parking spot" attractor and froze when
  the planner moved the goal mid-episode. Fix: mid-episode goal resampling
  during training (`resample_goals=True`) + a stall penalty in the reward.
- **Make the planner verify, not narrate.** Early prompts let Gemini declare
  phases complete without checking state; it would close the jaw in midair and
  report success. The system prompt now forces verification against
  gripper/cube coordinates and defines failure-retry behavior.
- **Don't let the planner's jaw command fight a jointly-trained policy.** The
  original design trained SAC over all 6 actuators (arm + jaw) and then, at
  deployment, force-zeroed the jaw action so the planner's `set_goal(jaw=...)`
  could take over. That silently broke reaching: overriding *any* single
  learned action dim post-hoc (not just the jaw) pushes the closed-loop
  trajectory off-distribution and the policy never converges — verified by
  freezing arbitrary dims and watching the gripper get stuck ~30cm from goal
  instead of ~1cm. Fix: the jaw was removed from the action space entirely
  (5-dim action, arm only); it's always externally driven, so training and
  deployment see the same dynamics.

## Notes & known limits

- The IK fallback in `run_vla.py` reaches targets reliably but only controls
  position, not wrist orientation, so grasps succeed only sometimes. The
  trained policy (with the orientation shaping term) is meant to close that gap
  — and it's the interesting RL problem here.
- Gemini calls cost ~1 image per second of sim time. Use `--scripted` while
  developing; switch to Gemini for the demos.
- Coordinate convention (also stated in the Gemini system prompt): arm base at
  origin, +x toward the camera, +y left, +z up, metres.

## Roadmap to v0.1 (open source)

1. Trained SAC checkpoint + training curves committed to the repo
2. Success-rate benchmark: scripted vs Gemini planner, 50 episodes each
3. Second task (place cube on target) + task registry
4. Domain randomization toggles (cube size/mass/friction, camera jitter)
5. LeRobot-format dataset export of successful rollouts (for imitation learning)

## License / credits

Arm model: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`trs_so_arm100`, Apache-2.0), based on TheRobotStudio's SO-ARM100.
