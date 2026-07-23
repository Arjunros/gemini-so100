# gemini-so100: Gemini-as-VLA + RL control for the SO-100/101 arm (MuJoCo)

A minimal, hackable stack where **Gemini acts as the vision-language-action brain**
and a **SAC policy acts as the low-level motor controller** for the SO-ARM100 —
the real, open-hardware robot arm from the LeRobot ecosystem — simulated in
MuJoCo. Everything runs natively on Windows. No robot hardware required; the
model is the same MJCF that hardware owners use, so policies and code transfer.

Two tasks live here:
- **`--scene cube`** (default): reach/grasp/lift a fixed red cube. Gemini is
  handed oracle object coordinates each tick (it verifies against them, but
  doesn't have to *find* the cube itself).
- **`--scene pickplace`**: fully open-vocabulary pick-and-place — "put the
  banana in the bowl." Gemini locates both objects from the camera image
  **alone** (no coordinates given, no hardcoded object names), and a real
  physical containment check (not a proximity heuristic) decides success.

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

# 5. Open-vocabulary pick-and-place: no coordinates or object names given to Gemini
python run_vla.py --scene pickplace --scripted --model runs/best_model.zip --live   # offline mechanics check
python run_vla.py --scene pickplace --model runs/best_model.zip                     # the real thing
```

Each `run_vla.py` run saves a `rollout.mp4` of what happened.

## Repo layout

```
assets/trs_so_arm100/      SO-ARM100 MJCF + meshes (Apache-2.0, from mujoco_menagerie)
  scene_pick.xml           cube scene: arm + red cube + goal marker + cameras
  scene_pickplace.xml      pick-place scene: arm + YCB banana + bowl + cameras
  assets/ycb/               banana/bowl meshes (see credits)
src/geminibot/env.py       Gymnasium envs: SO100ReachEnv (cube) + SO100PickPlaceEnv (banana/bowl),
                            sharing a _SO100Base with the reach policy's action/physics mechanics
src/geminibot/geometry.py  pixel<->world camera projection, for vision grounding
src/geminibot/gemini_vla.py  GeminiPlanner (cube: plan(); pick-place: locate_task_objects() +
                            plan_pickplace()) + ScriptedPlanner / ScriptedPickPlacePlanner (oracle, offline)
train_sac.py                SB3 SAC training with eval callback + TensorBoard
finetune_grasp.py            fine-tunes an existing checkpoint (cube-grasp precision curriculum)
finetune_pickplace.py        fine-tunes an existing checkpoint (banana/bowl curriculum) --
                            has learned some scars, see "Gotchas" below before running it
run_vla.py                   the full slow/fast control loop, saves video, --scene {cube,pickplace}
view_env.py                  interactive MuJoCo viewer
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

### Pick-place scene (`SO100PickPlaceEnv`)

Same action space, same 24-dim obs shape as the cube task (the object slot
holds the banana's position) — the underlying reach skill is genuinely
object-agnostic, so the *same* checkpoint drives both tasks without
retraining the interface. What's new:

- **Real objects, not primitives.** YCB banana (~0.45x scale, ~8.9cm long,
  20g) and bowl (~0.5x scale, ~8cm dia, 80g) — real-world YCB scale (banana
  ~20cm, bowl ~16cm dia) is too big for this arm's ~30cm reach and a jaw
  sized for a 2.5cm cube.
- **Real physical containment**, not a proximity heuristic: `is_placed()`
  checks the banana is within the bowl's horizontal radius *and* within its
  cavity height band, both derived from the actual scaled mesh dimensions.
- **Fully open-vocabulary grounding**: `GeminiPlanner.locate_task_objects()`
  identifies which object to pick and which is the destination from the
  image + instruction alone — no coordinates, no hardcoded object names.
  `geometry.pixel_to_world()` back-projects Gemini's bounding-box center to
  3D via the camera's actual pose and a known table-height plane. Measured
  accuracy: ~2.5mm (banana), ~6mm (bowl) against sim ground truth — not the
  bottleneck in this pipeline.

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
- **Look at your rendered frames, not just the numbers.** `vla_cam`'s
  `xyaxes` had a sign error that pointed the camera up and away from the
  workspace — its forward vector was `(0.8, 0, 0.6)` from `(0.55, 0, 0.45)`,
  i.e. into the sky. Every rollout video for most of this project's
  development was, when actually opened frame-by-frame, blank sky — this
  went unnoticed because verification relied entirely on numeric telemetry
  (gripper/object distance), never on looking at an actual frame. The
  physics and control logic were fine throughout; only the camera (and
  therefore what Gemini actually saw) was broken. Fixed by negating the
  local x-axis. Lesson: numeric telemetry can't catch a bug that's purely
  visual — this one would have silently broken vision grounding entirely.
- **SAC fine-tuning can diverge quietly, and physics bugs can cause it.** A
  fine-tune for the pick-place curriculum blew up (`ent_coef` ran from
  ~0.03 to 30+, actor/critic losses exploded) and produced a checkpoint
  *worse* than doing nothing. `MUJOCO_LOG.TXT` (easy to miss — it appears
  wherever the process's cwd is, not in a log directory) showed the actual
  cause: NaN/Inf accelerations from banana/bowl spawning close enough to
  interpenetrate (the separation check only compared center-to-center
  distance against a fixed 8cm threshold, not the objects' actual extents
  — banana's worst-case horizontal half-extent is ~4.7cm, bowl's radius
  ~4cm). One bad contact resolution corrupted the replay buffer and the
  entropy auto-tuner never recovered. Fixed the spawn separation (0.13m)
  and added a training-time safeguard (`finetune_pickplace.py`'s
  `StopOnEntropyRunaway`) that aborts if `ent_coef` exceeds 1.0, rather
  than silently burning the full step budget on a diverged policy.
- **`model.learning_rate = x` doesn't change SB3's effective learning
  rate.** `BaseAlgorithm._update_learning_rate` resets every optimizer's LR
  from `model.lr_schedule` before each `train()` call — you have to replace
  `lr_schedule` itself (`from stable_baselines3.common.utils import
  get_schedule_fn; model.lr_schedule = get_schedule_fn(new_lr)`), or your
  override gets silently overwritten on the first gradient step.
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
- **Grasp success rate is genuinely low (~10% in a 30-episode scripted-planner
  eval), and it's not primarily a positioning problem.** Diagnosed by
  measuring xy alignment error at the exact moment the planner closes the
  jaw: the base checkpoint had a systematic ~3.5cm error at low, cube-
  proximate targets (curriculum goal sampling toward exactly those targets,
  see `_sample_goal` in `env.py`, fixed this to ~0.6cm). But fixing alignment
  alone barely moved the success rate (5%→10%) — most well-aligned,
  correct-height grasp attempts still don't hold the cube. That points to the
  contact/orientation dynamics at the jaw-closing instant (only loosely
  shaped by the 0.1-weight top-down-orientation reward term), not xy/z
  position. A next step worth trying: an explicit orientation-at-contact
  reward term, or approach-vector verification in the grasp phase itself.
- Also found but *not* fixed (deliberately, to avoid retraining against a
  moving target twice): `WORKSPACE_LOW`'s z-floor (0.02) silently clips the
  intended 0.015 grasp height (the cube's centroid) upward by 0.5cm. The
  current checkpoint was curriculum-tuned against the *clipped* height, so
  removing the clip without a matching retrain measurably hurt success
  (10%→3%). If you retrain, fix both together: lower `WORKSPACE_LOW[2]` to
  ~0.01 and use 0.015 in `GRASP_HEIGHTS`.
- Gemini calls cost ~1 image per second of sim time. Use `--scripted` while
  developing; switch to Gemini for the demos.
- Coordinate convention (also stated in the Gemini system prompt): arm base at
  origin, +x toward the camera, +y left, +z up, metres.

### Pick-place task: honest results

Full end-to-end placement success (banana genuinely resting in the bowl,
`ScriptedPickPlacePlanner` + trained policy, 20-episode eval): **0/20**.
This is a harder, *compound* version of the cube's already-imperfect grasp
problem — it needs a successful grasp (banana is a harder shape to grasp
reliably than a cube) **and** a precise, separate placement afterward.
Multiplying two already-marginal probabilities together plausibly lands
well under 5% true success; 0/20 isn't strong evidence of "broken" so much
as "hard, and not measurably fixed yet." What was tried:

- Zero-shot transfer of the cube-tuned checkpoint reached banana-height
  targets fine (1.6cm) but was ~11-12cm off on bowl-proximate targets
  (that xy region + the two-object curriculum weren't in the original
  training distribution).
- A curriculum-extended fine-tune (targets biased toward both the banana
  and bowl regions, see `_primary_object_xy` in `env.py`) genuinely
  improved aggregate reach metrics (success rate 0%→40% peak, reward from
  deeply negative to +136) but didn't move the specific bowl-precision
  probe or the full-task success rate in a 20-episode sample — the
  improvement, whatever it was, didn't concentrate where this eval looks.
- **The planner's own success claim isn't verifiable the way the cube
  task's is.** In a live run, Gemini declared `done=true` ("the banana has
  been successfully placed") when the ground-truth containment check said
  otherwise. The cube task's planner verifies against a continuously
  fed oracle z-coordinate each tick; the pick-place planner only has
  vision-grounded positions from whenever it last looked, so "did this
  actually work" ultimately falls back to visual judgment, which turned
  out to be overconfident here. A fix worth trying: re-run
  `locate_task_objects` after release and have the planner compare the
  banana's new grounded position against the bowl's, instead of trusting
  its own narrative.
- Not yet tried: an explicit reward term for "object stays near the
  gripper once grasped" / "object is near the bowl once released" — the
  current reward only shapes gripper-to-goal distance and has zero
  awareness of whether anything is actually being carried.

## Roadmap to v0.1 (open source)

1. Trained SAC checkpoint + training curves committed to the repo
2. Success-rate benchmark: scripted vs Gemini planner, 50 episodes each
3. Grasp-contact reward shaping (cube and pick-place both stall on physical
   grasp/hold reliability, not just reach precision)
4. Domain randomization toggles (object size/mass/friction, camera jitter)
5. LeRobot-format dataset export of successful rollouts (for imitation learning)

## License / credits

Arm model: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`trs_so_arm100`, Apache-2.0), based on TheRobotStudio's SO-ARM100.

Banana/bowl meshes: [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset)
(MIT), a MuJoCo/PyBullet-ready packaging of the
[YCB Object and Model Set](https://www.ycbbenchmarks.com/) (Calli et al.,
CC BY 4.0).
