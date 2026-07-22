"""SO-ARM100 goal-conditioned manipulation environment (MuJoCo, Windows-friendly).

The RL policy learns ONE general skill: drive the gripper to a commanded 3D
target and set a commanded jaw opening. Gemini (the VLA layer) decides WHERE
the target should be and WHEN to open/close the jaw. This split keeps the RL
problem small enough to train on a laptop while Gemini handles semantics.

Observation (24,):  qpos(6) | qvel(6) | gripper_xyz(3) | approach_vec(3) | cube_xyz(3) | goal_xyz(3)
Action (6,) in [-1, 1]:     delta joint-position targets (5 arm joints + jaw)
Reward: shaped distance-to-goal + success bonus (+ optional grasp shaping)
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
SCENE_XML = os.path.abspath(os.path.join(_ASSETS, "trs_so_arm100", "scene_pick.xml"))

# Reachable workspace of the SO-100 (conservative box, metres, arm base at origin)
WORKSPACE_LOW = np.array([0.10, -0.20, 0.02])
WORKSPACE_HIGH = np.array([0.32, 0.20, 0.25])

CUBE_SPAWN_LOW = np.array([0.14, -0.15])
CUBE_SPAWN_HIGH = np.array([0.28, 0.15])

# Heights the VLA planner actually commands during a grasp sequence
# (hover / descend-grasp / lift) -- used to bias training goal sampling
# toward the region the policy needs to be precise in. NOTE: the physically
# "correct" grasp height is 0.015 (cube centroid), but the deployed
# checkpoint was curriculum-tuned against the workspace floor clipping it to
# 0.02 -- lowering WORKSPACE_LOW to unclip this measurably hurt success
# (untrained-for height) without a matching retrain. Keep the two in sync;
# don't change one without the other.
GRASP_HEIGHTS = np.array([0.02, 0.03, 0.075, 0.15])


class SO100ReachEnv(gym.Env):
    """Goal-conditioned reach (and optionally grasp) with the SO-ARM100."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        success_radius: float = 0.025,
        action_scale: float = 0.05,
        randomize_cube: bool = True,
        resample_goals: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None

        self.max_episode_steps = max_episode_steps
        self.success_radius = success_radius
        self.action_scale = action_scale
        self.randomize_cube = randomize_cube
        self.resample_goals = resample_goals

        self._site_gripper = self.model.site("gripper").id
        self._body_jaw = self.model.body("Fixed_Jaw").id
        self._site_goal = self.model.site("goal").id
        self._cube_qpos_adr = self.model.joint("cube_free").qposadr[0]
        self._home = self.model.key("home").qpos.copy()

        self.n_act = self.model.nu  # 6 actuators total (5 arm + jaw)
        self.n_arm = 5              # only the arm joints are RL-controlled
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Jaw is never part of the learned action space: it's driven directly
        # by set_goal() (the VLA planner's open/close command), both here and
        # at deployment. A jointly-trained policy that also owns the jaw
        # collapses on the arm dims too if that action is overridden post-hoc
        # (verified empirically -- freezing any single learned dim after the
        # network has committed to a joint action derails the whole policy).
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_arm,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float64)

        self.goal = np.zeros(3)
        self.jaw_cmd = 1.0
        self._steps = 0
        # Physics runs at 500 Hz (model default 0.002); control at 25 Hz
        self._sim_steps_per_ctrl = 20

    def _clip_to_reach(self, g, rmax=0.30):
        """Clamp a goal into the arm's actual reachable sphere AND the box."""
        center = np.array([0.0, 0.0, 0.06])   # approx shoulder position
        v = np.asarray(g, dtype=float) - center
        n = np.linalg.norm(v)
        if n > rmax:
            v = v / n * rmax
        return np.clip(center + v, WORKSPACE_LOW, WORKSPACE_HIGH)

    def _sample_goal(self, cube_xy: np.ndarray) -> np.ndarray:
        """Training-time curriculum: mostly sample goals near the cube at
        grasp-relevant heights (hover/descend/grasp/lift -- exactly what the
        deployed VLA loop asks for), with some uniform-workspace goals mixed
        in for generalization. Uniform sampling alone left the policy
        imprecise (~3.5cm off) at the low, cube-proximate targets that
        actually matter for grasping."""
        if self.np_random.random() < 0.7:
            jitter = self.np_random.uniform(-0.01, 0.01, size=2)
            z = self.np_random.choice(GRASP_HEIGHTS)
            g = np.array([cube_xy[0] + jitter[0], cube_xy[1] + jitter[1], z])
        else:
            g = self.np_random.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)
        return self._clip_to_reach(g)

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Arm to home keyframe (first 6 dofs), zero velocities
        self.data.qpos[: len(self._home)] = self._home
        self.data.ctrl[:] = np.clip(self._home[: self.n_act], self.ctrl_low, self.ctrl_high)

        # Cube spawn
        if self.randomize_cube:
            xy = self.np_random.uniform(CUBE_SPAWN_LOW, CUBE_SPAWN_HIGH)
        else:
            xy = np.array([0.22, 0.0])
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], 0.0125]
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

        # Goal: either supplied by the caller (Gemini), curriculum-sampled
        # near the cube during training, or uniform in the workspace (eval).
        if options and "goal" in options:
            self.goal = self._clip_to_reach(options["goal"])
        elif self.resample_goals:  # proxy for "training mode"
            self.goal = self._sample_goal(xy)
        else:
            self.goal = self._clip_to_reach(
                self.np_random.uniform(WORKSPACE_LOW, WORKSPACE_HIGH))
        self.model.site_pos[self._site_goal] = self.goal

        # Jaw command: randomized per episode during training so the policy
        # (which only ever sees the jaw's effect via obs, never controls it)
        # is robust to both states; a fixed default at deployment/eval.
        if options and "jaw" in options:
            self.jaw_cmd = float(np.clip(options["jaw"], 0.0, 1.0))
        elif self.resample_goals:  # proxy for "training mode"
            self.jaw_cmd = float(self.np_random.integers(0, 2))
        else:
            self.jaw_cmd = 1.0
        self._apply_jaw()

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._obs(), self._info()

    def _apply_jaw(self) -> None:
        self.data.ctrl[5] = self.ctrl_low[5] + self.jaw_cmd * (self.ctrl_high[5] - self.ctrl_low[5])

    def set_goal(self, goal_xyz, jaw: float | None = None) -> None:
        """Update the target mid-episode (called by the VLA loop)."""
        self.goal = self._clip_to_reach(goal_xyz)
        self.model.site_pos[self._site_goal] = self.goal
        if jaw is not None:  # 0 = closed, 1 = open — mapped onto the Jaw actuator range
            self.jaw_cmd = float(np.clip(jaw, 0.0, 1.0))
            self._apply_jaw()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.data.ctrl[:5] + action * self.action_scale * (self.ctrl_high[:5] - self.ctrl_low[:5])
        self.data.ctrl[:5] = np.clip(target, self.ctrl_low[:5], self.ctrl_high[:5])
        self._apply_jaw()  # ctrl[5] is never touched by the learned action

        for _ in range(self._sim_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        # resample goal (and jaw command) mid-episode so the policy learns
        # goal switches (matches how the VLA planner moves the goal at
        # deployment) without ever coupling jaw state to its own actions
        if self.resample_goals and self._steps % 60 == 0:
            self.goal = self._sample_goal(self.cube_pos()[:2])
            self.model.site_pos[self._site_goal] = self.goal
            self.jaw_cmd = float(self.np_random.integers(0, 2))
            self._apply_jaw()
        obs = self._obs()
        dist = float(np.linalg.norm(self.gripper_pos() - self.goal))
        success = dist < self.success_radius

        reward = -dist                       # dense shaping
        reward += 2.0 if success else 0.0    # sparse bonus
        # top-down orientation shaping: approach axis should point at -z
        reward += 0.1 * float(self.approach_vec() @ np.array([0.0, 0.0, -1.0]))
        reward -= 0.001 * float(np.square(action).sum())
        # discourage freezing while far from goal (kills parking-spot attractor)
        if dist > 0.05 and float(np.abs(self.data.qvel[:5]).sum()) < 0.05:
            reward -= 0.5  # action penalty

        terminated = False                   # keep-alive: goal can move mid-episode
        truncated = self._steps >= self.max_episode_steps
        info = self._info()
        info["is_success"] = success
        info["success"] = success
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- helpers
    def gripper_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site_gripper].copy()

    def approach_vec(self) -> np.ndarray:
        """Unit vector along the jaw's approach axis (points 'out of' the gripper)."""
        v = self.gripper_pos() - self.data.xpos[self._body_jaw]
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])

    def cube_pos(self) -> np.ndarray:
        adr = self._cube_qpos_adr
        return self.data.qpos[adr : adr + 3].copy()

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[: self.n_act],
            self.data.qvel[: self.n_act],
            self.gripper_pos(),
            self.approach_vec(),
            self.cube_pos(),
            self.goal,
        ])

    def _info(self) -> dict[str, Any]:
        return {
            "gripper_pos": self.gripper_pos(),
            "cube_pos": self.cube_pos(),
            "goal": self.goal.copy(),
            "distance": float(np.linalg.norm(self.gripper_pos() - self.goal)),
        }

    def render(self, camera: str = "vla_cam") -> np.ndarray:
        """Return an RGB frame (H, W, 3). Used both for videos and Gemini input."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(**kwargs) -> SO100ReachEnv:
    return SO100ReachEnv(**kwargs)
