"""Gemini-as-VLA planner.

Gemini never controls joints. It looks at the camera frame + task instruction
+ proprioceptive state, and emits ONE structured subgoal:

    {"target_xyz": [x, y, z], "jaw": 0..1, "phase": "...", "done": bool, "reason": "..."}

The trained RL policy then chases that subgoal at 25 Hz until the next
planning tick. Requires GOOGLE_API_KEY in the environment.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

SYSTEM_PROMPT = """You are the high-level planner for a real SO-101 robot arm
(simulated in MuJoCo) with a parallel-jaw gripper. You see the workspace from a
fixed front camera. The arm base is at the origin; +x points away from the base
toward the camera side, +y is left, +z is up. All units are metres.

Reachable workspace: x in [0.10, 0.32], y in [-0.20, 0.20], z in [0.02, 0.25].
The red cube is 2.5 cm wide. Grasp procedure (never skip a phase):
  1. hover:   target = cube position but z = 0.075, jaw=1 (open)
  2. descend: target = cube position with z = 0.015, jaw=1 (still open)
  3. grasp:   same target, jaw=0 (close)
  4. lift:    same x,y, z = 0.15, jaw=0

VERIFY BEFORE ADVANCING — you are given exact gripper_xyz and cube_xyz each
call. Trust these numbers over the image:
- Advance from hover only when gripper_xyz is within 4 cm of the hover point.
- Close the jaw only when gripper_xyz is within 4 cm of the cube.
- Never skip the descend phase.
- Set done=true ONLY if cube_xyz has z > 0.08 (verifiably lifted).
- If after a grasp+lift the cube still has z < 0.03, the grasp FAILED:
  reopen the jaw (jaw=1), go back to hover above the cube's CURRENT position,
  and try again. The cube may have moved — always use its current position.

Respond with STRICT JSON only, no markdown fences, matching:
{"target_xyz": [x, y, z], "jaw": 0.0-1.0, "phase": "<short label>",
 "done": true/false, "reason": "<one sentence>"}"""


PICKPLACE_SYSTEM_PROMPT = """You are the high-level planner for a real SO-101
robot arm (simulated in MuJoCo) with a parallel-jaw gripper, performing a
pick-and-place task: move one object to another. You see the workspace from
a fixed front camera. The arm base is at the origin; +x points away from the
base toward the camera side, +y is left, +z is up. All units are metres.

Reachable workspace: x in [0.10, 0.32], y in [-0.20, 0.20], z in [0.02, 0.25].

Each call you are given the current best-known 3D positions of the pick
object and the place destination (pick_xyz, place_xyz). These come from a
VISION system (not ground truth), so trust them but expect a little error
(usually under 1cm). You also get exact gripper_xyz (robot proprioception --
this one IS ground truth).

Procedure (never skip a phase):
  1. hover:       target = pick_xyz but z = pick_xyz.z + 0.06, jaw=1 (open)
  2. descend:     target = pick_xyz with z = 0.015, jaw=1 (still open)
  3. grasp:       same target, jaw=0 (close)
  4. lift:        same x,y, z = 0.15, jaw=0
  5. place_hover: target = place_xyz but z = place_xyz.z + 0.08, jaw=0 (still holding)
  6. release:     target = place_xyz with z = place_xyz.z + 0.03, jaw=1 (open, drop it)

VERIFY BEFORE ADVANCING -- trust gripper_xyz over the image for distances:
- Advance from hover only when gripper_xyz is within 4 cm of the hover point.
- Close the jaw only when gripper_xyz is within 4 cm of pick_xyz.
- Never skip the descend phase.
- Advance from lift to place_hover only once gripper z > 0.12 (actually
  lifted, not still on the way up).
- Look at the image to judge whether the object is actually being carried
  (visible near/below the jaw as the arm moves) -- if it looks like the
  grasp failed (nothing visible in the gripper at lift height), reopen the
  jaw, go back to hover, and try again.
- Set done=true only after release, once the image suggests the object has
  settled at the destination. You do not get a direct oracle success signal
  here (unlike a simpler grasp-only task) -- use your visual judgment.

Respond with STRICT JSON only, no markdown fences, matching:
{"target_xyz": [x, y, z], "jaw": 0.0-1.0, "phase": "<short label>",
 "done": true/false, "reason": "<one sentence>"}"""


@dataclass
class Subgoal:
    target_xyz: list[float]
    jaw: float
    phase: str
    done: bool
    reason: str


class GeminiPlanner:
    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        from google import genai  # deferred so sim-only use needs no key

        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.history: list[str] = []

    def locate_objects(self, frame: np.ndarray, labels: list[str]) -> dict[str, tuple[float, float]]:
        """Open-vocabulary grounding: ask Gemini to find named objects from
        the image ALONE (no ground-truth coordinates given, unlike plan()'s
        gripper_xyz/object_xyz args). Returns pixel-space box centers (u, v)
        per label actually found; the caller converts to world xyz via
        geometry.pixel_to_world using the real camera pose."""
        from google.genai import types

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        h, w = frame.shape[:2]

        prompt = (
            f"Locate each of these objects in the image: {', '.join(labels)}.\n"
            "Respond with STRICT JSON, no markdown fences: a list of objects "
            '{"label": "<name>", "box_2d": [ymin, xmin, ymax, xmax]}, with '
            "box_2d coordinates normalized to a 0-1000 scale for this image "
            "regardless of its actual resolution. Omit any object you cannot see."
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        # Gemini occasionally appends trailing text after the JSON array;
        # decode just the first valid JSON value and ignore the rest.
        detections = json.JSONDecoder().raw_decode(raw)[0]

        out: dict[str, tuple[float, float]] = {}
        for d in detections:
            ymin, xmin, ymax, xmax = d["box_2d"]
            u = (xmin + xmax) / 2.0 / 1000.0 * w
            v = (ymin + ymax) / 2.0 / 1000.0 * h
            out[d["label"]] = (u, v)
        return out

    def locate_task_objects(self, frame: np.ndarray, instruction: str) -> dict[str, tuple[float, float]]:
        """Fully open-vocabulary: given ONLY the image and a free-form
        instruction like 'put the banana in the bowl', ask Gemini which
        object should be picked and which is the destination -- no object
        names are hardcoded or supplied in advance. Returns pixel-space box
        centers keyed by role ("pick", "place"); the caller converts to
        world xyz via geometry.pixel_to_world."""
        from google.genai import types

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        h, w = frame.shape[:2]

        prompt = (
            f"TASK: {instruction}\n"
            "Identify the object that should be PICKED UP and the "
            "destination object/location it should be PLACED at. Respond "
            "with STRICT JSON, no markdown fences:\n"
            '{"pick": {"label": "<name>", "box_2d": [ymin,xmin,ymax,xmax]}, '
            '"place": {"label": "<name>", "box_2d": [ymin,xmin,ymax,xmax]}}\n'
            "box_2d normalized to a 0-1000 scale for this image."
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        d = json.JSONDecoder().raw_decode(raw)[0]

        out: dict[str, tuple[float, float]] = {}
        for role in ("pick", "place"):
            ymin, xmin, ymax, xmax = d[role]["box_2d"]
            u = (xmin + xmax) / 2.0 / 1000.0 * w
            v = (ymin + ymax) / 2.0 / 1000.0 * h
            out[role] = (u, v)
        return out

    def plan_pickplace(self, frame: np.ndarray, instruction: str, gripper_xyz: np.ndarray,
                        pick_xyz: np.ndarray, place_xyz: np.ndarray) -> Subgoal:
        """Like plan(), but for the generalized pick-and-place task: pick_xyz
        and place_xyz are vision-grounded (see locate_task_objects), not
        oracle simulator state."""
        from google.genai import errors as genai_errors
        from google.genai import types

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)

        state = (
            f"TASK: {instruction}\n"
            f"gripper_xyz: {np.round(gripper_xyz, 3).tolist()}\n"
            f"pick_xyz: {np.round(pick_xyz, 3).tolist()}\n"
            f"place_xyz: {np.round(place_xyz, 3).tolist()}\n"
            f"previous_phases: {self.history[-6:]}"
        )

        resp = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                        state,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=PICKPLACE_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                break
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                    wait = float(m.group(1)) + 2 if m else 60.0
                    print(f"  [rate limited, waiting {wait:.0f}s (attempt {attempt+1}/6)]")
                elif "503" in msg or "UNAVAILABLE" in msg:
                    wait = 2 ** attempt
                    print(f"  [gemini busy, retry {attempt+1}/6 in {wait}s]")
                else:
                    raise
                time.sleep(wait)
        if resp is None:
            raise RuntimeError("Gemini unavailable after 6 retries")

        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        d = json.JSONDecoder().raw_decode(raw)[0]
        sg = Subgoal(
            target_xyz=[float(v) for v in d["target_xyz"]],
            jaw=float(d.get("jaw", 1.0)),
            phase=str(d.get("phase", "?")),
            done=bool(d.get("done", False)),
            reason=str(d.get("reason", "")),
        )
        self.history.append(sg.phase)
        return sg

    def plan(self, frame: np.ndarray, instruction: str,
             gripper_xyz: np.ndarray, cube_xyz: np.ndarray) -> Subgoal:
        from google.genai import errors as genai_errors
        from google.genai import types

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)

        state = (
            f"TASK: {instruction}\n"
            f"gripper_xyz: {np.round(gripper_xyz, 3).tolist()}\n"
            f"cube_xyz: {np.round(cube_xyz, 3).tolist()}\n"
            f"previous_phases: {self.history[-6:]}"
        )

        resp = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                        state,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                break
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                    wait = float(m.group(1)) + 2 if m else 60.0
                    print(f"  [rate limited, waiting {wait:.0f}s "
                          f"(attempt {attempt+1}/6)]")
                elif "503" in msg or "UNAVAILABLE" in msg:
                    wait = 2 ** attempt
                    print(f"  [gemini busy, retry {attempt+1}/6 in {wait}s]")
                else:
                    raise
                time.sleep(wait)
        if resp is None:
            raise RuntimeError("Gemini unavailable after 6 retries")

        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        d = json.loads(raw)
        sg = Subgoal(
            target_xyz=[float(v) for v in d["target_xyz"]],
            jaw=float(d.get("jaw", 1.0)),
            phase=str(d.get("phase", "?")),
            done=bool(d.get("done", False)),
            reason=str(d.get("reason", "")),
        )
        self.history.append(sg.phase)
        return sg


class ScriptedPlanner:
    """Drop-in offline replacement for GeminiPlanner (no API key needed)."""

    def __init__(self):
        self.phase = "hover"

    def plan(self, frame, instruction, gripper_xyz, cube_xyz) -> Subgoal:
        cx, cy, cz = cube_xyz
        xy_err = float(np.linalg.norm(gripper_xyz[:2] - np.array([cx, cy])))
        if self.phase == "hover":
            sg = Subgoal([cx, cy, cz + 0.06], 1.0, "hover", False, "move above cube")
            if np.linalg.norm(gripper_xyz - np.array([cx, cy, cz + 0.06])) < 0.045:
                self.phase = "descend"
        elif self.phase == "descend":
            sg = Subgoal([cx, cy, 0.015], 1.0, "descend", False, "lower to cube")
            # require tight XY alignment (cube is 2.5cm wide) as well as depth
            # before closing -- z-only was letting the jaw close 2-5cm off
            # center, missing the cube entirely.
            if gripper_xyz[2] < 0.045 and xy_err < 0.01:
                self.phase = "grasp"
        elif self.phase == "grasp":
            sg = Subgoal([cx, cy, 0.015], 0.0, "grasp", False, "close jaw")
            self.phase = "lift"
        else:
            done = cz > 0.08
            if not done and gripper_xyz[2] > 0.1 and cz < 0.03:
                self.phase = "hover"  # grasp failed, retry
                return Subgoal([cx, cy, cz + 0.06], 1.0, "hover", False, "grasp failed, retrying")
            sg = Subgoal([cx, cy, 0.15], 0.0, "lift", done, "lift cube")
        return sg


class ScriptedPickPlacePlanner:
    """Oracle-driven pick-and-place, for fast offline testing without API
    calls (same role ScriptedPlanner plays for the single-cube task).
    Unlike GeminiPlanner.plan_pickplace, pick_xyz/place_xyz here are true
    simulator state, refreshed every call -- so retry logic can verify
    outcomes directly instead of relying on visual judgment."""

    def __init__(self):
        self.phase = "hover"

    def plan(self, frame, instruction, gripper_xyz, pick_xyz, place_xyz) -> Subgoal:
        px, py, pz = pick_xyz
        dx, dy, dz = place_xyz
        xy_err = float(np.linalg.norm(gripper_xyz[:2] - np.array([px, py])))

        if self.phase == "hover":
            sg = Subgoal([px, py, pz + 0.06], 1.0, "hover", False, "move above pick object")
            if np.linalg.norm(gripper_xyz - np.array([px, py, pz + 0.06])) < 0.045:
                self.phase = "descend"
        elif self.phase == "descend":
            sg = Subgoal([px, py, 0.015], 1.0, "descend", False, "lower to pick object")
            if gripper_xyz[2] < 0.045 and xy_err < 0.01:
                self.phase = "grasp"
        elif self.phase == "grasp":
            sg = Subgoal([px, py, 0.015], 0.0, "grasp", False, "close jaw")
            self.phase = "lift"
        elif self.phase == "lift":
            if gripper_xyz[2] > 0.1 and pz < 0.03:
                self.phase = "hover"  # grasp failed, retry
                return Subgoal([px, py, pz + 0.06], 1.0, "hover", False, "grasp failed, retrying")
            sg = Subgoal([px, py, 0.15], 0.0, "lift", False, "lift picked object")
            if gripper_xyz[2] > 0.12:
                self.phase = "place_hover"
        elif self.phase == "place_hover":
            sg = Subgoal([dx, dy, dz + 0.08], 0.0, "place_hover", False, "move above place location")
            if np.linalg.norm(gripper_xyz - np.array([dx, dy, dz + 0.08])) < 0.045:
                self.phase = "release"
        elif self.phase == "release":
            self.phase = "verify"
            sg = Subgoal([dx, dy, dz + 0.03], 1.0, "release", False, "release object")
        else:  # verify
            xy_place_err = float(np.linalg.norm(np.array([px, py]) - np.array([dx, dy])))
            placed = xy_place_err < 0.03 and abs(pz - dz) < 0.03
            if placed:
                sg = Subgoal([dx, dy, dz + 0.08], 1.0, "done", True, "placed successfully")
            else:
                self.phase = "hover"
                sg = Subgoal([px, py, pz + 0.06], 1.0, "hover", False, "placement failed, retrying")
        return sg