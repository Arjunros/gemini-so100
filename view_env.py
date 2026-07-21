"""Interactive viewer (Windows): drag the camera, watch the scene live.
Usage: python view_env.py
"""
import os, sys, time
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from geminibot.env import SO100ReachEnv

env = SO100ReachEnv()
obs, info = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        env.step(env.action_space.sample() * 0.2)
        v.sync()
        time.sleep(0.02)
