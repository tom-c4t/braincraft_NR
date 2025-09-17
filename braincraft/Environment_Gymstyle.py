import gymnasium as gym
from gymnasium import spaces
import numpy as np
from bot import Bot
from typing import Optional
from environment_1 import Environment

class BotEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(BotEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Action space of size 1 for continuous action
    self.action_space = spaces.Box(low = -5, high=5, shape=(1,), dtype=np.float32)
    # input from distance sensors
    self.observation_space = spaces.Box(low=0.0, high=1001.0, shape=(67,), dtype=np.float32)
    self.bot = Bot()

  def step(self, action):

    environment = Environment()
    energy, hit, distances, values = self.bot.forward(action, environment)
    observation = self._get_obs()
    reward = self._get_reward(distances)

    if self.bot.energy <= 0:
      done = True
    else:
      done = False

    truncated = False
    info = {"Step done": 1}

    return observation, reward, done, truncated, info
  
  def reset(self, seed: Optional[int] = None, options: Optional[dict]=None):
    super().reset(seed=seed)
    self.bot.position = (0.5, 0.5)
    self.bot.direction = np.radians(90) + np.radians(np.random.uniform(-5, +5))

    observation = self._get_obs()
    info = {"Environment reset": 1}

    print(f"Observation: {observation}")
    print(f"Self obs shape: {self.observation_space.shape}")
    return observation, info
  
  def render(self, mode='human'):
    ...
  def close (self):
    ...
  def _get_obs(self):
    states = np.zeros(shape=(67,), dtype=np.float32)
    states[:64] = self.bot.camera.depths
    states[64:] = self.bot.hit, self.bot.energy, 1.0

    return states
  
  def _get_reward(self, distances):
    # difference between leftmost and rightmost sensor
    dist_diff = abs(distances[0] - distances[-1])
    # subtract difference from 1, so that higher values mean better action
    reward = 1-dist_diff
    # scale with 10 for better resolution
    return reward * 10