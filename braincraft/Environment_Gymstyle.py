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
    self.action_space = spaces.Box(low = -1, high=1, shape=(1,), dtype=np.float32)
    # input from distance sensors
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(67,), dtype=np.float32)
    self.bot = Bot()
    self.reward = 0

  def step(self, action):

    environment = Environment()
    action = action * 5
    pre_position = self.bot.position
    energy, hit, distances, values = self.bot.forward(action, environment)
    post_position = self.bot.position
    observation = self._get_obs()
    self.reward = self._get_reward(self.reward, distances, pre_position, post_position, hit, action)
    print(f"Reward: {self.reward}")

    if self.bot.energy <= 0:
      done = True
    else:
      done = False

    truncated = False
    info = {"Step done": 1}

    return observation, self.reward, done, truncated, info
  
  def reset(self, seed: Optional[int] = None, options: Optional[dict]=None):
    super().reset(seed=seed)
    self.bot.position = (0.5, 0.5)
    self.bot.direction = np.radians(90)
    self.bot.energy = 1000

    observation = self._get_obs()
    info = {"Environment reset": 1}

    return observation, info
  
  def render(self, mode='human'):
    ...
  def close (self):
    ...
  def _get_obs(self):
    states = np.zeros(shape=(67,), dtype=np.float32)
    if self.bot.camera.depths.max() != 0:
      states[:64] = self.bot.camera.depths / self.bot.camera.depths.max()
    else:
      states[:64] = self.bot.camera.depths 
    states[64:] = self.bot.hit, self.bot.energy/self.bot.energy_max, 1.0

    return states
  
  def _get_reward(self, reward, distances, pre_position, post_position, hit, action):
    if hit is False:
      reward += 1
    else:
      if abs(action) > 3.5:
        reward = 1

    # difference between leftmost and rightmost sensor
    dist_diff = abs(distances[0] - distances[-1])
    # subtract difference from 1, so that higher values mean better action
    dist_diff = 1- dist_diff
    # all sensors should be as free as possible
    dist_sum = np.sum(distances)
    no_move_penalty = 0
    if np.linalg.norm(pre_position - post_position) < 0.006:
      no_move_penalty = -10

    #reward = dist_sum + dist_diff + no_move_penalty
    #reward = np.random.randint(-20, 30)
    # scale with 10 for better resolution
    return reward