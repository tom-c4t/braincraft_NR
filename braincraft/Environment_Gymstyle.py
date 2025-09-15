import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BotEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(BotEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Action space of size 1 for continuous action
    self.action_space = spaces.Box(low = -5.0, high=5.0, shape=(1), dtype=np.float32)
    # input from distance sensors
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(67), dtype=np.float32)

  def step(self, action):
    ...
    return observation, reward, done, info
  def reset(self):
    ...
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    ...
  def close (self):
    ...