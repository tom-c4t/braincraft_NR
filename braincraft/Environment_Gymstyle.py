import gymnasium as gym
from gymnasium import spaces
import numpy as np
from bot import Bot
from typing import Optional
from environment_1 import Environment
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

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
    self.fig = plt.figure(figsize=(10,5))
    self.env = Environment()
    self.steer = False
    self.sum_actions = 0.0

  def step(self, action):

    action = action * 5
    pre_position = self.bot.position
    energy, hit, distances, values = self.bot.forward(action, self.env)
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

    """
    # rendering
    world = self.env.world
    world_rgb = self.env.world_rgb
    
    ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
    ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
    ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
    ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()

    graphics = {
            "topview" : ax1.imshow(self.env.world_rgb, interpolation="nearest", origin="lower",
                                   extent = [0.0, world.shape[1]/max(world.shape),
                                             0.0, world.shape[0]/max(world.shape)]),
            "bot" : ax1.add_artist(Circle((0,0), 0.05,
                                          zorder=50, facecolor="white", edgecolor="black")),
            "rays" : ax1.add_collection(LineCollection([], color="C1", linewidth=0.5, zorder=30)),
            "hits" :  ax1.scatter([], [], s=1, linewidth=0, color="black", zorder=40),
            "camera" : ax2.imshow(np.zeros((1,1,3)), interpolation="nearest",
                                  origin="lower", extent = [0.0, 1.0, 0.0, 1.0]),
            "energy" : ax2.add_collection(
                LineCollection([[(0.1, 0.1),(0.9, 0.1)],
                                [(0.1, 0.1),(0.9, 0.1)],
                                [(0.1, 0.1),(0.9, 0.1)]],
                               color=("black", "white", "C1"), linewidth=(20,18,12),
                               capstyle="round", zorder=150)) }

    self.bot.camera.render(self.bot.position, self.bot.direction,self.env.world, self.env.colormap)
    graphics["rays"].set_segments(self.bot.camera.rays)
    graphics["hits"].set_offsets(self.bot.camera.rays[:,1,:])
    graphics["bot"].set_center(self.bot.position)
    if energy < self.bot.energy:
        graphics["energy"].set_color( ("black", "white", "C2") )
    else:
        graphics["energy"].set_color( ("black", "white", "C1") )        

    if self.bot.energy > 0:
        ratio = self.bot.energy/self.bot.energy_max
        graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                          [(0.1, 0.1),(0.9, 0.1)],
                                          [(0.1, 0.1),(0.1 + ratio*0.8, 0.1)]])
    else:
        graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                          [(0.1, 0.1),(0.9, 0.1)]])            
    graphics["camera"].set_data(self.bot.camera.framebuffer)
    plt.pause(1/60)
    """

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
    
    #print(f"Distances: {distances[31]}")
    if distances[31] < 0.2:
      self.steer = True
    if distances[31] > 0.4:
      self.steer = False

    if hit is True:
      print( "Hit!" )
      reward = 0.0
      self.sum_actions = 0.0
      reward -= 10.0
      if np.sign(action) > 0:
        reward += 10.0
       
    else:
      if self.steer is True:
        print("Steering")
        self.sum_actions += action
        reward = abs(self.sum_actions)

      else:
        print("Going straight")
        self.sum_actions = 0.0
        # maximize smaller value between leftmost and rightmost sensor
        max_dist = min(distances[0], distances[-1])

        reward = max_dist * 100


    
    return reward