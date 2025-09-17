import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import Environment_Gymstyle as Env

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def training_function():

    gym.register(
        id="BotEnv",
        entry_point=Env.BotEnv,
        max_episode_steps=100000
    )

    env = gym.make("BotEnv")

    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    training_function()