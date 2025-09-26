import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import Environment_Gymstyle as Env
from bot import Bot
from environment_1 import Environment

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def training_function(timesteps=10):

    gym.register(
        id="BotEnv",
        entry_point=Env.BotEnv,
        max_episode_steps=100000
    )

    env = gym.make("BotEnv")

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1)
    print(f"Timesteps: {timesteps}")
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save("ddpg_bot")

def evaluate_self(model, Bot, Environment, runs=10, seed=None, debug=False):
    if seed is None:
        seed = np.random.randint(10_000_000)
    
    if debug:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection

        environment = Environment()
        world = environment.world
        world_rgb = environment.world_rgb
        
        fig = plt.figure(figsize=(10,5))
        ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
        ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
        ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
        ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()

        graphics = {
            "topview" : ax1.imshow(environment.world_rgb, interpolation="nearest", origin="lower",
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

    scores = []
    seeds = np.random.randint(0, 1_000_000, runs)
    states = np.zeros(shape=(67,), dtype=np.float32)
    # print(f"Seeds : {seeds}")

    for i in range(runs):
        np.random.seed(seeds[i])
        environment = Environment()

        if debug:
            graphics["topview"].set_data(environment.world_rgb)
        
        bot = Bot()

        distance = 0
        hits = 0
        iteration = 0

        # Initial update
        if debug:
            bot.camera.render(bot.position, bot.direction,
                              environment.world, environment.colormap)
        else:
            bot.camera.update(bot.position, bot.direction,
                              environment.world, environment.colormap)

        # Run until no energy
        while bot.energy > 0:

            energy = bot.energy
            states[:64] = bot.camera.depths
            states[64:] = bot.hit, bot.energy, 1.0
            action = model.predict(states)
            
            # No warmup here
            if iteration > 0:
                p = bot.position
                bot.forward(action[0][0], environment, debug)
                distance += np.linalg.norm(p - bot.position)
                hits += bot.hit
            iteration += 1

            if debug:
                graphics["rays"].set_segments(bot.camera.rays)
                graphics["hits"].set_offsets(bot.camera.rays[:,1,:])
                graphics["bot"].set_center(bot.position)
                if energy < bot.energy:
                    graphics["energy"].set_color( ("black", "white", "C2") )
                else:
                    graphics["energy"].set_color( ("black", "white", "C1") )        

                if bot.energy > 0:
                    ratio = bot.energy/bot.energy_max
                    graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.1 + ratio*0.8, 0.1)]])
                else:
                    graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.9, 0.1)]])            
                graphics["camera"].set_data(bot.camera.framebuffer)
                plt.pause(1/60)

            scores.append (distance)

    return np.mean(scores), np.std(scores)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    training_function(timesteps=1000)
    model = SAC.load("ddpg_bot")
    seed = 42
    score, std_dev = evaluate_self(model, Bot, Environment,runs=5,debug=True)
    print(f"score: {score} +- {std_dev}")