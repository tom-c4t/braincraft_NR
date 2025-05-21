# braintenberg.py - Raycast maze simulator
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Maze simulator using the raycast Digital Differential Analyzer (DDA) algorithm.

See:

* Ray-Casting Tutorial For Game Development And Other Purposes - F. Permadi (1996)
  https://permadi.com/1996/05/ray-casting-tutorial-table-of-contents/

* Tangentially, we can fix your raycaster.- S. Mitelli (2024)
  https://www.scottsmitelli.com/articles/we-can-fix-your-raycaster/
"""


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    from bot import Bot
    from camera import Camera
    
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1,-1,-1, 1, 0, 0, 1, 0, 0, 1],
        [1,-1,-1, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    colormap = {
        -3 : np.array([200, 200, 255]), # ground (light blue)
        -2 : np.array([255, 255, 255]), # ground (normal)
        -1 : np.array([100, 100, 255]), # sky
         0 : np.array([255, 255, 255]), # empty
         1 : np.array([200, 200, 200]), # wall (light)
         2 : np.array([100, 100, 100]), # wall (dark)
         3 : np.array([255, 255,   0]), # wall (yellow)
         4 : np.array([  0,   0, 255]), # wall (blue)
         5 : np.array([255,   0,   0]), # wall (red)
         6 : np.array([  0, 255,   0]), # wall (green)
    }


    bot = Bot(maze, colormap, Camera(fov = 60, resolution = 64))
    
        
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
    ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
    ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
    ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()

    maze_rgb = [colormap[i] for i in maze.ravel()]
    maze_rgb = np.array(maze_rgb).reshape(maze.shape[0], maze.shape[1], 3)    
    ax1.imshow(maze_rgb, interpolation="nearest", origin="lower",
               extent = [0.0, maze.shape[1]/max(maze.shape),
                         0.0, maze.shape[0]/max(maze.shape)])

    graphics = {
        "bot" : ax1.add_artist(Circle(bot.position, bot.radius,
                                        zorder=50, facecolor="white", edgecolor="black")),
        "rays" : ax1.add_collection(LineCollection([], color="C1", linewidth=0.5, zorder=30)),
        "hits" :  ax1.scatter([], [], s=1, linewidth=0, color="black", zorder=40),
        "camera" : ax2.imshow(bot.camera.framebuffer, interpolation="nearest",
                              origin="lower", extent = [0.0, 1.0, 0.0, 1.0]),
        "energy" : ax2.add_collection(
            LineCollection([[(0.1, 0.1),(0.9, 0.1)],
                            [(0.1, 0.1),(0.9, 0.1)],
                            [(0.1, 0.1),(0.9, 0.1)]],
                           color=("black", "white", "C1"), linewidth=(20,18,12),
                           capstyle="round", zorder=150))
    }

    def update(frame=0):
        global bot, graphics

        if bot.energy < 0:
            return
                    
        direction = 0
        D = bot.camera.depths
        D = 0.2 - D * (D < 0.2)
        n = bot.camera.resolution//2
        d = D[:n].sum() - D[n:].sum()
        if d > 0:   direction = +np.radians(0.5)
        elif d < 0: direction = -np.radians(0.5)
        energy = bot.energy
        bot.forward(direction)

        graphics["rays"].set_segments(bot.camera.rays)
        graphics["hits"].set_offsets(bot.camera.rays[:,1,:])
        graphics["bot"].set_center(bot.position)
        if energy < bot.energy:
            graphics["energy"].set_color( ("black", "white", "C2") )
        else:
            graphics["energy"].set_color( ("black", "white", "C1") )        
        if bot.energy > 0:
            ratio = bot.energy/bot.max_energy
            graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                             [(0.1, 0.1),(0.9, 0.1)],
                                             [(0.1, 0.1),(0.1 + ratio*0.8, 0.1)]])
        else:
            graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                             [(0.1, 0.1),(0.9, 0.1)]])            
        graphics["camera"].set_data(bot.camera.framebuffer)

    update()
    # fig.savefig("raycast.png")        
    ani = FuncAnimation(fig, update, frames=360, interval=1, repeat=True)
    # ani.save(filename="raycast.mp4", writer="ffmpeg", fps=30)
    plt.show()
