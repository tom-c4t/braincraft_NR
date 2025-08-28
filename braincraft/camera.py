# camerat.py - Raycast maze simulator
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
import numpy as np


class Camera:
    """ A camera based on raycasting """
    
    def __init__(self, fov = 60, resolution = 64):
        """Build a new camera with a fov (field of view) and a given resolution"""
        
        self.fov = fov
        self.resolution = resolution
        self.depths = np.zeros(resolution)
        self.values = np.zeros(resolution, dtype=int)
        self.rays = np.zeros((resolution,2,2))
        self.cells = np.zeros((resolution,2), dtype=int)
        self.framebuffer = np.zeros((resolution,resolution,3), dtype=np.ubyte)

    def raycast(self, origin, direction, world):
        """Compute the end coordinates of a ray that is cast from given
        origin and direction, until it reaches a wall or exit the
        world. The physical size of the world is normalized to fit the unit
        square originating at (0,0). The method is called Digital
        Differential Analyzer (DDA)

        Parameters
        ----------

        origin: tuple
          Origin of the ray that must be inside the world

        direction: float
          Direction of the ray (radians)

        world: ndarray
          2D array describing world occupancy where value > 0 means
          occupîed and 0 means empty.

        Returns
        -------

        end_position: tuple

          End point of the ray in normalized coordinates or None if the
          ray exits the world without hitting a wall

        end_cell: tuple

          End point of the ray in grid coordinates or None if the ray
          exits the world without hitting a wall

        face : tuple

          Face that was hit: (1,0), (-1,0), (0,1) or (0, -1)

        steps: int

          Number of steps before hitting a wall (debug)

        """

        # Cell size from normalized world dimension
        cell_size = 1/max(world.shape)

        # Grid position
        x, y = origin[0] / cell_size, origin[1] / cell_size

        # Ray direction
        dx, dy = np.cos(direction), np.sin(direction)

        # Raycasting loop
        steps = 0
        cell_x_prev, cell_y_prev = x, y

        while True:
            steps += 1

            # Determine the cell the ray is currently in
            cell_x, cell_y = int(np.floor(x)), int(np.floor(y))

            # Check if the ray exits the world without hitting a wall
            if cell_x < 0 or cell_y < 0 or cell_x >= world.shape[1] or cell_y >=  world.shape[0]:
                return None, None, None, steps 

            # Check for wall collision
            if world[cell_y, cell_x] > 0:
                face = cell_x - cell_x_prev, cell_y - cell_y_prev
                return np.array([x*cell_size, y*cell_size]), (cell_y, cell_x), face, steps

            cell_x_prev, cell_y_prev = cell_x, cell_y

            # Calculate next grid crossing
            epsilon = 1e-10
            if dx > epsilon:    tx = (cell_x + 1 - x) / dx
            elif dx < -epsilon: tx = (cell_x - x) / dx
            else:               tx = float('inf')

            if dy > epsilon:    ty = (cell_y + 1 - y) / dy
            elif dy < -epsilon: ty = (cell_y - y) / dy
            else:               ty = float('inf')

            # Ensure t_min is strictly positive to move the ray forward
            t_min = max(min(tx, ty), cell_size/10)  # Small epsilon ensures progress

            # Move ray to next cell boundary, talking vertical/horizontal cases into account
            epsilon = 1e-15
            if abs(dx) < epsilon:
                y += np.sign(dy)
            elif abs(dy) < epsilon:
                x += np.sign(dx)
            else:
                x += dx * t_min
                y += dy * t_min


    def update(self, position, direction, world, colormap):
        """Update the sensors with the current view"""

        # See https://www.scottsmitelli.com/articles/we-can-fix-your-raycaster/
        # angles = direction + np.radians(np.linspace(+self.fov/2,-self.fov/2, n, endpoint=True))
        n = self.resolution
        D = 0.25 # projection distance
        W = 2 * D * np.tan(np.radians(self.fov)/2)
        X = W/2 * np.linspace(+1, -1, n, endpoint=True)
        angles = direction + np.atan2(X, D)
        start = position
                
        for i, angle in enumerate(angles):
            end, cell, face, steps = self.raycast(start, angle, world)
            d = np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
            self.rays[i] = start, end
            self.depths[i] = d
            self.values[i] = world[cell]
            self.cells[i] = cell

            
    def render(self, position, direction, world, colormap, outline=True, lighting=True):
        """Update the sensors with the current view and render to framebuffer."""
                
        # Clear framebuffer by coloring sky (top half) and ground (bottom half)
        n = self.resolution

        sky = colormap[-1]
        self.framebuffer[n//2:, :] = sky * np.linspace(0.75, 1.00, n//2).reshape(n//2,1,1)

        ground = colormap[-2]
        self.framebuffer[:n//2, :] = ground * np.linspace(1.00, 0.65, n//2).reshape(n//2,1,1)
        
        cell_prev = None
        face_prev = None
        start = position

        # Cast each ray and write them in the framebuffer
        # angles = direction + np.radians(np.linspace(+self.fov/2,-self.fov/2, n, endpoint=True))
        # See https://www.scottsmitelli.com/articles/we-can-fix-your-raycaster/
        D = 0.25 # projection distance
        W = 2 * D * np.tan(np.radians(self.fov)/2)
        X = W/2 * np.linspace(+1, -1, n, endpoint=True)
        angles = direction + np.atan2(X, D)
        
        for i, angle in enumerate(angles):
            end, cell, face, steps = self.raycast(start, angle, world)
            d = np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)

            self.rays[i] = start, end
            self.depths[i] = d
            self.values[i] = world[cell]
            self.cells[i] = cell
            
            # We should use a fovy instead of hardcoding height
            height = 0.08/(d*np.cos(direction - angle))

            ymin = int(max(np.floor((0.5 - height/2)*n), 0))
            ymax = int(min(np.floor((0.5 + height/2)*n), n))
            depth = abs(d) - abs(d) % (1/n)
            darken = 1.0
            
            # Lighting
            if lighting and face == (-1,0):
                darken = 0.75*darken
                
            # Outline (vertical)
            if outline and i > 0:
                if np.any(self.cells[i-1] != self.cells[i]):
                    darken = 0.75*darken
                elif np.all(self.cells[i-1] == self.cells[i]) and face != face_prev:
                    darken = 0.75*darken
            self.framebuffer[ymin:ymax,i] = np.array(colormap[world[cell]])*(1 - depth) * darken

            # Outline (top and bottom)
            if outline:
                if ymax > 0:
                    self.framebuffer[ymax-1:ymax,i] = self.framebuffer[ymax-1,i]*0.75
                if ymin < n:
                    self.framebuffer[ymin:ymin+1,i] = self.framebuffer[ymin,  i]*0.75

            face_prev = face

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    "Usage example"
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from environment_1 import Environment

    world = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 4, 4, 0, 0, 3, 3, 0, 1],
        [1, 0, 4, 0, 0, 0, 0, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 6, 0, 0, 0, 0, 5, 0, 1],
        [1, 0, 6, 6, 0, 0, 5, 5, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    env = Environment(world = world)
    
    radius = 0.05
    position, direction = (0.5, 0.5),  np.radians(0)
    camera = Camera(fov = 60, resolution = 256)
    
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
    ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
    ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
    ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()
    ax1.imshow(env.world_rgb, interpolation="nearest", origin="lower",
               extent = [0.0, env.world.shape[1]/max(env.world.shape),
                         0.0, env.world.shape[0]/max(env.world.shape)])
    bot = ax1.add_artist(Circle(position, radius, zorder=50, facecolor="white", edgecolor="black"))
    rays = ax1.add_collection(LineCollection([], color="C1", linewidth=0.5, zorder=30))
    hits = ax1.scatter([], [], s=1, linewidth=0, color="black", zorder=40)    
    framebuffer = ax2.imshow(camera.framebuffer, interpolation="nearest",
                             origin="lower", extent = [0.0, 1.0, 0.0, 1.0])

    def update(frame=0):
        global position, direction
        direction += np.radians(0.5)
        camera.update(position, direction, env.world, env.colormap)
        camera.render(position, direction, env.world, env.colormap)
        rays.set_segments(camera.rays)
        hits.set_offsets(camera.rays[:,1,:])
        framebuffer.set_data(camera.framebuffer)

    # update()
    # fig.savefig("raycast.png")        
    ani = FuncAnimation(fig, update, frames=360, interval=1, repeat=True)
    # ani.save(filename="raycast.mp4", writer="ffmpeg", fps=30)
    plt.show()
