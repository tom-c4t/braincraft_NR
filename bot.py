# bot.py - Raycast maze simulator
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
import numpy as np
from camera import Camera

class Bot:
    """ A circular bot with a camera and an energy gauge"""
    
    def __init__(self, maze, colormap, camera = None):
        """
        Create a new bot for the given maze and associated colormap

        Parameters
        ----------

        maze: ndarray
          2D array describing maze occupancy where value > 0 means
          occupîed and <=0 means empty.

        colormap: list
          List of tuple of (int, color) corresponding to wall colors                 
        """
        self.radius    = 0.05
        self.position  = 0.5, 0.5
        self.direction = np.radians(95)
        self.speed     = 0.001
        if camera is not None:
            self.camera = camera
        else:
            self.camera = Camera(fov=60, resolution=64)
        self.maze       = maze
        self.colormap   = colormap
        self.max_energy = 5000
        self.energy     = self.max_energy
        self.source     = 1000
        
        self.hit_penalty   = -10
        self.move_penalty  = -1
        self.source_leak   = -1
        self.source_refill = +5


    def render(self):
        """ Rendering through camera. """
        
        self.camera.render(self.position, self.direction,
                           self.maze, self.colormap, outline=True)
        
    def is_legal(self, position):
        """
        Check whether a circular bot with given radius and position
        is colliding with any wall in the maze.

        Parameters
        ----------

        position : tuple of float
            (x, y) position of the robot center in normalized [0, 1] coordinates.

        Returns
        -------
        bool
            True if any part of the robot overlaps a wall, False otherwise.
        """

        x, y = position
        h, w = self.maze.shape
        cell_size = 1/max(self.maze.shape)
        radius = 1.05*self.radius

        imin = max(0, int((x-radius)/cell_size))
        imax = min(w-1, int((x+radius)/cell_size)) + 1
        jmin = max(0, int((y-radius)/cell_size))
        jmax = min(h-1, int((y+radius)/cell_size)) + 1    
        for j in range(jmin, jmax):
            for i in range(imin, imax):
                if self.maze[j, i] == 1:
                    cmin = np.array([i,j])*cell_size
                    cmax = cmin + cell_size
                    closest = np.maximum(cmin, np.minimum(position, cmax))
                    if np.linalg.norm(closest - position) <= radius:
                        return False
        return True


    def move_to(self, position):
        """
        Move the bot toward a positon using discrete steps and stop at first collision.

        Parameters
        ----------

        positon : tuple of float
            Target position (x, y) in normalized [0, 1] coordinates.

        Returns
        -------

        position : tuple of float
            Final position (x, y) before collision or at target.

        collision : bool
            True if a collision occurred during movement.
        """

        origin = np.array(self.position)
        target = np.array(position)
        direction = target - origin
        epsilon = 1e-4 # precision, this is necessary to avoid floating point problems.
        if np.allclose(direction, 0):
            return tuple(origin), False
        n = int(np.linalg.norm(direction) / epsilon)
        P = origin + np.arange(n).reshape(-1,1)*epsilon*direction
        P[-1] = position
        for i in range(n):
            if not self.is_legal(P[i]):
                return P[max(0,i-1)], True
        return position, False

    def forward(self, dtheta):
        """Move the bot by first re-orienting it by dtheta (degrees)
        and then making it to move forward.

        Parameters
        ----------

        dtheta : float
         Change in direction in radians, must be between -5° and +5° (in degrees)

        Returns
        -------
        
        (energy, hit, distance sensors, color sensors)
        """

        # If there is no energy left, run is ended
        if self.energy <= 0:
            return None

        # Try to move in the provided direction
        self.direction += max(np.radians(-5), min(np.radians(+5), dtheta))
        T = np.array([np.cos(self.direction), np.sin(self.direction)])
        self.position, self.hit = self.move_to(self.position + T*self.speed)
        
        # Decrease energy source
        self.source += self.source_leak

        # If the bot is on a source, refill energy
        x, y = self.position
        cell_size = 1/max(self.maze.shape)
        cx,cy = int(x/cell_size), int(y/cell_size)
        if self.maze[cy,cx] == -1:
            self.energy += max(0, self.source_refill)
            self.source -= max(0, self.source_refill)

        # Decrease energy by cost of move and possible hit penalty
        self.energy += self.move_penalty
        if self.hit:
            self.energy += self.hit_penalty

        # This is mostly for debug (first person view) but it also provides
        # an up to date sensor reading (this could be simplified to compute
        # only the sensors values)
        self.camera.render(self.position, self.direction, self.maze, self.colormap)

        # Return inputs as (energy, hit, depths, colors)
        return self.energy, self.hit, self.camera.depths, self.camera.values



# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    task = {
        "bot" : {
            "position" : (0.5,0.5),
            "direction": 90,
            "radius"   : 0.05,
            "speed"    : 0.01,
            "energy"   : 100,
            "camera"   : { "fov" :  60,
                           "resolution" : 64},
            "penalty"  : { "hit":   -3,
                           "move" : -1 }
        },
        "environment" : {
            "maze" :  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                       [1,-1,-1, 1, 0, 0, 1, 0, 0, 1],
                       [1,-1,-1, 1, 0, 0, 1, 0, 0, 1],
                       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            "colormap" : {
                -3 : [200, 200, 255], # ground (light blue)
                -2 : [255, 255, 255], # ground (normal)
                -1 : [100, 100, 255], # sky
                 0 : [255, 255, 255], # empty
                 1 : [200, 200, 200], # wall (light)
                 2 : [100, 100, 100], # wall (dark)
                 3 : [255, 255,   0], # wall (yellow)
                 4 : [  0,   0, 255], # wall (blue)
                 5 : [255,   0,   0], # wall (red)
                 6 : [  0, 255,   0] }, # wall (green)
            "source" : { "index" : (-1,-2),
                         "energy": 1000,
                         "leak" : -1,
                         "refill" : +5 }
        }
    }
        
