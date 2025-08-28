from bot import Bot
from environment_1 import Environment

class PlayerDDPG:
    def __init__(self):
        # Number of hidden neurons
        # Number of actions (decided to have 2 discrete actions --> move left (turn by -1°)/ move right (turn by +1°)/ stay clear (0° = no turning))
        self.n_actions = 3
        # Number of observations
        self.n_observations = 64
        # Target Network
        # Value Network
        # Loss function
        # Replay buffer
#-------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np
    from challenge import train, evaluate

    #Training (100 seconds)