import gym
from gym import spaces
import numpy as np


class Soccer(gym.Env):


    def __init__(self):

        self.action_space = spaces.Box(
                                        low=np.array([-np.pi, 0], dtype=np.float32),
                                        high=np.array([np.pi, 1], dtype=np.float32),
                                        dtype=np.float32
                                      )
        low, high = np.array([-45, -30, -10, -10, -np.pi, -45, -30, -10, -10], dtype=np.float32), np.array([45, 30, 10, 10, np.pi, 45, 30, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
