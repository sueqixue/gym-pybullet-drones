"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Call mpc()

Notes       Drone MPC algorithm.

------------------------------------------------------------------------
Implemented by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

import numpy as np
import pybullet as p
import random
from droneDetectCollision import droneDetectCollision

# Debug boolens
PRINTING = False

################################################################################

def mpc(env, start, goal, num_iter=500):
    """
    RRT algorithm
    :param env:         the environment struct
    :param start:       start position of the drone (0x3)
    :param goal:        goal position of the drone (0x3)
    :param n_iter:      number of iteration for finding the path
    :return:            return an mx3 matrix, where each row contain the target
                        position of the drone in the path. The first row is start
                        and the last row is goal. If no path is found, return
                        empty PATH.
    """

    

