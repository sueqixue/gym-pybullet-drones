"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Call rrt()

Notes       Drone RRT algorithm.

------------------------------------------------------------------------
Implemented by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

import numpy as np
import pybullet as p
import random
from droneDetectCollision import droneDetectCollision

# Debug boolens
PRINTING = False

class Node:
    def __init__(self, pos, parentIdx=None):
        self.pos = pos
        self.parentIdx = parentIdx

################################################################################

def isWithinLimit(pos, lowerLim, upperLim):
    return np.all((lowerLim < pos) & (pos <= upperLim))

################################################################################

# Old naive collision detection which only check whether a pos is collided with
# the base position of obstacles
def isPosCollided(pos, obstacles, safe_dist=0.1):
    for obstacle in obstacles:
        obstacle_pos = obstacle[0]
        dist = np.linalg.norm(pos - obstacle_pos)

        if PRINTING:
            print(f"dist = {dist} and safe_dist = {safe_dist}")
            
        if dist < safe_dist:
            return True
    return False

################################################################################

# def isPathCollided(pos_end, pos_start, obstacles, num_steps=500):
#     step = (pos_end - pos_start) / num_steps
#     for i in range(1, 1 + num_steps):
#         if isPosCollided(pos_start + (i * step), obstacles):
#             return True
#     return False

################################################################################

def isPathCollided(pos_end, pos_start, physics_client_id, drone_id, obstacles_id, num_steps=500):
    step = (pos_end - pos_start) / num_steps
    for i in range(1, 1 + num_steps):
        if droneDetectCollision(physics_client_id, drone_id, obstacles_id, pos_start + (i * step)):
            return True
    return False

################################################################################

def randomFreePos(lowerLim, upperLim, physics_client_id, drone_id, obstacles_id,):
    while True:
        pos_random = np.random.uniform(lowerLim, upperLim)
        if not droneDetectCollision(physics_client_id, drone_id, obstacles_id, pos_random):
            break

    return pos_random

################################################################################

def getClosestNode(pos, nodeList):
    len_nodeList = len(nodeList)
    diff = 1000000
    pos_closest_idx = -1

    for i in range(len_nodeList):
        this_diff = np.linalg.norm(pos - nodeList[i].pos)
        if this_diff < diff:
            diff = this_diff
            pos_closest_idx = i

    return pos_closest_idx

################################################################################

def prunedPath(path, physics_client_id, drone_id, obstacles_id):
    sub_paths = []

    # MPC
    for i in range(len(path) - 2):
        sub_path = path
        for j in range(i + 2, len(path)):
            if not isPathCollided(path[i], path[j], physics_client_id, drone_id, obstacles_id):
                sub_path = np.vstack((path[:i+1], path[j:]))
        sub_paths.append(sub_path)

    costs = np.array([np.linalg.norm(p[1:] - p[:-1]).sum() for p in sub_paths])
    pruned_path = sub_paths[np.argmin(costs)]

    return pruned_path

################################################################################

def rrt(env, start, goal, num_iter=500):
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

    # Initialize the path
    path = []
    path_start = []
    path_goal = []

    # Drone workspace limits
    upperLim = np.array([5, 5, 5]).reshape(1,3)    # Ceiling height - maybe camera height
    lowerLim = np.array([-5, -5, 0]).reshape(1,3)  # Lower limit to avoid ground effect

    if PRINTING:
        print(f"--------------------\nstart = {start}\ngoal = {goal}\n--------------------")

    # Check if the start and goal is within the limits
    check_start = not isWithinLimit(start, lowerLim, upperLim)
    check_goal = not isWithinLimit(goal, lowerLim, upperLim)  
    if check_start or check_goal:
        print("src or dest out of limits.\n")
        return path

    # Loading the obstacles and drone
    obstacles = env.obstacles_list
    obstacles_id = env.obstacles_id_list
    drone_id = env.DRONE_IDS
    physics_client_id = env.CLIENT
    if PRINTING:
        print(f"There is {len(obstacles)} obstacles in the environment.\n--------------------")
        for j in range(len(obstacles)):
            print(f"obstacles_{j} = {obstacles[j]}\n")

    # Testing droneDetectCollision method
    # res = droneDetectCollision(physics_client_id, drone_id, obstacles_id, start)
    # print(f"{res}\n")

    # Check if there is any obstacles at the start and goal positions
    # if isPosCollided(start, obstacles) or isPosCollided(goal, obstacles):
    if (droneDetectCollision(physics_client_id, drone_id, obstacles_id, start) or 
        droneDetectCollision(physics_client_id, drone_id, obstacles_id, goal)):
        return path

    # Initialize the nodelist for start and goal
    T_start = [Node(start)]
    T_goal = [Node(goal)]
    
    # Finding the path
    for i in range(num_iter):
        pos = randomFreePos(lowerLim, upperLim, physics_client_id, drone_id, obstacles_id)            

        idx_pos_a = getClosestNode(pos, T_start)                                                        
        pos_a = T_start[idx_pos_a].pos                                                                       
        pos_a_flag = isPathCollided(pos_a, pos, physics_client_id, drone_id, obstacles_id)        

        if not pos_a_flag:                                                                          
            T_start.append(Node(pos, idx_pos_a))

        idx_pos_b = getClosestNode(pos, T_goal)                                                          
        pos_b = T_goal[idx_pos_b].pos                                                                   
        pos_b_flag = isPathCollided(pos_b, pos, physics_client_id, drone_id, obstacles_id)      

        if not pos_b_flag:                                                                           
            T_goal.append(Node(pos, idx_pos_b))

        if not pos_a_flag and not pos_b_flag:                                                           
            # Connect the nodes from T_start
            curr_node = T_start[-1]
            path_start = [curr_node.pos]

            while curr_node.parentIdx is not None:
                curr_node = T_start[curr_node.parentIdx]
                path_start.append(curr_node.pos)

            # Connect the nodes from T_goal
            curr_node = T_goal[-1]                                                                      # T_goal[-1] = T_start[-1]
            path_goal = [curr_node.pos]

            while curr_node.parentIdx is not None:
                curr_node = T_goal[curr_node.parentIdx]
                path_goal.append(curr_node.pos)

            # Get the path
            path = np.array(path_start[::-1] + path_goal[1:])
            path = path.reshape(path.shape[0], 3)
            print(f"path = {path}")

            pruned_path = prunedPath(path, physics_client_id, drone_id, obstacles_id)
            print(f"pruned_path = {pruned_path}")

            if PRINTING:
                print(f"path shape = {path.shape}")
            return pruned_path

    return []