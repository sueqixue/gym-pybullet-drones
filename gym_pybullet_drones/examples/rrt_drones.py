import numpy as np
import random

class Node:
    def __init__(self, q, parentIdx=None):
        self.q = q
        self.parentIdx = parentIdx

################################################################################

def isWithinLimit(pos, lowerLim, upperLim):
    return np.all((lowerLim < pos) & (pos <= upperLim))

################################################################################

def isCollided(q, obstacles):
    # TODO: find how to extract the obstacles data from the env struct, and check
    #       how the data is stored. Then implement the isCollided().
    return True

################################################################################

def randomFreePos(lowerLim, upperLim, obstacles):
    while True:
        q_random = np.random.uniform(lowerLim, upperLim)
        if not isCollided(q_random, obstacles):
            break

    return q_random

################################################################################

def getClosestNode(q, nodeList):
    len_nodeList = len(nodeList)
    diff = 1000000
    q_closest_idx = -1

    for i in range(len_nodeList):
        this_diff = np.linalg.norm(q - nodeList[i].q)
        if this_diff < diff:
            diff = this_diff
            q_closest_idx = i

    return q_closest_idx

################################################################################

def rrt(env, start, goal):
    """
    RRT algorithm
    :param env:         the environment struct
    :param start:       start position of the drone (0x3)
    :param goal:        goal position of the drone (0x3)
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
    n_iter = 500
    upperLim = np.array([0, 0, 2]).reshape(1,3)    # Ceiling height - maybe camera height
    lowerLim = np.array([0, 0, 0.2]).reshape(1,3)  # Lower limit to avoid ground effect

    # Check if the start and goal is within the limits
    check_start = not isWithinLimit(start, lowerLim, upperLim)
    check_goal = not isWithinLimit(goal, lowerLim, upperLim)  
    if check_start or check_goal:
        return path

    # Loading the obstacles
    obstacles = env.obstacles

    # Check if there is any obstacles at the start and goal positions
    if isCollision(start) or isCollision(goal):
        return path

    # Initialize the nodelist for start and goal
    T_start = [Node(start)]
    T_goal = [Node(goal)]

    # Finding the path
    for i in range(n_iter):
        q = randomFreePos(lowerLim, upperLim, obstacles)    # Random configuration in Q_free

        idx_q_a = getClosestNode(q, T_start)                # Get closest node in T_start
        q_a = T_start[idx_q_a].q                            # Get point q_a
        q_a_flag = isCollided(q_a, q, obstacles)            # Check if edge qq_a collide with obstacles

        if not q_a_flag:                                    # Add (q, q_a) to T_start
            T_start.append(Node(q, idx_q_a))

        idx_q_b = getClosestNode(q, T_goal)                 # Get closest node in T_gaol
        q_b = T_goal[idx_q_b].q                             # Get point q_b
        q_b_flag = isCollided(q_b, q, obstacles)            # Check if edge qq_b collide with obstacles

        if not q_b_flag:                                    # Add (q_b, q) to T_goal
            T_goal.append(Node(q, idx_q_b))

        if not q_a_flag and not q_b_flag:                   # No collisions between qq_a and qq_b
            # Connect the nodes from T_start
            curr_node = T_start[-1]
            path_start = [curr_node.q]

            while curr_node.parentIdx is not None:
                curr_node = T_start[curr_node.parentIdx]
                path_start.append(curr_node.q)

            # Connect the nodes from T_goal
            curr_node = T_goal[-1]                            # T_goal[-1] = T_start[-1]
            path_goal = [curr_node.q]

            while curr_node.parentIdx is not None:
                curr_node = T_goal[curr_node.parentIdx]
                path_goal.append(curr_node.q)

            # Get the path
            path = np.array(path_start[::-1] + path_goal[1:])
            return path

    return []