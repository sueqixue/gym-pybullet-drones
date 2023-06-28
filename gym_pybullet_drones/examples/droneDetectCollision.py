import numpy as np

def droneDetectCollision(drone_id, obstacle_id_list, drone_pos):
    """
    Check if the drone position is collided with any obstacle in the
    obstacles list.
    :param drone_id: int which is the pybullet id of the drone
    :param obstacle_id_list: array of int which contains the pybullet id of all obstacles
    :param drone_pos: the future/potential positin of the drone
    "return: boolean, true if the drone is in collision with any obstacle
    """
    n_obstacles = len(obstacle_id_list)

    for i in range(n_obstacles):
        if detectCollisionEachObstacle(drone_id, obstacle_id_list[i], drone_pos):
            return True

    return False

def detectCollisionEachObstacle(drone_id, obstacle_id, drone_pos):
    """
    Check if the drone position is collided with any obstacle in the
    obstacles list.
    :param drone_id: int which is the pybullet id of the drone
    :param obstacle_id: int which is the pybullet id of the obstacle
    :param drone_pos: the future/potential positin of the drone
    "return: boolean, true if the drone is in collision with this obstacle
    """
    
    