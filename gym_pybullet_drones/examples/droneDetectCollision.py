import numpy as np
import pybullet as p

def droneDetectCollision(physics_client_id, drone_id, obstacles_id_list, drone_pos):
    """
    Check if the drone position is collided with any obstacle in the
    obstacles list.
    :param physics_client_id: int which is current pybullet phsics client id
    :param drone_id: int which is the pybullet id of the drone
    :param obstacle_id_list: array of int which contains the pybullet id of all obstacles
    :param drone_pos: the future/potential positin of the drone
    "return: boolean, true if the drone is in collision with any obstacle
    """
    print(f"Start collision detection with physics_client_id = {physics_client_id}\n")
    print(f"drone_id = {drone_id[0]}\n")
    p.performCollisionDetection(physics_client_id)

    n_obstacles = len(obstacles_id_list)
    for i in range(n_obstacles):
        print(f"Checking obstacle {i}, id = {obstacles_id_list[i]}\n")
        if detectCollisionEachObstacle(physics_client_id, drone_id[0], obstacles_id_list[i], drone_pos):
            return True
    return False

def detectCollisionEachObstacle(physics_client_id, drone_id, obstacle_id, drone_pos):
    """
    Check if the drone position is collided with any obstacle in the
    obstacles list.
    :param physics_client_id: int which is current pybullet phsics client id
    :param drone_id: int which is the pybullet id of the drone
    :param obstacle_id: int which is the pybullet id of the obstacle
    :param drone_pos: the future/potential positin of the drone
    "return: boolean, true if the drone is in collision with this obstacle
    """
    print(f"Calling getContactPoints...\n")
    contact_points_list = p.getContactPoints(bodyA=drone_id,
                   bodyB=obstacle_id,
                   physicsClientId=physics_client_id
                   )

    if contact_points_list:
        return True
    return False
    