"""The simulation is run by a `FLabCtrlAviary` environment.

Example
-------
In a terminal, run as:

    $ python fly_to_dest_safe.py

Notes
-----
Let the drone take off and fly from a src position to a dest position.
Use to test different collision avoidance algorithms.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from copy import deepcopy
from rrt_drones import *

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.FLabCtrlAviary import FLabCtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics.PYB_GND ## or Physics.PYB for comparision
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_DURATION_SEC = 24
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_GD = False

""" Collision avoidance algorithms
    none - dummy trajectory
    rrt - rrt algorithm
    pp - potential field
"""
DUMMY = 'none'
RRT = 'rrt'
PP = 'pp'
DEFAULT_COLLISION_AVOIDANCE = RRT

# Debug boolens
PRINTING = False
RRT_PRINTING = False
TP_PRINTING = True

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        ground_effect=DEFAULT_GD,
        collision_avoidance=DEFAULT_COLLISION_AVOIDANCE
        ):

    #### Initialize the simulation #############################
    INIT_XYZ = np.array([0, 0, 0]).reshape(1,3)
    HOVER_XYZ = np.array([0, 0, 1]).reshape(1,3)
    DEST_XYZ = np.array([1, 1, 1]).reshape(1,3)
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

    #### Create the environment ################################
    env = FLabCtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZ,
                     physics=physics,
                     neighbourhood_radius=10,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui
                     )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controller #############################
    ctrl = DSLPIDControl(drone_model=drone)

    #### Initialize a desired trajectory ######################
    """
    TODO: taking off the drone to the hover xyz and move to the dest position.
        Note that TARGET_POS is an array contains the position at each sampling
        point, which means it contains the infomation of the desired trajectory/
        path of the current task. So when implementing any collision avoidance
        algorithms/controller, use the input src and dest position to find the
        optimal trajectory/path.
    """
    TAKE_OFF_PAR = 0.15*3
    TAKE_OFF_H = 1
    HOVER_FLAG = False
    GROUND_EFFECT = ground_effect

    TAKEOFF_PERIOD = 8
    TASK_PERIOD = 12
    HOVER_PERIOD = 4
    NUM_WP_TAKEOFF = control_freq_hz*TAKEOFF_PERIOD
    NUM_WP_TASK = control_freq_hz*TASK_PERIOD
    NUM_WP_HOVER = control_freq_hz*HOVER_PERIOD
    NUM_WP = NUM_WP_TAKEOFF + NUM_WP_TASK + NUM_WP_HOVER

    TARGET_POS = np.zeros((NUM_WP,3))
    
    if GROUND_EFFECT:
        print(f"\n---------- TAKE OFF WITH GROUND EFFECT ----------\n")
    else:
        print(f"\n---------- TAKE OFF WITHOUT GROUND EFFECT ----------\n")

    # Take off
    for i in range(NUM_WP_TAKEOFF):
        if GROUND_EFFECT:
            if not HOVER_FLAG:
                TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], INIT_XYZ[0, 2] + TAKE_OFF_PAR * (np.sin((i/NUM_WP_TAKEOFF)*(2*np.pi)) + 1)
            else:
                TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], TARGET_POS[i-1, 2]

            if TARGET_POS[i, 2] < TARGET_POS[i-1, 2]:
                HOVER_FLAG = True
        else:
            TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], INIT_XYZ[0, 2] + i * (TAKE_OFF_H/NUM_WP_TAKEOFF)
        
        if TP_PRINTING:
            if i % 100 == 0:
                print(f"TARGET_POS[{i}, :] = {TARGET_POS[i, :]}")
    # print(f"TARGET_POS[{0}, {0}] = {TARGET_POS[0, 0]}")
   
    # Finding the path to the destination
    task_start = TARGET_POS[NUM_WP_TAKEOFF-1, :].reshape(1,3)
    task_goal = np.array([1, -1, 1.2]).reshape(1,3)
    task_path = []
    smooth_task_path = []
    if collision_avoidance == 'rrt':
        print(f"\n---------- RRT ----------\n")
        task_path = rrt(deepcopy(env), deepcopy(task_start), deepcopy(task_goal), NUM_WP_TASK)
        task_path_len = len(task_path)
        if PRINTING:
            print(f"task_path = {task_path}")
            print(f"RRT path length = {task_path_len}")
            print(f"RRT path shape = {task_path.shape}")
        
        # Smooth the path
        smooth_task_path = np.zeros((NUM_WP_TASK,3))
        if RRT_PRINTING:
            print(f"smooth_task_path shape is {smooth_task_path.shape}")
        if task_path_len < NUM_WP_TASK:
            part_len = NUM_WP_TASK // (task_path_len - 1)
            if RRT_PRINTING:
                print(f"part_len = {part_len}\n")
                
            for i in range(task_path_len-1):
                if RRT_PRINTING:
                    print(f"section {i}:\n")
                
                sec_index = i*part_len
                smooth_task_path[sec_index] = task_path[i]
                if RRT_PRINTING:
                    print(f"smooth_task_path[{sec_index}] = {smooth_task_path[sec_index]}")
                
                for j in range(1, part_len):
                    step_diff_x = (task_path[i+1, 0] - task_path[i, 0]) / part_len
                    step_diff_y = (task_path[i+1, 1] - task_path[i, 1]) / part_len
                    step_diff_z = (task_path[i+1, 2] - task_path[i, 2]) / part_len
                    smooth_task_path[sec_index+j] = smooth_task_path[sec_index+j-1] + [step_diff_x, step_diff_y, step_diff_z]
                    if RRT_PRINTING:
                        if sec_index+j % 100 == 0:
                            print(f"smooth_task_path[{sec_index+j}] = {smooth_task_path[sec_index+j]}")

    if GROUND_EFFECT:
        print(f"\n---------- FLY TO DESTINATION WITH GROUND EFFECT ----------\n")
    else:
        print(f"\n---------- FLY TO DESTINATION WITHOUT GROUND EFFECT ----------\n")

    if collision_avoidance == 'none':
        # [DEBUG]: Generating the path to the destination without collision avoidance
        print(f"\n---------- DEBUG TRAJ ----------\n")
        for i in range(NUM_WP_TASK):
            TARGET_POS[i+NUM_WP_TAKEOFF, :] = TARGET_POS[NUM_WP_TAKEOFF-1, 0] + i * (TAKE_OFF_H/NUM_WP_TASK), TARGET_POS[NUM_WP_TAKEOFF-1, 1], TARGET_POS[NUM_WP_TAKEOFF-1, 2]
            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF, :]}")
    elif collision_avoidance == 'rrt':
        for i in range(NUM_WP_TASK):
            if i < len(smooth_task_path):
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = smooth_task_path[i]
            elif len(smooth_task_path) > 0:
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = smooth_task_path[-1]
            else:
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = TARGET_POS[NUM_WP_TAKEOFF-1, :]

            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF}, :] = TASK_POS[{i}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF, :]}")
    
    if TP_PRINTING:
        print(f"TARGET_POS[{NUM_WP_TASK+NUM_WP_TAKEOFF-1}, :] = {TARGET_POS[NUM_WP_TASK+NUM_WP_TAKEOFF-1, :]}")

    if GROUND_EFFECT:
        print(f"\n---------- HOVER WITH GROUND EFFECT ----------\n")
    else:
        print(f"\n---------- HOVER WITHOUT GROUND EFFECT ----------\n")

    for i in range(NUM_WP_HOVER):
        TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK, :] = TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK-1, :]

        if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF+NUM_WP_TASK}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK, :]}")

        
    
    wp_counter = 0

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {"0": np.array([0,0,0,0])}
    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action["0"], _, _ = ctrl.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                             state=obs["0"]["state"],
                                                             target_pos=TARGET_POS[wp_counter, :],
                                                             )

            #### Go to the next way point and loop #####################
            wp_counter = wp_counter + 1 if wp_counter < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state= obs["0"]["state"],
                   control=np.hstack([TARGET_POS[wp_counter, :], np.zeros(9)])
                   )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("gnd") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Hover script with or without ground effect using FLabCtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,             type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,         type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,            type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,                type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,       type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,               type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,     type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,          type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,          type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,    type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,       type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,      type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,              type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--ground_effect',      dest='ground_effect',       action='store_true')
    parser.add_argument('--no_ground_effect',   dest='ground_effect',       action='store_false')
    parser.set_defaults(ground_effect=False)
    parser.add_argument('--collision_avoidance',default=DEFAULT_COLLISION_AVOIDANCE,type=str,           help='Which collision avoidance algorithm to apply (default: "none")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
