"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Run $ python fly_task.py

Notes       The simulation is run by a `FLabCtrlAviary` environment.
            Let the drone take off and fly from a src position to a 
            dest position.
            Use to test different collision avoidance algorithms.

------------------------------------------------------------------------
Modified from examples/groundeffect.py by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

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
from traj_opt import *

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.FLabCtrlAviary import FLabCtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
# from gym_pybullet_drones.control.MPCControl import MPCControl
from gym_pybullet_drones.control.ModulationXYControl import ModulationXYControl
from gym_pybullet_drones.control.CBFXYControl import CBFXYControl


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
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_GD = True
DEFAULT_SRC_POS = [0, 0, 0]
DEFAULT_HOVER_POS = [0, 0, 1]
DEFAULT_DEST_POS = [1, 1, 1]

""" Control algorithms
        PID - DSLPIDControl()
        MOD2D - ModulationXYControl()
        CBF2D - CBFXYControl()
"""
PID = 'pid'
MOD2D = 'modulationXY'
CBF2D = 'cbfXY'
DEFAULT_CONTROL = PID

""" Collision avoidance algorithms
        none - no static collision avoidance
        rrt - rrt algorithm
        pp - potential field
"""
NONE = 'none'
RRT = 'rrt'
MPC = 'mpc'
DEFAULT_COLLISION_AVOIDANCE = NONE

# Debug boolens
PRINTING = False
RRT_PRINTING = False
TP_PRINTING = True

def run_fly_task_single(
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
        collision_avoidance=DEFAULT_COLLISION_AVOIDANCE,
        src_pos=DEFAULT_SRC_POS,
        hover_pos=DEFAULT_HOVER_POS,
        dest_pos=DEFAULT_DEST_POS,
        control=DEFAULT_CONTROL
        ):

    #### Initialize the simulation #############################
    TASK_INIT_XYZ = np.array(src_pos).reshape(1,3) 
    TASK_INIT_XY_Z0 = np.array([src_pos[0], src_pos[1], 0]).reshape(1,3)
    HOVER_XYZ = np.array(hover_pos).reshape(1,3) 
    TASK_DEST_XYZ = np.array(dest_pos).reshape(1,3) 
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

    #### Create the environment ################################
    env = FLabCtrlAviary(drone_model=drone,
                    num_drones=num_drones,
                    initial_xyzs=TASK_INIT_XY_Z0,
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
    if control == 'pid':
        ctrl = DSLPIDControl(drone_model=drone)
    elif control == 'modulationXY':
        ctrl = ModulationXYControl(drone_model=drone, env=env)
    elif control == 'cbfXY':
        ctrl = CBFXYControl(drone_model=drone)

    #### Drone Desired Trajectory ##############################
    TAKEOFF_PERIOD = 8
    TASK_PERIOD = 12
    HOVER_PERIOD = 4
    TOTAL_PERIOD = TAKEOFF_PERIOD + TASK_PERIOD + HOVER_PERIOD
    control_freq_hz = 240

    ENV_TOTAL_WP = TOTAL_PERIOD * control_freq_hz

    # Get the origin pos and ori of the drone, since they might be changed during collision detection
    drone_origin_pos, drone_origin_ori = p.getBasePositionAndOrientation(env.DRONE_IDS[0], env.CLIENT)
    drone_sim_origin_pos = drone_origin_pos
    drone_sim_origin_ori = drone_origin_ori

    ASSIST = False
    if not (TASK_INIT_XYZ[0][2] == 0):
        # If the drone need to take off before doing the task, taking it off first
        print("\nHelp to take off the drone.\n")
        ASSIST = True
        TARGET_POS_prep, NUM_WP_prep = traj_opt(INIT_XYZ=TASK_INIT_XY_Z0, 
                                                HOVER_XYZ=HOVER_XYZ, 
                                                DEST_XYZ=TASK_INIT_XYZ, 
                                                drone_origin_pos=drone_origin_pos, 
                                                drone_origin_ori=drone_origin_ori, 
                                                env=env,
                                                GROUND_EFFECT=ground_effect,
                                                TAKEOFF_PERIOD=TAKEOFF_PERIOD,
                                                TASK_PERIOD=TASK_PERIOD,
                                                HOVER_PERIOD=HOVER_PERIOD,
                                                control_freq_hz=control_freq_hz,
                                                collision_avoidance=collision_avoidance,
                                                take_off_flag=True)
        drone_origin_pos, drone_origin_ori = p.getBasePositionAndOrientation(env.DRONE_IDS[0], env.CLIENT)     

    TARGET_POS, NUM_WP = traj_opt(INIT_XYZ=TASK_INIT_XYZ, 
                                HOVER_XYZ=HOVER_XYZ, 
                                DEST_XYZ=TASK_DEST_XYZ, 
                                drone_origin_pos=drone_origin_pos, 
                                drone_origin_ori=drone_origin_ori, 
                                env=env,
                                GROUND_EFFECT=ground_effect,
                                TAKEOFF_PERIOD=TAKEOFF_PERIOD,
                                TASK_PERIOD=TASK_PERIOD,
                                HOVER_PERIOD=HOVER_PERIOD,
                                control_freq_hz=control_freq_hz,
                                collision_avoidance=collision_avoidance,
                                take_off_flag=(not ASSIST))
    
    if ASSIST:
        if PRINTING:
            print(f"\nTARGET_POS_prep.shape = {TARGET_POS_prep.shape}")
            print(f"TARGET_POS_task.shape = {TARGET_POS.shape}")
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], drone_sim_origin_pos, drone_sim_origin_ori, env.CLIENT)
        TARGET_POS = np.append(TARGET_POS_prep, TARGET_POS, axis = 0)
        NUM_WP += NUM_WP_prep
 
    if PRINTING:
        print(f"NUM_WP = {NUM_WP}")
        print(f"ENV_TOTAL_WP = {ENV_TOTAL_WP}")
    
    if NUM_WP < ENV_TOTAL_WP:
        END_HOVER_WP = ENV_TOTAL_WP - NUM_WP
        TARGET_POS = np.append(TARGET_POS, np.zeros((END_HOVER_WP,3)), axis = 0)
        for i in range(NUM_WP, ENV_TOTAL_WP):
            TARGET_POS[i, :] = TARGET_POS[NUM_WP-1, :]

            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i}, :] = {TARGET_POS[i, :]}")
        
        NUM_WP = ENV_TOTAL_WP

    if PRINTING:
        print(f"\nTARGET_POS.shape = {TARGET_POS.shape}")
        print(f"NUM_WP = {NUM_WP}")

    #### Dynamic Obstacles #####################################
    num_dy_obstacles = 2
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS_OBS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_dy_obstacles)])
    INIT_RPYS_OBS  = np.array([[0, 0,  i * (np.pi/2)/num_dy_obstacles] for i in range(num_dy_obstacles)])

    # Circular motion
    TARGET_POS_OBS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS_OBS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS_OBS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS_OBS[0, 1], 0
 
    #### Run the simulation ####################################
    wp_counter = 0

    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {"0": np.array([0,0,0,0])}
    START = time.time()
    # for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    for i in range(0, NUM_WP, AGGR_PHY_STEPS):

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
    # parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,         type=int,           help='Number of drones (default: 3)', metavar='')
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
    parser.add_argument('--src_pos',            default=DEFAULT_SRC_POS)
    parser.add_argument('--hover_pos',          default=DEFAULT_HOVER_POS)
    parser.add_argument('--dest_pos',           default=DEFAULT_DEST_POS)
    parser.add_argument('--control',            default=DEFAULT_CONTROL)
    ARGS = parser.parse_args()

    run_fly_task_single(**vars(ARGS))
