"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Run $ python my_tasks.py

Notes       Let the drone do some tasks in designed order.
            Use to test different collision avoidance algorithms.

------------------------------------------------------------------------
Implemented by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

from fly_task import *

""" Collision avoidance algorithms
        none - dummy trajectory
        rrt - rrt algorithm
        pp - potential field
"""
DUMMY = 'none'
RRT = 'rrt'
MPC = 'mpc'

"""
run_fly_task_single(
    -- Drone
    drone=DEFAULT_DRONES=1,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,

    -- GUI
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VIDEO,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    aggregate=DEFAULT_AGGREGATE,

    -- Obstacles
    obstacles=DEFAULT_OBSTACLES,

    -- Simulation frequncy
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,

    -- Log
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,

    -- Task parameters
    ground_effect=DEFAULT_GD,
    collision_avoidance=DEFAULT_COLLISION_AVOIDANCE,
    src_pos=DEFAULT_SRC_POS,
    hover_pos=DEFAULT_HOVER_POS,
    dest_pos=DEFAULT_DEST_POS
    )
"""
TEST_TAKE_OFF_ASSIST = False

if TEST_TAKE_OFF_ASSIST:
    run_fly_task_single(src_pos=[1.5, 1.5, 1.2], dest_pos=[1.0, 1.0, 2.0], collision_avoidance=RRT)
else:
    run_fly_task_single(src_pos=[0.0, 0.0, 0.0], dest_pos=[0.0, 0.0, 2.0], collision_avoidance=RRT)
