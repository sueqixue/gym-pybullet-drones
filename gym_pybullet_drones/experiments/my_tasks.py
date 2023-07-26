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
PP = 'pp'

"""
run_fly_task(
    -- Drone
    drone=DEFAULT_DRONES,
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
run_fly_task(src_pos=[0.5, 0.5, 1], dest_pos=[1, 2, 1.2], collision_avoidance=RRT)
