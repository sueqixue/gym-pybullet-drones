"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Run $ python my_tasks.py

Notes       Let the drone do some tasks in designed order.
            Use to test different collision avoidance algorithms.

------------------------------------------------------------------------
Implemented by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

from fly_task import run_fly_task_single

""" Control algorithms
        PID - DSLPIDControl()
        MOD2D - ModulationXYControl()
"""
PID = 'pid'
MOD2D = 'modulationXY'

""" Collision avoidance algorithms
        none - no static collision avoidance
        rrt - rrt algorithm
        pp - potential field
"""
NONE = 'none'
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
    dest_pos=DEFAULT_DEST_POS,

    -- Control parameters
    control=DEFAULT_CONTROL
    )
"""
CNTL = PID
# CNTL = MOD2D
STATIC_COLLISION_AVOIDANCE = RRT

start  = [-1.0, -0.5, 0.7]
end    = [1.0, 2.0, 1.0]

run_fly_task_single(src_pos=start, dest_pos=end, collision_avoidance=STATIC_COLLISION_AVOIDANCE, control=CNTL)
